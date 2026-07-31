from database import BM25Index

class HybridRetriever:
    """
    Unified interface for hybrid indexing and retrieval:
    - Indexes into both ChromaDB (semantic vectors) and SQLite FTS5 (BM25 keywords).
    - Merges and normalizes search results from both engines using a weighted sum (alpha).
    """
    def __init__(self, db_session, vector_store=None):
        if vector_store is None:
            from database import VectorStore
            vector_store = VectorStore()
        self.vector_store = vector_store
        self.bm25_index = BM25Index(db_session)

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str] = None):
        """
        Unified indexing interface:
        Adds documents to BOTH the ChromaDB VectorStore and the SQLite FTS5 BM25 index at a single place.
        """
        if not ids:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
            
        # 1. Index into ChromaDB VectorStore
        self.vector_store.add_documents(documents, metadatas, ids)
        # 2. Index into SQLite FTS5 BM25 table
        self.bm25_index.add_documents(documents, ids)

    def ensure_indexed(self) -> bool:
        """
        Ensures both the semantic VectorStore and lexical BM25 index are populated.
        Returns True if documents exist in the database, False otherwise.
        """
        return self.bm25_index.load_or_build(self.vector_store)

    def hybrid_search(self, query: str, top_k: int = None, alpha: float = None) -> list[dict]:
        """
        Combines results from Vector Search and BM25.
        alpha = 1.0 means pure vector search.
        alpha = 0.0 means pure bm25.
        """
        from config import get_config
        config = get_config()["retrieval"]
        if top_k is None:
            top_k = config["hybrid_top_k"]
        if alpha is None:
            alpha = config["hybrid_alpha"]

        vector_results = self.vector_store.search(query, top_k=top_k)
        bm25_results = self.bm25_index.search(query, top_k=top_k)
        
        # Min-max normalize scores to combine them properly
        def normalize(results):
            if not results:
                return []
            scores = [r['score'] for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                for r in results: 
                    r['norm_score'] = 1.0
            else:
                for r in results: 
                    r['norm_score'] = (r['score'] - min_s) / (max_s - min_s)
            return results
            
        vec_norm = normalize(vector_results)
        bm25_norm = normalize(bm25_results)
        
        # Merge results using a weighted sum (alpha)
        combined_scores = {}
        merged_docs = {}
        
        for res in vec_norm:
            doc_id = res['id']
            merged_docs[doc_id] = res
            combined_scores[doc_id] = alpha * res['norm_score']
            
        for res in bm25_norm:
            doc_id = res['id']
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * res['norm_score']
            else:
                merged_docs[doc_id] = res
                combined_scores[doc_id] = (1 - alpha) * res['norm_score']
                
        # Sort by combined score
        sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
        
        final_results = []
        for doc_id in sorted_ids:
            doc = merged_docs[doc_id]
            doc['hybrid_score'] = combined_scores[doc_id]
            final_results.append(doc)
            
        return final_results
