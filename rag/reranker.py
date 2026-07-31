from sentence_transformers import CrossEncoder
from config import get_config
from config.logger import setup_logger

logger = setup_logger("LocalReranker")

class LocalReranker:
    def __init__(self, model_name=None):
        from config import DEVICE
        config = get_config()
        self.model_name = model_name or config["retrieval"]["reranker_model"]
        logger.info(f"Loading reranker model '{self.model_name}' on device: {DEVICE.upper()}...")
        self.model = CrossEncoder(self.model_name, device=DEVICE)
        
    def rerank(self, query: str, documents: list[dict], top_k: int = None) -> list[dict]:
        """
        Re-ranks a list of candidate documents based on a query using a Cross-Encoder.
        documents should be a list of dicts containing at least a 'content' key.

        How it works:
        - The Cross-Encoder reads each [query, doc_content] pair side-by-side.
        - It outputs a numerical similarity score (higher = better match, e.g., 5.84 vs -3.12).
        - We sort the documents from highest score to lowest and return the top_k best results.
        """
        if top_k is None:
            top_k = get_config()["retrieval"]["rerank_top_k"]
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} candidate documents using '{self.model_name}' (top_k={top_k})...")
        # 1. Prepare pairs of [query, doc_content] for the Cross-Encoder to inspect together
        pairs = [[query, doc.get('content', '')] for doc in documents]
        
        # 2. Predict relevance scores on local GPU (higher = better match)
        scores = self.model.predict(pairs)
        
        # 3. Attach the numerical score to each document dictionary
        for idx, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[idx])
            
        # 4. Sort by rerank score descending (highest scoring document first)
        sorted_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_docs[:top_k]
