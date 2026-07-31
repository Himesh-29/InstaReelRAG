import re
from sqlalchemy import text
from config import get_config
from config.logger import setup_logger

logger = setup_logger("BM25Index")

class BM25Index:
    """
    Manages SQLite FTS5 (Full-Text Search 5) virtual table for Okapi BM25 keyword ranking.
    Separated from HybridRetriever so keyword search logic is encapsulated in the database layer,
    alongside VectorStore (ChromaDB) and Metadata DB (SQLAlchemy ORM).
    """
    def __init__(self, db_session):
        self.db_session = db_session
        self.engine = db_session.get_bind()
        self._init_fts_table()

    def _init_fts_table(self):
        """Initializes the SQLite FTS5 virtual table for Okapi BM25 ranking."""
        with self.engine.connect() as conn:
            conn.exec_driver_sql("""
                CREATE VIRTUAL TABLE IF NOT EXISTS bm25_fts USING fts5(
                    id UNINDEXED,
                    content
                );
            """)
            conn.commit()

    def add_documents(self, documents: list[str], ids: list[str]):
        """Incrementally indexes new documents into the SQLite FTS5 BM25 index."""
        with self.engine.connect() as conn:
            for doc_id, doc_content in zip(ids, documents):
                conn.exec_driver_sql("DELETE FROM bm25_fts WHERE id = ?", (doc_id,))
                conn.exec_driver_sql(
                    "INSERT INTO bm25_fts (id, content) VALUES (?, ?)",
                    (doc_id, doc_content)
                )
            conn.commit()
            logger.info(f"Indexed {len(documents)} documents into SQLite FTS5 BM25 table.")

    def load_or_build(self, vector_store) -> bool:
        """
        Checks if the SQLite FTS5 index document count matches the VectorStore count.
        If empty or out-of-sync, populates it from the VectorStore.
        """
        vec_count = vector_store.count()
        if vec_count == 0:
            logger.info("VectorStore is empty. No documents to index into BM25.")
            return False

        with self.engine.connect() as conn:
            bm25_count = conn.exec_driver_sql("SELECT COUNT(*) FROM bm25_fts").scalar() or 0
            if bm25_count == vec_count:
                logger.info(f"BM25 FTS5 table count ({bm25_count}) matches VectorStore count ({vec_count}). Index is ready.")
                return True
            else:
                logger.info(f"BM25 FTS5 table count ({bm25_count}) differs from VectorStore count ({vec_count}). Syncing BM25 index from VectorStore...")
                
        # Sync FTS5 table from vector store
        all_docs = vector_store.collection.get()
        if all_docs and all_docs.get("documents"):
            self.add_documents(all_docs["documents"], all_docs["ids"])
            return True
        return False

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Executes Okapi BM25 keyword search using SQLite FTS5.
        Returns a list of dicts with keys: 'id', 'content', 'score', 'source'.
        """
        if top_k is None:
            top_k = get_config()["retrieval"]["hybrid_top_k"]
        words = [w for w in re.findall(r'\w+', query) if len(w) > 1]
        if not words:
            return []
            
        fts_query = " OR ".join(words)
        results = []
        try:
            with self.engine.connect() as conn:
                cursor = conn.exec_driver_sql(
                    """
                    SELECT id, content, bm25(bm25_fts) as raw_score
                    FROM bm25_fts
                    WHERE bm25_fts MATCH ?
                    ORDER BY raw_score ASC
                    LIMIT ?
                    """,
                    (fts_query, top_k)
                )
                for row in cursor:
                    # SQLite bm25() returns negative scores where more negative = better match
                    # Multiply by -1.0 so higher score always means better match
                    score = -1.0 * float(row[2])
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "score": score,
                        "metadata": {},
                        "source": "bm25"
                    })
        except Exception as e:
            logger.error(f"FTS5 search error: {e}")
            return []
        return results
