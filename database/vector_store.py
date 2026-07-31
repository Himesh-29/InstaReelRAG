import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
from config import get_config
from config.logger import setup_logger

logger = setup_logger("VectorStore")

class LocalEmbeddingFunction:
    """Wrapper around sentence-transformers to use as a Chroma embedding function."""
    def __init__(self, model_name=None):
        from config import DEVICE
        config = get_config()
        self.model_name = model_name or config["retrieval"]["embedding_model"]
        logger.info(f"Loading embedding model '{self.model_name}' on device: {DEVICE.upper()}...")
        self.model = SentenceTransformer(self.model_name, device=DEVICE)

    def name(self) -> str:
        """Returns the name of the embedding function/model for ChromaDB validation."""
        return self.model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB method for embedding queries."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """ChromaDB method for embedding documents."""
        return self(input)

class VectorStore:
    def __init__(self, persist_directory: str = "./chromadb", collection_name: str = "instareelrag_docs"):
        """Initializes the ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = LocalEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def count(self) -> int:
        """Returns the number of documents currently stored in ChromaDB."""
        return self.collection.count()

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str] = None):
        """
        Adds new documents to ChromaDB.
        Skips any document text that is already stored in the database.
        """
        if not documents:
            return

        # 1. Get all document texts that are already in ChromaDB
        existing_docs = set(self.collection.get()["documents"])

        # 2. Filter out duplicates using a simple, readable loop
        new_docs = []
        new_metas = []
        new_ids = []

        for i in range(len(documents)):
            doc = documents[i]
            meta = metadatas[i]
            doc_id = ids[i] if ids else str(uuid.uuid4())

            # Only keep this document if it's not already in ChromaDB
            if doc not in existing_docs:
                new_docs.append(doc)
                new_metas.append(meta)
                new_ids.append(doc_id)

        # 3. If there are no new documents, we are done
        if not new_docs:
            logger.info("All documents are already present in ChromaDB.")
            return

        # 4. Save only the new documents
        self.collection.add(
            documents=new_docs,
            metadatas=new_metas,
            ids=new_ids
        )
        logger.info(f"Added {len(new_docs)} new documents to ChromaDB.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Searches ChromaDB for relevant documents matching the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # ChromaDB returns a list-of-lists (one list for each query searched).
        # Since we only passed 1 query, our results are at index 0.
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0] if 'ids' in results and results['ids'] else [metas[i].get('chunk_id', str(i)) for i in range(len(docs))]
            
            for i in range(len(docs)):
                # Convert distance into a similarity score (higher = better match)
                similarity_score = 1.0 - distances[i] if distances else 0.0
                
                formatted_results.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metas[i],
                    "score": similarity_score,
                    "source": "vector"
                })
                
        return formatted_results
