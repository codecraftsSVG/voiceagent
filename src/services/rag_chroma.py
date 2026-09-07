"""Lightweight RAG with ChromaDB - standalone, no Pipecat deps."""
import os
import time
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaRAG:
    """
    In-process ChromaDB with all-MiniLM-L6-v2 embeddings (80MB).
    Good for KBs up to ~50k documents on 8GB RAM.
    """

    def __init__(
        self,
        collection_name: str = "voice_kb",
        persist_dir: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
    ):
        started_at = time.perf_counter()
        print(
            f"[RAG] init start: collection={collection_name}, path={persist_dir}, "
            f"embedding={embedding_model}",
            flush=True,
        )
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=persist_dir)
        print("[RAG] PersistentClient ready", flush=True)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        print("[RAG] embedding function ready", flush=True)

        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_fn
            )
            print(f"[RAG] Loaded collection '{collection_name}'", flush=True)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name, embedding_function=self.embedding_fn
            )
            print(f"[RAG] Created new collection '{collection_name}'", flush=True)
        print(f"[RAG] init complete in {time.perf_counter() - started_at:.2f}s", flush=True)

    def count(self) -> int:
        """Return the collection size when the caller explicitly needs it."""
        return self.collection.count()

    def query(self, query_text: str) -> List[str]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=self.top_k,
        )
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d] if docs else []

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        ctx_block = "\n".join(f"• {c}" for c in contexts)
        return (
            f"Use this context to answer:\n{ctx_block}\n\n"
            f"User: {query}\nAssistant:"
        )

    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
        print(f"[RAG] Indexed {len(texts)} documents")
