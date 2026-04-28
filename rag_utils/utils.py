"""
RAG utilities for clinical protocol retrieval.

Supports two retrieval methods:
  - BM25 (keyword-based, rank-bm25)
  - OpenAI embeddings (semantic, text-embedding-3-small + numpy cosine)

And two modes:
  - Full document retrieval (BM25 only — docs exceed embedding token limit)
  - Chunked retrieval (both methods)
"""

import hashlib
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_character_split(
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text recursively by trying separators in order.

    Tries to split on the first separator that produces pieces <= chunk_size.
    If a piece still exceeds chunk_size, recurses with the next separator.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = separators or DEFAULT_SEPARATORS

    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators):
            # Last resort: hard cut
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunks.append(text[i : i + chunk_size])
                if i + chunk_size >= len(text):
                    break
            return chunks

        sep = separators[sep_idx]
        if sep:
            parts = text.split(sep)
        else:
            # Empty separator = character-level split (hard cut)
            return _split(text, sep_idx + 1) if sep_idx + 1 < len(separators) else [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size - chunk_overlap)
            ]

        # Merge small parts into chunks respecting chunk_size
        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the part itself exceeds chunk_size, recurse deeper
                if len(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)

        # Add overlap between consecutive chunks
        if chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                overlapped.append(overlap_text + chunks[i])
            chunks = overlapped

        return chunks

    return _split(text, 0)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(
    dataset_name: str = "hugo/protocolos-clinicos-v1",
) -> list[dict]:
    """Load all clinical protocols from HuggingFace.

    Returns list of {titulo, texto, arquivo} dicts (train + test = 178 docs).
    """
    ds = load_dataset(dataset_name)
    docs = []
    for split in ds:
        for row in ds[split]:
            docs.append({
                "titulo": row["titulo"],
                "texto": row["texto"],
                "arquivo": row["arquivo"],
            })
    return docs


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    titulo: str
    arquivo: str
    chunk_index: int = -1  # -1 = full document


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Simple RAG retriever over clinical protocols.

    Args:
        method: "bm25" or "embeddings"
        chunk: Whether to chunk documents before indexing
        chunk_size: Characters per chunk (only used if chunk=True)
        chunk_overlap: Overlap characters between chunks
        separators: Custom separators for chunking
        dataset_name: HuggingFace dataset with protocols
        embedding_model: OpenAI embedding model name
        cache_dir: Directory for embedding cache (default: rag_utils/.cache/)
        api_key: OpenAI API key (default: OPENAI_API_KEY env var)
    """

    def __init__(
        self,
        method: str = "bm25",
        chunk: bool = True,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        dataset_name: str = "hugo/protocolos-clinicos-v1",
        embedding_model: str = "text-embedding-3-small",
        cache_dir: str | None = None,
        api_key: str | None = None,
    ):
        assert method in ("bm25", "embeddings"), f"method must be 'bm25' or 'embeddings', got '{method}'"

        self.method = method
        self.chunk = chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.dataset_name = dataset_name
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Force chunking for embeddings (docs exceed token limit)
        if method == "embeddings" and not chunk:
            warnings.warn(
                "Embedding mode requires chunking (docs exceed 8191 token limit). "
                "Enabling chunk=True automatically.",
                stacklevel=2,
            )
            self.chunk = True

        # Load corpus and build documents
        raw_docs = load_corpus(dataset_name)
        self.documents = self._build_documents(raw_docs)
        print(f"RAGRetriever: {len(self.documents)} {'chunks' if self.chunk else 'documents'} indexed ({method})")

        # Build index
        if method == "bm25":
            self._build_bm25()
        else:
            self._build_embeddings()

    def _build_documents(self, raw_docs: list[dict]) -> list[Document]:
        documents = []
        if self.chunk:
            for doc in raw_docs:
                chunks = recursive_character_split(
                    doc["texto"],
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators,
                )
                for i, chunk_text in enumerate(chunks):
                    documents.append(Document(
                        text=chunk_text,
                        titulo=doc["titulo"],
                        arquivo=doc["arquivo"],
                        chunk_index=i,
                    ))
        else:
            for doc in raw_docs:
                documents.append(Document(
                    text=doc["texto"],
                    titulo=doc["titulo"],
                    arquivo=doc["arquivo"],
                    chunk_index=-1,
                ))
        return documents

    # --- BM25 ---

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi

        tokenized = [doc.text.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def _retrieve_bm25(self, query: str, top_k: int) -> list[int]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()

    # --- Embeddings ---

    def _build_embeddings(self):
        self.embeddings = self._load_or_compute_embeddings()

    def _cache_path(self) -> Path:
        key = f"{self.dataset_name}|{self.chunk_size}|{self.chunk_overlap}|{self.embedding_model}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"embeddings_{h}.npz"

    def _load_or_compute_embeddings(self) -> np.ndarray:
        cache = self._cache_path()
        if cache.exists():
            data = np.load(cache)
            if int(data["doc_count"]) == len(self.documents):
                print(f"RAGRetriever: loaded cached embeddings from {cache}")
                return data["embeddings"]
            print(f"RAGRetriever: cache mismatch (expected {len(self.documents)}, got {int(data['doc_count'])}), recomputing")

        texts = [doc.text for doc in self.documents]
        embeddings = self._embed_texts(texts)

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, embeddings=embeddings, doc_count=len(self.documents))
        print(f"RAGRetriever: cached embeddings to {cache}")
        return embeddings

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        all_embeddings = []
        batch_size = 256

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"RAGRetriever: embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} texts)")
            resp = client.embeddings.create(model=self.embedding_model, input=batch)
            all_embeddings.extend([e.embedding for e in resp.data])

        return np.array(all_embeddings, dtype=np.float32)

    def _retrieve_embeddings(self, query: str, top_k: int) -> list[int]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model=self.embedding_model, input=[query])
        query_emb = np.array(resp.data[0].embedding, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        sims = (self.embeddings @ query_emb) / (norms * query_norm + 1e-10)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return top_indices.tolist()

    # --- Public API ---

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top_k most relevant text fragments for the query."""
        docs = self.retrieve_with_metadata(query, top_k)
        return [doc.text for doc in docs]

    def retrieve_with_metadata(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top_k Documents with metadata."""
        if self.method == "bm25":
            indices = self._retrieve_bm25(query, top_k)
        else:
            indices = self._retrieve_embeddings(query, top_k)
        return [self.documents[i] for i in indices]

    def format_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve and format as a context string ready for prompt injection."""
        docs = self.retrieve_with_metadata(query, top_k)
        parts = []
        for i, doc in enumerate(docs, 1):
            header = f"[Trecho {i} — {doc.titulo}]"
            parts.append(f"{header}\n{doc.text}")
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def augment_prompt(query: str, retriever: RAGRetriever, top_k: int = 5) -> str:
    """Augment a query with RAG context. Returns the new user message."""
    context = retriever.format_context(query, top_k)
    return (
        f"Contexto relevante dos protocolos clínicos:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Pergunta: {query}"
    )
