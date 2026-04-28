"""
Helpers for RAG-augmented evaluation scripts.

Provides oracle (full document) and chunk-based (BM25/embeddings) prompt augmentation.
"""

import re

from .utils import load_corpus, augment_prompt


def _normalize(s: str) -> str:
    """Normalize title for fuzzy matching: lowercase, remove common prefixes and punctuation."""
    s = s.strip().lower()
    # Remove common PCDT prefixes
    s = re.sub(r"^protocolo cl[ií]nico e diretrizes terap[êe]uticas\s*[-–—]?\s*(d[aoe]s?\s+)?", "", s)
    # Remove parenthetical suffixes like "(forma neovascular)" or "(TDAH)"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    return s.strip()


def build_oracle_lookup(dataset_name: str = "hugo/protocolos-clinicos-v1"):
    """Build lookup dicts from protocol corpus.

    Returns (by_titulo, by_arquivo, by_normalized) where keys are lowered/stripped strings.
    """
    docs = load_corpus(dataset_name)
    by_titulo = {}
    by_arquivo = {}
    by_normalized = {}
    for doc in docs:
        by_titulo[doc["titulo"].strip().lower()] = doc
        by_arquivo[doc["arquivo"].strip().lower()] = doc
        by_normalized[_normalize(doc["titulo"])] = doc
    return by_titulo, by_arquivo, by_normalized


def make_rag_prompt(
    question: str,
    row_titulo: str,
    row_arquivo: str | None,
    rag_mode: str,
    oracle_lookup: tuple | None = None,
    retriever=None,
    top_k: int = 5,
    max_context_chars: int | None = None,
) -> str:
    """Build RAG-augmented prompt for a question.

    Args:
        question: The original question text.
        row_titulo: Protocol title from the dataset row.
        row_arquivo: Protocol filename from the dataset row (HB only).
        rag_mode: "oracle", "bm25", or "embeddings".
        oracle_lookup: Tuple (by_titulo, by_arquivo) from build_oracle_lookup().
        retriever: RAGRetriever instance (for bm25/embeddings).
        top_k: Number of chunks to retrieve (bm25/embeddings only).
        max_context_chars: Truncate oracle text to this many chars (None = no limit).
    """
    if rag_mode == "oracle":
        by_titulo, by_arquivo, by_normalized = oracle_lookup
        doc = by_titulo.get(row_titulo.strip().lower())
        if doc is None and row_arquivo:
            doc = by_arquivo.get(row_arquivo.strip().lower())
        if doc is None:
            doc = by_normalized.get(_normalize(row_titulo))
        if doc is None:
            print(f"WARNING: protocolo não encontrado para titulo='{row_titulo}' arquivo='{row_arquivo}'")
            return question
        texto = doc["texto"]
        if max_context_chars and len(texto) > max_context_chars:
            texto = texto[:max_context_chars] + "\n[...truncado]"
        return (
            f"Protocolo: {doc['titulo']}\n\n"
            f"{texto}\n\n"
            f"Pergunta: {question}"
        )

    # bm25 or embeddings
    return augment_prompt(question, retriever, top_k)


def get_rag_context(
    row_titulo: str,
    row_arquivo: str | None,
    rag_mode: str,
    oracle_lookup: tuple | None = None,
    retriever=None,
    top_k: int = 5,
    max_context_chars: int | None = None,
) -> str:
    """Return just the RAG context (without the question), for system message injection."""
    if rag_mode == "oracle":
        by_titulo, by_arquivo, by_normalized = oracle_lookup
        doc = by_titulo.get(row_titulo.strip().lower())
        if doc is None and row_arquivo:
            doc = by_arquivo.get(row_arquivo.strip().lower())
        if doc is None:
            doc = by_normalized.get(_normalize(row_titulo))
        if doc is None:
            return ""
        texto = doc["texto"]
        if max_context_chars and len(texto) > max_context_chars:
            texto = texto[:max_context_chars] + "\n[...truncado]"
        return f"Protocolo: {doc['titulo']}\n\n{texto}"

    # bm25 or embeddings — use query=titulo as proxy
    return retriever.format_context(row_titulo, top_k)
