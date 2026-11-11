"""Hybrid retrieval: dense embeddings plus BM25, merged with reciprocal rank fusion."""
import re

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from .ingest import Section

SUFFIXES = ("ings", "ing", "als", "al", "ed", "es", "s", "e")


def stem(word: str) -> str:
    """Tiny suffix stripper so "approve", "approved" and "Approvals" share one BM25 term."""
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def section_view(s: Section) -> str:
    return f"{s.doc_title}: {s.number}. {s.title}\n{s.text}"


class Retriever:
    def __init__(self, sections: list[Section], model_name: str):
        self.sections = sections
        self.model = TextEmbedding(model_name)   # ONNX runtime, no PyTorch needed
        texts = [section_view(s) for s in sections]
        self.emb = self._embed(texts)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)   # unit length, so dot = cosine

    def search(self, query: str, k: int) -> tuple[list[tuple[Section, float]], float]:
        q = self._embed([query])[0]
        dense = self.emb @ q
        sparse = self.bm25.get_scores(tokenize(query))
        rrf = np.zeros(len(self.sections))
        for scores in (dense, sparse):
            for rank, idx in enumerate(np.argsort(-scores)):
                rrf[idx] += 1.0 / (60 + rank)
        top = np.argsort(-rrf)[:k]
        return [(self.sections[i], float(dense[i])) for i in top], float(dense.max())
