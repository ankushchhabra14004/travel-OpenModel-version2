"""
FAISS-based RAG pipeline for visa data.

Uses a local sentence-transformers embedding model (all-MiniLM-L6-v2)
so there are *zero* API rate-limit concerns. Each supported country
gets its own FAISS index persisted to disk.
"""

import json
import os
import logging
import pickle
import numpy as np
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import (
    EMBEDDING_MODEL_NAME,
    SCRAPPED_DATA_DIR,
    FAISS_PERSIST_DIR,
    SUPPORTED_COUNTRIES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


# ── Data Loading ───────────────────────────────────────────────────

def load_visa_data(country_name: str) -> List[Document]:
    """Load the *_visa_data.json for a country into LangChain Documents."""
    file_path = os.path.join(
        SCRAPPED_DATA_DIR, country_name, f"{country_name}_visa_data.json"
    )
    if not os.path.exists(file_path):
        logger.error("Data file not found: %s", file_path)
        return []

    logger.info("Loading visa data for %s from %s", country_name, file_path)

    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    documents: List[Document] = []
    for page in data.get("visa_pages", []):
        content = page.get("content", "").strip()
        if not content or len(content) < 50:
            continue

        title = page.get("title", "Unknown")
        url = page.get("url", "")

        # Flatten table rows
        tables_text = ""
        for table in page.get("tables", []):
            for row in table.get("rows", []):
                tables_text += " | ".join(str(cell) for cell in row) + "\n"

        full_content = f"Title: {title}\nSource: {url}\n\n{content}"
        if tables_text.strip():
            full_content += f"\n\nTable Data:\n{tables_text}"

        documents.append(
            Document(
                page_content=full_content,
                metadata={"country": country_name, "title": title, "url": url},
            )
        )

    logger.info("Loaded %d pages for %s", len(documents), country_name)
    return documents


# ── RAG Pipeline ───────────────────────────────────────────────────

class RAGPipeline:
    """Per-country FAISS vector stores with local embeddings."""

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dim = self.embed_model.get_sentence_embedding_dimension()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )
        # country -> (faiss.Index, list[Document])
        self.stores: dict[str, tuple] = {}
        self._initialized = False

    # ── public ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
        for country in SUPPORTED_COUNTRIES:
            try:
                self._init_country(country)
            except Exception:
                logger.exception("Failed to init FAISS for %s", country)
        self._initialized = True
        logger.info("RAG ready – %d country stores loaded", len(self.stores))

    def retrieve(self, country: str, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        if country not in self.stores:
            logger.warning("No FAISS index for country: %s", country)
            return []
        index, chunks = self.stores[country]
        q_vec = self.embed_model.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")
        k = min(k, index.ntotal)
        distances, indices = index.search(q_vec, k)
        results = []
        for i in indices[0]:
            if 0 <= i < len(chunks):
                results.append(chunks[i])
        return results

    @property
    def is_ready(self) -> bool:
        return self._initialized and len(self.stores) > 0

    @property
    def available_countries(self) -> list[str]:
        return list(self.stores.keys())

    # ── private ────────────────────────────────────────────────────

    def _init_country(self, country: str) -> None:
        idx_path = os.path.join(FAISS_PERSIST_DIR, f"{country.lower()}.index")
        docs_path = os.path.join(FAISS_PERSIST_DIR, f"{country.lower()}_docs.pkl")

        if os.path.exists(idx_path) and os.path.exists(docs_path):
            logger.info("Loading persisted FAISS index for %s", country)
            index = faiss.read_index(idx_path)
            with open(docs_path, "rb") as f:
                chunks = pickle.load(f)
            self.stores[country] = (index, chunks)
            logger.info("Loaded %d chunks for %s", index.ntotal, country)
            return

        self._build_country(country, idx_path, docs_path)

    def _build_country(self, country: str, idx_path: str, docs_path: str) -> None:
        logger.info("Building FAISS index for %s …", country)
        documents = load_visa_data(country)
        if not documents:
            logger.warning("No documents for %s", country)
            return

        chunks = self.text_splitter.split_documents(documents)
        logger.info("Chunked into %d pieces for %s", len(chunks), country)

        texts = [c.page_content for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=True,
                                              normalize_embeddings=True,
                                              batch_size=256)
        embeddings = np.array(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(self.dim)      # inner-product (cosine on normalised vecs)
        index.add(embeddings)

        faiss.write_index(index, idx_path)
        with open(docs_path, "wb") as f:
            pickle.dump(chunks, f)

        self.stores[country] = (index, chunks)
        logger.info("FAISS index built for %s – %d chunks", country, len(chunks))
