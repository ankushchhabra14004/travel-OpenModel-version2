"""
Configuration for the Visa Assistant Backend.
"""
import os

# -- API Configuration ---------------------------------------------------
GEMINI_API_KEY = "AIzaSyBbC1w6lsyiic1WqJ0Hn4SrIQ0UY3d6bgs"
LLM_MODEL = "gemma-3-4b-it"

# -- Local Embedding Model (runs on CPU, zero API limits) ----------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -- Paths ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
SCRAPPED_DATA_DIR = os.path.join(PROJECT_DIR, "scrapped_results")
FAISS_PERSIST_DIR = os.path.join(BASE_DIR, "faiss_store")

# -- RAG Configuration ---------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 10          # retrieve 10 chunks per query

# -- Supported Countries (folder names inside scrapped_results/) ----------
SUPPORTED_COUNTRIES = ["USA", "Singapore", "Qatar"]

# -- Server ---------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8002
