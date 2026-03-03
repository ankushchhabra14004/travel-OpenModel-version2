# ── Visa Assistant Backend ─────────────────────────────────────────
# Single-stage build.  Pre-built FAISS files are committed to the repo,
# so startup is instant (no index rebuild needed).
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps: gcc is needed for some Python packages; cleanup keeps the layer small
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Working dir matches the path expectations in config.py:
#   BASE_DIR  = /app/Backend
#   PROJECT_DIR = /app
WORKDIR /app/Backend

# Install Python dependencies first (cached layer)
COPY Backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend source (includes pre-built faiss_store/)
COPY Backend/ .

# Pre-download the sentence-transformers embedding model so the
# first request is fast even if the HuggingFace cache is cold.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# PORT is injected by Render at runtime (defaults to 8002 locally)
EXPOSE 8002

CMD ["python", "main.py"]
