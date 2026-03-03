"""
FastAPI server for the Visa Assistant backend.

Endpoints:
  POST /chat   – conversational chat
  GET  /health – readiness probe
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import HOST, PORT
from rag import RAGPipeline
from agent import VisaAssistantAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

rag_pipeline: RAGPipeline | None = None
agent: VisaAssistantAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, agent
    try:
        logger.info("Starting up – building FAISS RAG pipeline …")
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize()
        agent = VisaAssistantAgent(rag_pipeline)
        logger.info("Agent ready – accepting requests")
        yield
    except Exception as e:
        logger.error("Startup error: %s", e)
        raise
    finally:
        logger.info("Shutting down gracefully...")
        # Clean up resources if needed
        rag_pipeline = None
        agent = None
        logger.info("Shutdown complete")


app = FastAPI(title="Visa Assistant API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    ready = rag_pipeline is not None and rag_pipeline.is_ready and agent is not None
    return {"status": "ok" if ready else "initializing", "ready": ready}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if agent is None:
        return ChatResponse(answer="System is still initializing. Please wait.")
    logger.info("session=%s  q=%s", req.session_id[:8], req.question[:80])
    try:
        answer = agent.chat(req.session_id, req.question)
    except Exception:
        logger.exception("Chat error")
        answer = "Sorry, something went wrong. Please try again."
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down gracefully...")
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        uvicorn.run(
            "main:app", 
            host=HOST, 
            port=PORT, 
            reload=False,
            log_level="info",
            access_log=False,  # Reduce noise
            use_colors=True,
            timeout_keep_alive=5,  # Faster shutdown
            timeout_graceful_shutdown=10  # Max 10s for graceful shutdown
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error("Server error: %s", e)
    finally:
        logger.info("Server shutdown complete")
