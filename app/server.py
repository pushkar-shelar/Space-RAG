from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.workflow import (
    reset_session_retriever,
    run_space_rag,
    stream_space_rag_answer,
)
from src.indexing.corpus_builder import build_corpus_index
from src.indexing.index_document import index_private_session
from src.ingestion.pipeline import create_and_process_session
from src.ingestion.session_manager import cleanup_session, cleanup_stale_sessions

CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
CORPUS_INDEX = PROJECT_ROOT / "data" / "corpus_index"

# Periodic cleanup of stale sessions (>12h)
cleanup_stale_sessions(max_age_hours=12)

app = FastAPI(title="Space RAG Mission Control API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    session_root: Optional[str] = None
    document_name: Optional[str] = None
    model_mode: str = "offline"
    chat_history: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[str] = None
    top_k: int = 5
    retrieval_k: int = 10


class DetachRequest(BaseModel):
    session_id: str


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "Space RAG Mission Control",
        "version": "1.0.0",
        "corpus_ready": CORPUS_INDEX.exists(),
    }


@app.get("/api/status")
def get_status():
    manifest_file = CORPUS_INDEX / "corpus_manifest.json"
    manifest_data = {}
    if manifest_file.exists():
        try:
            manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    corpus_docs = []
    if CORPUS_DIR.exists():
        corpus_docs = [p.name for p in sorted(CORPUS_DIR.glob("*.pdf"))]

    return {
        "status": "nominal",
        "models": {
            "offline": "Qwen/Qwen2.5-1.5B-Instruct",
            "online": "grok-2-latest (xAI)",
            "reranker": "BAAI/bge-reranker-base",
            "visual": "ColPali + SmolVLM-500M",
        },
        "corpus": {
            "ready": CORPUS_INDEX.exists(),
            "document_count": manifest_data.get("document_count", len(corpus_docs)),
            "bm25_chunks": manifest_data.get("bm25_chunks", 0),
            "dense_chunks": manifest_data.get("dense_chunks", 0),
            "visual_pages": manifest_data.get("visual_pages", 0),
            "documents": manifest_data.get("documents", corpus_docs),
        },
    }


@app.post("/api/session/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
        result = create_and_process_session(file.filename, content)
        session = result["session"]
        manifest = result["manifest"]

        # Build initial text retrieval indexes (BM25 + Chroma)
        index_private_session(session["session_root"])

        return {
            "success": True,
            "session_id": session["session_id"],
            "session_root": session["session_root"],
            "document_name": session["document_name"],
            "total_pages": manifest.get("total_pages", 1),
            "metadata": manifest,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/session/detach")
def detach_session(req: DetachRequest):
    try:
        reset_session_retriever(req.session_id)
        gc.collect()
        success = cleanup_session(req.session_id)
        return {"success": success, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detach session: {str(e)}")


@app.post("/api/corpus/sync")
def sync_corpus():
    try:
        stats = build_corpus_index(CORPUS_DIR, CORPUS_INDEX)
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync corpus: {str(e)}")


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint."""
    def event_stream():
        try:
            stream_result = stream_space_rag_answer(
                query=req.query.strip(),
                session_id=req.session_id,
                session_root=req.session_root,
                document_name=req.document_name,
                model_mode=req.model_mode,
                corpus_root=CORPUS_INDEX,
                top_k=req.top_k,
                retrieval_k=req.retrieval_k,
                chat_history=req.chat_history or [],
                api_key=req.api_key,
            )

            # Send metadata header first (citations, source_mix, retrieval_results)
            # Filter page images so URLs are relative for web rendering
            clean_results = []
            for item in stream_result.get("retrieval_results", []):
                img_path = item.get("page_image")
                web_img_url = None
                if img_path and Path(img_path).exists():
                    web_img_url = f"/api/image?path={Path(img_path).resolve()}"
                
                clean_results.append({
                    "source_type": item.get("source_type"),
                    "document": item.get("document"),
                    "page_number": item.get("page_number"),
                    "text": item.get("text", "")[:400],
                    "fusion_score": round(item.get("fusion_score", 0.0), 3),
                    "page_image_url": web_img_url,
                    "verification": item.get("verification"),
                })

            init_payload = {
                "type": "meta",
                "citations": stream_result.get("citations", []),
                "source_mix": stream_result.get("source_mix", {}),
                "retrieval_results": clean_results,
            }
            yield f"data: {json.dumps(init_payload)}\n\n"

            # Stream tokens
            for token in stream_result["stream"]:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/image")
def get_image(path: str):
    p = Path(path).resolve()
    # Security check: ensure path is within project root
    if not str(p).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(p, media_type="image/png")


# Mount static production build of React frontend if present
from fastapi.staticfiles import StaticFiles

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)

