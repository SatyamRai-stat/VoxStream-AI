"""
VoxStream AI — FastAPI Backend
Wraps run_pipeline() and exposes REST + SSE endpoints consumed by the frontend.

Endpoints:
  POST /api/analyze          — kick off pipeline, returns session_id
  GET  /api/status/{sid}     — SSE stream of pipeline phase updates
  GET  /api/result/{sid}     — fetch completed result JSON
  POST /api/chat/{sid}       — RAG chat turn
  POST /api/upload           — receive audio file, returns temp path
  DELETE /api/session/{sid}  — cleanup
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # project root containing main.py
sys.path.insert(0, str(ROOT))

# Import pipeline pieces (lazy so import errors surface clearly)
def _import_pipeline():
    from utils.audio_processor import process_input
    from core.transcribe import transcribe_all
    from core.summarize import summarize, generate_title
    from core.extractor import (
        extract_action_items,
        extract_key_decisions,
        extract_questions,
    )
    from core.rag_engine import build_rag_chain, ask_question
    return (
        process_input, transcribe_all, summarize, generate_title,
        extract_action_items, extract_key_decisions, extract_questions,
        build_rag_chain, ask_question,
    )


# ── In-memory session store ────────────────────────────────────────────────────
# Each session:
#   status   : "queued" | "running" | "done" | "error"
#   phases   : list of {phase, message, done} dicts
#   result   : dict (populated when done)
#   rag_chain: live chain object
#   events   : asyncio.Queue for SSE
SESSIONS: dict[str, dict] = {}

PHASE_LABELS = [
    ("ingest",     "Ingesting media source"),
    ("transcribe", "Transcribing audio with Whisper ASR"),
    ("summarize",  "Generating title & summary"),
    ("extract",    "Extracting actions, decisions, questions"),
    ("vectorize",  "Building RAG vector store"),
]


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    yield


app = FastAPI(title="VoxStream AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    source: str          # YouTube URL or previously-uploaded temp path
    language: str = "english"


class ChatRequest(BaseModel):
    question: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _new_session() -> dict:
    return {
        "status": "queued",
        "phases": [],
        "result": None,
        "rag_chain": None,
        "events": asyncio.Queue(),
        "error": None,
    }


async def _emit(session: dict, event: str, data: dict):
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    await session["events"].put(payload)


async def _run_pipeline_async(session_id: str, source: str, language: str):
    """Run the blocking pipeline in a thread pool and stream phase events."""
    session = SESSIONS[session_id]
    loop = asyncio.get_running_loop()

    async def phase(label: str, message: str, fn, *args):
        await _emit(session, "phase", {"label": label, "message": message, "done": False})
        result = await loop.run_in_executor(None, fn, *args)
        await _emit(session, "phase", {"label": label, "message": message, "done": True})
        return result

    try:
        session["status"] = "running"
        (
            process_input, transcribe_all, summarize, generate_title,
            extract_action_items, extract_key_decisions, extract_questions,
            build_rag_chain, ask_question,
        ) = _import_pipeline()

        # Store ask_question so chat endpoint can use it
        session["ask_question"] = ask_question

        # ── Phase 1: ingest ──────────────────────────────────────────────
        chunks = await phase("ingest", "Ingesting media source", process_input, source)

        # ── Phase 2: transcribe ──────────────────────────────────────────
        transcript = await phase(
            "transcribe", "Transcribing audio with Whisper ASR",
            transcribe_all, chunks
        )

        # ── Phase 3: summarize ───────────────────────────────────────────
        await _emit(session, "phase", {"label": "summarize", "message": "Generating title & summary", "done": False})
        title   = await loop.run_in_executor(None, generate_title, transcript)
        summary = await loop.run_in_executor(None, summarize, transcript)
        await _emit(session, "phase", {"label": "summarize", "message": "Generating title & summary", "done": True})

        # ── Phase 4: extract ─────────────────────────────────────────────
        await _emit(session, "phase", {"label": "extract", "message": "Extracting actions, decisions & questions", "done": False})
        action_items = await loop.run_in_executor(None, extract_action_items, transcript)
        decisions    = await loop.run_in_executor(None, extract_key_decisions, transcript)
        questions    = await loop.run_in_executor(None, extract_questions, transcript)
        await _emit(session, "phase", {"label": "extract", "message": "Extracting actions, decisions & questions", "done": True})

        # ── Phase 5: vectorize ───────────────────────────────────────────
        rag_chain = await phase(
            "vectorize", "Building RAG vector store",
            build_rag_chain, transcript
        )

        session["rag_chain"]   = rag_chain
        session["result"] = {
            "title":        title,
            "transcript":   transcript,
            "summary":      summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
        }
        session["status"] = "done"
        await _emit(session, "done", {"session_id": session_id, "title": title})

    except Exception as exc:
        session["status"] = "error"
        session["error"]  = str(exc)
        await _emit(session, "error", {"message": str(exc)})
    finally:
        await session["events"].put(None)   # sentinel → close SSE


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Start pipeline. Returns session_id immediately."""
    sid = str(uuid.uuid4())
    SESSIONS[sid] = _new_session()
    asyncio.create_task(_run_pipeline_async(sid, req.source, req.language))
    return {"session_id": sid}


@app.get("/api/status/{session_id}")
async def status_stream(session_id: str):
    """SSE stream: phase events until done/error."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        q: asyncio.Queue = session["events"]
        while True:
            msg = await q.get()
            if msg is None:
                break
            yield msg

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session["status"] == "error":
        raise HTTPException(500, session["error"])
    if session["status"] != "done":
        raise HTTPException(202, "Pipeline still running")
    result = dict(session["result"])
    # Truncate transcript for JSON response (can be huge)
    result["transcript_preview"] = result["transcript"][:500]
    del result["transcript"]
    return JSONResponse(result)


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, req: ChatRequest):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if session["status"] != "done":
        raise HTTPException(400, "Pipeline not complete yet")

    rag_chain   = session["rag_chain"]
    ask_question = session.get("ask_question")

    if rag_chain is None or ask_question is None:
        raise HTTPException(500, "RAG chain not initialised")

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, ask_question, rag_chain, req.question)
    return {"answer": answer}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Save uploaded audio to a temp file and return its path."""
    suffix = Path(file.filename).suffix or ".audio"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    try:
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()
    return {"path": tmp.name, "filename": file.filename}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


# ── Serve frontend static files ───────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")