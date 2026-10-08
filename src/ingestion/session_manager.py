from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "data" / "runtime_sessions"


def sanitize_filename(filename: str) -> str:
    name = Path(filename or "uploaded.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def create_session(
    original_filename: str,
    pdf_bytes: bytes,
    base_dir: Optional[str | Path] = None,
) -> Dict[str, str]:
    base_path = Path(base_dir or DEFAULT_SESSION_ROOT)
    base_path.mkdir(parents=True, exist_ok=True)

    session_id = uuid.uuid4().hex
    session_root = base_path / session_id
    document_dir = session_root / "document"
    document_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(original_filename)
    pdf_path = document_dir / filename
    pdf_path.write_bytes(pdf_bytes)

    metadata = {
        "session_id": session_id,
        "document_name": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_epoch": time.time(),
    }
    (session_root / "session.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "session_id": session_id,
        "session_root": str(session_root),
        "document_name": filename,
        "pdf_path": str(pdf_path),
    }


def get_session_root(
    session_id: str,
    base_dir: Optional[str | Path] = None,
) -> Path:
    return Path(base_dir or DEFAULT_SESSION_ROOT) / session_id


def cleanup_session(
    session_id: str,
    base_dir: Optional[str | Path] = None,
) -> bool:
    import gc
    try:
        from src.agents.workflow import reset_session_retriever
        reset_session_retriever(session_id)
    except Exception:
        pass

    session_root = get_session_root(session_id, base_dir)
    if not session_root.exists() or not session_root.is_dir():
        return False

    gc.collect()
    for attempt in range(5):
        try:
            shutil.rmtree(session_root)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(0.2)

    shutil.rmtree(session_root, ignore_errors=True)
    return not session_root.exists()


def cleanup_stale_sessions(
    max_age_hours: float = 12.0,
    base_dir: Optional[str | Path] = None,
) -> int:
    """Delete old runtime sessions. Permanent corpus data is untouched."""
    root = Path(base_dir or DEFAULT_SESSION_ROOT)
    if not root.exists():
        return 0

    cutoff = time.time() - (max_age_hours * 3600.0)
    removed = 0

    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            mtime = session_dir.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
            removed += 1

    return removed
