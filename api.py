from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
import logging
import json
import mimetypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import traceback
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .contracts import PipelineStageError
from .in_process import run_operation
from .job_store import JobStore
from .media import FileArtifactBackend
from .service_adapters import (
    LANGUAGES, prepare_source_transcript, prepare_translation_review, validate_review_srt,
)


DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", str(Path(tempfile.gettempdir()) / "dir-language-pipeline"))).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("PIPELINE_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
JOB_ROOT = DATA_DIR / "jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
store = JobStore(DATA_DIR / "jobs.sqlite3")
stop_worker = threading.Event()
worker_thread: threading.Thread | None = None


class _TerminalAndFileStream:
    """Mirror every worker print to the server terminal and a persistent log."""

    def __init__(self, terminal, logfile, job_id: str):
        self.terminal = terminal
        self.logfile = logfile
        self.prefix = f"[job {job_id}] "
        self.line_start = True
        self.lock = threading.RLock()

    def write(self, value: str) -> int:
        if not value:
            return 0
        with self.lock:
            offset = 0
            while offset < len(value):
                end = value.find("\n", offset)
                end = len(value) if end < 0 else end + 1
                piece = value[offset:end]
                if self.line_start:
                    self.logfile.write(self.prefix)
                    self._terminal_write(self.prefix)
                self.logfile.write(piece)
                self._terminal_write(piece)
                self.line_start = piece.endswith("\n")
                offset = end
            self.flush()
        return len(value)

    def _terminal_write(self, text: str) -> None:
        # Consoles with a legacy code page (e.g. cp1252 on Windows) cannot encode
        # Indic scripts; replace those characters on screen. process.log stays UTF-8.
        try:
            self.terminal.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", None) or "ascii"
            self.terminal.write(text.encode(encoding, errors="replace").decode(encoding))

    def flush(self) -> None:
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def _process_job(job: dict) -> None:
    job_id = job["job_id"]
    job_root = JOB_ROOT / job_id
    artifacts_root = job_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    backend = FileArtifactBackend(artifacts_root)
    log_path = job_root / "process.log"
    review_dir = job_root / "review"
    paused_for_review = False
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            terminal = sys.__stdout__ or sys.stdout
            stream = _TerminalAndFileStream(terminal, log, job_id)
            root_logger = logging.getLogger()
            previous_level = root_logger.level
            handler = logging.StreamHandler(stream)
            root_logger.addHandler(handler)
            if previous_level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    print(f"Starting pipeline for {job['input_name']} | source={job['source_language']} | target={job['target_language']}")
                    source = Path(job["input_path"])
                    step = job.get("review_step", "processing")
                    if step == "source_transcript":
                        store.stage(job_id, "transcribing_source")
                        generated = prepare_source_transcript(source, review_dir, job["source_language"])
                        artifact = artifacts_root / "source_transcript.srt"
                        shutil.copy2(generated, artifact)
                        store.pause_for_review(job_id, "source_transcript_review", "translation", {"source_transcript": str(artifact)})
                        print("--- waiting for source transcript review ---", flush=True)
                        paused_for_review = True
                    elif step == "translation":
                        source_srt = review_dir / "approved_source_transcript.srt"
                        if not source_srt.is_file():
                            raise RuntimeError("Approved source transcript is missing; resubmit the source transcript review.")
                        store.stage(job_id, "translating_for_review")
                        generated = prepare_translation_review(
                            source_srt, job["target_language"], job["source_language"], review_dir,
                        )
                        artifact = artifacts_root / "target_translation.srt"
                        shutil.copy2(generated, artifact)
                        store.pause_for_review(job_id, "translation_review", "processing", {"target_translation": str(artifact)})
                        print("--- waiting for target translation review ---", flush=True)
                        paused_for_review = True
                    else:
                        content_type = mimetypes.guess_type(job["input_name"])[0] or "application/octet-stream"
                        incoming = backend.upload(source, filename=job["input_name"], content_type=content_type)
                        source_srt = review_dir / "approved_source_transcript.srt"
                        target_srt = review_dir / "approved_target_translation.srt"
                        result = run_operation(
                            incoming,
                            operation=job["operation"],
                            target_language=job["target_language"],
                            source_language=job["source_language"],
                            backend=backend,
                            job_id=job_id,
                            on_stage=lambda value: (store.stage(job_id, value), print(f"--- stage: {value} ---", flush=True)),
                            source_transcript=source_srt if source_srt.is_file() else None,
                            reviewed_translation=target_srt if target_srt.is_file() else None,
                            add_auto_captions=bool(job.get("add_auto_captions")),
                        )
            finally:
                root_logger.removeHandler(handler)
                root_logger.setLevel(previous_level)
        if not paused_for_review:
            output_artifacts = {
                name: artifact.metadata["path"] for name, artifact in result.items()
            }
            store.finish(job_id, output_artifacts)
    except Exception as exc:
        stage = exc.stage if isinstance(exc, PipelineStageError) else "processing"
        trace = traceback.format_exc()
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[job {job_id}] FAILED at {stage}: {exc}\n{trace}\n")
        error_console = sys.__stderr__ or sys.stderr
        print(f"[job {job_id}] FAILED at {stage}: {exc}\n{trace}", file=error_console, flush=True)
        store.fail(job_id, stage, str(exc))


def _worker() -> None:
    terminal = sys.__stderr__ or sys.stderr
    print("[pipeline-worker] started", file=terminal, flush=True)
    while not stop_worker.is_set():
        job = None
        try:
            job = store.claim()
            if job is None:
                stop_worker.wait(0.5)
                continue
            print(f"[pipeline-worker] claimed job {job['job_id']}", file=terminal, flush=True)
            _process_job(job)
        except Exception:
            details = traceback.format_exc()
            print(f"[pipeline-worker] loop error\n{details}", file=terminal, flush=True)
            if job is not None:
                try:
                    store.fail(job["job_id"], "worker", details.strip())
                except Exception:
                    print(f"[pipeline-worker] could not record failure for {job['job_id']}\n{traceback.format_exc()}", file=terminal, flush=True)
            stop_worker.wait(2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker_thread
    stop_worker.clear()
    worker_thread = threading.Thread(target=_worker, name="directory-language-worker", daemon=True)
    worker_thread.start()
    try:
        yield
    finally:
        stop_worker.set()
        if worker_thread:
            await asyncio.to_thread(worker_thread.join, 30)


app = FastAPI(title="Directory Language Studio", version="1.0.0", lifespan=lifespan)


class ReviewSubmission(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/languages")
def languages():
    return [{"code": code, "name": name} for code, name in LANGUAGES.items()]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_running": bool(worker_thread and worker_thread.is_alive()),
    }


@app.post("/api/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None),
    operation: str = Form("combine"),
    add_auto_captions: bool = Form(False),
):
    if operation not in {"masking", "dubbing", "combine"}:
        raise HTTPException(422, "Operation must be masking, dubbing, or combine.")
    target = target_language.lower().strip()
    if target not in LANGUAGES:
        raise HTTPException(422, f"Unsupported target language code: {target}")
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(415, "Upload an MP4, MOV, MKV, AVI, or WebM video.")

    job_id = uuid4().hex
    job_root = JOB_ROOT / job_id
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    safe_name = Path(file.filename or f"video{suffix}").name
    destination = input_dir / safe_name
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds configured upload size limit.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    source = (source_language or os.getenv("PIPELINE_SOURCE_LANGUAGE", "auto")).strip() or "auto"
    store.create(
        job_id, target, source, destination, safe_name, operation,
        add_auto_captions=add_auto_captions and operation in {"dubbing", "combine"},
    )
    return {"job_id": job_id, "status": "queued", "operation": operation, "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job_id,
        "status": record["status"],
        "stage": record["stage"],
        "target_language": record["target_language"],
        "source_language": record["source_language"],
        "operation": record["operation"],
        "add_auto_captions": bool(record["add_auto_captions"]),
        "error": record["error"],
        "artifacts": {
            name: f"/api/jobs/{job_id}/artifacts/{name}"
            for name in record["artifacts"]
        },
        "manifest_url": f"/api/jobs/{job_id}/manifest",
        "review_url": f"/api/jobs/{job_id}/review",
        "input_url": f"/api/jobs/{job_id}/input",
        "log_url": f"/api/jobs/{job_id}/logs",
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


@app.get("/api/jobs/{job_id}/input")
def get_job_input(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    path = Path(record["input_path"]).resolve()
    if not path.is_file() or JOB_ROOT.resolve() not in path.parents:
        raise HTTPException(404, "Uploaded video is unavailable")
    media_type = mimetypes.guess_type(record["input_name"])[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=record["input_name"])


@app.get("/api/jobs/{job_id}/review")
def get_job_review(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if record["status"] != "review_required":
        raise HTTPException(409, "This job is not waiting for review")
    stage = record["stage"]
    name = "source_transcript" if stage == "source_transcript_review" else "target_translation" if stage == "translation_review" else None
    if name is None or name not in record["artifacts"]:
        raise HTTPException(409, "No transcript is awaiting review")
    path = Path(record["artifacts"][name])
    if not path.is_file():
        raise HTTPException(404, "Review transcript is unavailable")
    report_name = "source_transcript_ai_review.json" if stage == "source_transcript_review" else "translation_ai_review.json"
    report_path = JOB_ROOT / job_id / "review" / report_name
    report = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            report = {"summary": "AI review report could not be read.", "issues": []}
    return {"stage": stage, "text": path.read_text(encoding="utf-8"), "report": report,
            "artifact": f"/api/jobs/{job_id}/artifacts/{name}"}


@app.post("/api/jobs/{job_id}/review")
def submit_job_review(job_id: str, submission: ReviewSubmission):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if record["status"] != "review_required":
        raise HTTPException(409, "This job is not waiting for review")
    if not submission.text.strip() or len(submission.text) > 1_000_000:
        raise HTTPException(422, "Transcript must contain text and be smaller than 1 MB.")
    try:
        validate_review_srt(submission.text)
    except Exception as exc:
        raise HTTPException(422, f"Invalid SRT: {exc}") from exc

    review_dir = JOB_ROOT / job_id / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    if record["stage"] == "source_transcript_review":
        destination = review_dir / "approved_source_transcript.srt"
        artifact_name = "source_transcript"
        next_step = "translation"
    elif record["stage"] == "translation_review":
        destination = review_dir / "approved_target_translation.srt"
        artifact_name = "target_translation"
        next_step = "processing"
    else:
        raise HTTPException(409, "Unknown review stage")
    destination.write_text(submission.text, encoding="utf-8")
    artifact_path = JOB_ROOT / job_id / "artifacts" / f"{artifact_name}.srt"
    shutil.copy2(destination, artifact_path)
    store.update_artifacts(job_id, {artifact_name: str(artifact_path)})
    try:
        store.resume_review(job_id, next_step)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job_id, "status": "queued", "stage": "queued"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_log(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    path = JOB_ROOT / job_id / "process.log"
    if not path.is_file():
        raise HTTPException(404, "Job log not available yet")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="process.log")


@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    path = JOB_ROOT / job_id / "manifest.json"
    if not path.is_file():
        raise HTTPException(404, "Manifest not available")
    return FileResponse(path, media_type="application/json", filename="manifest.json")


@app.get("/api/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if name not in record["artifacts"]:
        raise HTTPException(404, "Artifact not available")
    path = Path(record["artifacts"][name]).resolve()
    allowed = (JOB_ROOT / job_id / "artifacts").resolve()
    if allowed not in path.parents or not path.is_file():
        raise HTTPException(404, "Artifact not available")
    media = "video/mp4" if path.suffix.lower() == ".mp4" else "application/x-subrip" if path.suffix.lower() == ".srt" else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)
