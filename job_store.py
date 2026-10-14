from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                review_step TEXT NOT NULL DEFAULT 'processing',
                target_language TEXT NOT NULL,
                source_language TEXT NOT NULL,
                operation TEXT NOT NULL DEFAULT 'combine',
                add_auto_captions INTEGER NOT NULL DEFAULT 0,
                input_path TEXT NOT NULL,
                input_name TEXT NOT NULL,
                artifacts TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "operation" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN operation TEXT NOT NULL DEFAULT 'combine'")
            if "review_step" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN review_step TEXT NOT NULL DEFAULT 'processing'")
            if "add_auto_captions" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN add_auto_captions INTEGER NOT NULL DEFAULT 0")
            # A worker process may have stopped while a job was active. Keep its
            # uploaded source and retry it from the beginning after restart.
            db.execute(
                "UPDATE jobs SET status='queued', stage='queued', updated_at=? WHERE status='running'",
                (self._now(),),
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(
        self,
        job_id: str,
        target: str,
        source: str,
        input_path: Path,
        input_name: str,
        operation: str = "combine",
        add_auto_captions: bool = False,
    ) -> None:
        now = self._now()
        review_step = "source_transcript" if operation in {"dubbing", "combine"} else "processing"
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO jobs(job_id,status,stage,review_step,target_language,source_language,operation,add_auto_captions,input_path,input_name,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, "queued", "queued", review_step, target, source, operation, int(add_auto_captions), str(input_path), input_name, now, now),
            )
        self._write_manifest(job_id)

    def pause_for_review(self, job_id: str, stage: str, next_step: str, artifacts: dict[str, str]) -> None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT artifacts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            combined = json.loads(row["artifacts"])
            combined.update(artifacts)
            db.execute(
                "UPDATE jobs SET status='review_required',stage=?,review_step=?,artifacts=?,updated_at=? WHERE job_id=?",
                (stage, next_step, json.dumps(combined), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def resume_review(self, job_id: str, next_step: str) -> None:
        with self._lock, self._connect() as db:
            updated = db.execute(
                "UPDATE jobs SET status='queued',stage='queued',review_step=?,updated_at=? WHERE job_id=? AND status='review_required'",
                (next_step, self._now(), job_id),
            ).rowcount
            if not updated:
                raise ValueError("This job is not waiting for review.")
        self._write_manifest(job_id)

    def claim(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                "UPDATE jobs SET status='running',stage='starting',updated_at=? WHERE job_id=?",
                (self._now(), row["job_id"]),
            )
            db.commit()
            claimed = dict(row)
        self._write_manifest(claimed["job_id"])
        return claimed

    def stage(self, job_id: str, stage: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='running',stage=?,updated_at=? WHERE job_id=?",
                (stage, self._now(), job_id),
            )
        self._write_manifest(job_id)

    def finish(self, job_id: str, artifacts: dict[str, str]) -> None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT artifacts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            combined = json.loads(row["artifacts"]) if row else {}
            combined.update(artifacts)
            db.execute(
                "UPDATE jobs SET status='completed',stage='completed',artifacts=?,error=NULL,updated_at=? WHERE job_id=?",
                (json.dumps(combined), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def update_artifacts(self, job_id: str, artifacts: dict[str, str]) -> None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT artifacts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            combined = json.loads(row["artifacts"])
            combined.update(artifacts)
            db.execute("UPDATE jobs SET artifacts=?,updated_at=? WHERE job_id=?",
                       (json.dumps(combined), self._now(), job_id))
        self._write_manifest(job_id)

    def fail(self, job_id: str, stage: str, error: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed',stage=?,error=?,updated_at=? WHERE job_id=?",
                (stage, error, self._now(), job_id),
            )
        self._write_manifest(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["artifacts"] = json.loads(result["artifacts"])
        return result

    def _write_manifest(self, job_id: str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        path = self.path.parent / "jobs" / job_id / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": record["job_id"],
            "status": record["status"],
            "stage": record["stage"],
            "review_step": record["review_step"],
            "target_language": record["target_language"],
            "source_language": record["source_language"],
            "operation": record["operation"],
            "add_auto_captions": bool(record["add_auto_captions"]),
            "input": record["input_name"],
            "artifacts": {
                name: f"/api/jobs/{job_id}/artifacts/{name}"
                for name in record["artifacts"]
            },
            "log_url": f"/api/jobs/{job_id}/logs",
            "error": record["error"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
