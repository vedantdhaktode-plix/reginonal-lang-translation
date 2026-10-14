from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .contracts import Artifact
from .media import ArtifactBackend


SERVICE_A_ROOT = Path(__file__).resolve().parent / "vendor" / "service_a"
SERVICE_B_ROOT = Path(__file__).resolve().parent / "vendor" / "service_b"
_SERVICE_A_LOCK = threading.RLock()

LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "bn": "Bengali",
    "te": "Telugu",
    "ur": "Urdu",
}


def _add_import_root(root: Path) -> None:
    root = Path(root).resolve()
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)


def _guard_service_a_namespaces(root: Path) -> None:
    root = root.resolve()
    for name in ("config", "database", "models", "pipeline"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None) if module else None
        module_path = Path(module_file).resolve() if module_file else None
        if module_path and not module_path.is_relative_to(root):
            raise RuntimeError(
                f"Module namespace '{name}' is already loaded from another package; "
                "run the hosted application in its own Python process."
            )


class LocalSubtitleTranslatorAdapter:
    """Runs Service A through its Python modules, without its CLI or fixed paths."""

    def __init__(self, backend: ArtifactBackend, service_root: Path = SERVICE_A_ROOT):
        self.backend = backend
        self.service_root = Path(service_root)

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> dict[str, Artifact]:
        target_name = LANGUAGES.get(target_language.lower(), target_language)
        with _SERVICE_A_LOCK, tempfile.TemporaryDirectory(prefix=f"service-a-{job_id}-") as temp:
            root = Path(temp)
            _add_import_root(self.service_root)
            _guard_service_a_namespaces(self.service_root)
            config_module = importlib.import_module("config")
            database_module = importlib.import_module("database.db")
            processor_module = importlib.import_module("pipeline.processor")
            input_path = self.backend.download(video, root / video.filename)
            env = {
                "INPUT_DIR": str(root / "incoming"),
                "PROCESSING_DIR": str(root / "processing"),
                "COMPLETED_DIR": str(root / "completed"),
                "FAILED_DIR": str(root / "failed"),
                "TEMP_DIR": str(root / "temp"),
                "LOG_DIR": str(root / "logs"),
                "METADATA_DIR": str(root / "metadata"),
                "SCRIPT_DIR": str(root / "scripts"),
                "DB_PATH": str(root / "jobs.db"),
                "FILE_STABILITY_SECONDS": "0.01",
                "SOURCE_LANGUAGE": source_language,
                "OCR_PROVIDER": os.getenv("OCR_PROVIDER", "gemini"),
                "OCR_TARGET_BOX": os.getenv("OCR_TARGET_BOX", "agent"),
                "TRANSLATION_PROVIDER": os.getenv("TRANSLATION_PROVIDER", "gemini"),
                "TRANSLATION_MODEL": os.getenv("TRANSLATION_MODEL", "gemini-3.5-flash-lite"),
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
                "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                "GEMINI_VISION_MODEL": os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite"),
            }
            previous = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                config = config_module.Config.load(target_language=target_name, source_language=source_language)
                db = database_module.Database(config.db_path)
                checksum = hashlib.sha256()
                with input_path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(block)
                digest = checksum.hexdigest()
                created = db.create(input_path, digest, source_language, target_name, force=True)
                if not created:
                    raise RuntimeError("Service A could not create a job")
                job_path = config.processing_dir / created / input_path.name
                job_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, job_path)
                db.update(created, "QUEUED", processing_path=str(job_path))
                job = db.get(created)
                if not processor_module.Processor(config, db).process(job):
                    failure = db.get(created)
                    failure_path = config.failed_dir / created / "failure.json"
                    details = failure_path.read_text(encoding="utf-8") if failure_path.is_file() else ""
                    if details:
                        print(f"[Service A failure.json]\n{details}", file=sys.stderr, flush=True)
                    raise RuntimeError(f"Service A failed: {failure.error_message}")
                completed = Path(db.get(created).output_path)
                translated_track = config.processing_dir / created / "translated.ass"
                if not translated_track.is_file():
                    translated_track = config.processing_dir / created / "translated.srt"
                visual_srt = root / "visual-subtitles.srt"
                if translated_track.is_file():
                    import pysubs2
                    document = pysubs2.load(str(translated_track), encoding="utf-8")
                    document.save(str(visual_srt), format_="srt", encoding="utf-8")
                else:
                    visual_srt.write_text("", encoding="utf-8")
                return {
                    "video": self.backend.upload(completed, filename="service-a-video.mp4", content_type="video/mp4"),
                    "visual_subtitles": self.backend.upload(visual_srt, filename="visual-subtitles.srt", content_type="application/x-subrip"),
                }
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class TextToAudioAdapter:
    """Runs Service B's full engine on Service A's rendered video."""

    def __init__(self, backend: ArtifactBackend, service_root: Path = SERVICE_B_ROOT):
        self.backend = backend
        self.service_root = Path(service_root)

    def run(
        self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str,
        source_transcript: Path | None = None, reviewed_translation: Path | None = None,
    ) -> dict[str, Artifact]:
        _add_import_root(self.service_root)
        cached = sys.modules.get("src")
        if cached is not None and not Path(cached.__file__).resolve().is_relative_to(self.service_root.resolve()):
            raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
        engine = importlib.import_module("src.engine")
        data_dir = Path(os.getenv("PIPELINE_DATA_DIR", "/data"))
        root = data_dir / "cache" / "service-b" / job_id
        root.mkdir(parents=True, exist_ok=True)
        source = self.backend.download(video, root / "service-a-video.mp4")
        output = engine.run_dubbing_engine(
            source_video_path=source,
            target_lang=target_language,
            source_lang=source_language,
            job_dir=root / "job",
            sync_model="lipsync-2",
            dry_run=os.getenv("DUBBING_DRY_RUN", "false").lower() == "true",
            supplied_source_transcript=source_transcript,
            reviewed_translation_srt=reviewed_translation,
        )
        output = Path(output)
        result: dict[str, Artifact] = {
            "video": self.backend.upload(output, filename="service-b-video.mp4", content_type="video/mp4")
        }
        audio_path = output.parent / f"dubbed_master_{target_language}.mp3"
        if audio_path.is_file():
            result["audio"] = self.backend.upload(
                audio_path, filename="service-b-audio.mp3", content_type="audio/mpeg"
            )
        audio_srt = output.parent / f"transcript_{target_language}.srt"
        if audio_srt.is_file():
            result["audio_subtitles"] = self.backend.upload(
                audio_srt, filename="audio-subtitles.srt", content_type="application/x-subrip"
            )
        return result


def prepare_source_transcript(video_path: Path, review_dir: Path, source_language: str) -> Path:
    """Transcribe, run the AI quality review, and save an editable SRT for approval."""
    _add_import_root(SERVICE_B_ROOT)
    cached = sys.modules.get("src")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(SERVICE_B_ROOT.resolve()):
        raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
    common = importlib.import_module("src.common")
    common.load_env(Path(__file__).resolve().parent / ".env")
    transcriber = importlib.import_module("src.transcriber")
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    generated, _ = transcriber.extract_transcript(
        video_path=Path(video_path), job_dir=review_dir / "transcription",
        source_lang=source_language,
    )
    destination = review_dir / "source_transcript.srt"
    shutil.copy2(generated, destination)
    verifier = importlib.import_module("src.script_verifier")
    report = verifier.verify_srt(destination.read_text(encoding="utf-8"), kind="source", source_language=source_language)
    (review_dir / "source_transcript_ai_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def prepare_translation_review(source_srt: Path, target_language: str, source_language: str, review_dir: Path) -> Path:
    """Translate the approved source cues and save an editable target SRT."""
    _add_import_root(SERVICE_B_ROOT)
    cached = sys.modules.get("src")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(SERVICE_B_ROOT.resolve()):
        raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
    common = importlib.import_module("src.common")
    common.load_env(Path(__file__).resolve().parent / ".env")
    parser = importlib.import_module("src.srt_parser")
    dubber = importlib.import_module("src.el_dubber")
    cues = parser.parse_srt(Path(source_srt).read_text(encoding="utf-8"))
    translated = dubber.translate_cues(cues, target_language, source_language)
    destination = Path(review_dir) / "target_translation.srt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(parser.render_srt([
        {"start_ms": cue.start_ms, "end_ms": cue.end_ms, "target_text": cue.text}
        for cue in translated
    ]), encoding="utf-8")
    verifier = importlib.import_module("src.script_verifier")
    report = verifier.verify_srt(
        destination.read_text(encoding="utf-8"), kind="translation",
        source_text=Path(source_srt).read_text(encoding="utf-8"),
        target_language=target_language, source_language=source_language,
    )
    (Path(review_dir) / "translation_ai_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def validate_review_srt(text: str) -> None:
    _add_import_root(SERVICE_B_ROOT)
    cached = sys.modules.get("src")
    if cached is not None and not Path(cached.__file__).resolve().is_relative_to(SERVICE_B_ROOT.resolve()):
        raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
    parser = importlib.import_module("src.srt_parser")
    parser.parse_srt(text)
