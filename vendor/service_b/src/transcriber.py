"""
Stage 1 — Transcript Extraction
Extracts word-level timestamped transcript from source video using faster-whisper (local).
Falls back to OpenAI Whisper API or a pre-supplied SRT if whisper is unavailable.
Cache key: sha256 of first 2MB of video file (fast, deterministic, avoids re-transcription).
"""
import hashlib
import json
import subprocess
import os
from pathlib import Path

from .common import PipelineError
from .srt_parser import parse_srt, stamp


def _video_hash(video_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Content hash of the complete video for reliable transcript cache keying."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for data in iter(lambda: f.read(chunk_size), b""):
            h.update(data)
    return h.hexdigest()[:16]


def _extract_audio_for_whisper(video_path: Path, out_wav: Path) -> Path:
    """Extract 16kHz mono WAV for Whisper input."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio extraction for transcription failed: {r.stderr}")
    return out_wav


def _transcribe_with_faster_whisper(audio_path: Path, source_lang: str = None) -> list[dict]:
    """
    Transcribe using faster-whisper (local, no API cost).
    Returns list of word dicts: {word, start, end}.
    Raises ImportError if faster-whisper is not installed.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel("small", compute_type="int8")
    lang = None if not source_lang or source_lang == "auto" else source_lang
    segments, _ = model.transcribe(
        str(audio_path), language=lang, word_timestamps=True,
        vad_filter=True, condition_on_previous_text=False,
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    words = []
    for seg in segments:
        # faster-whisper exposes segment-level quality signals. Drop likely
        # silence hallucinations before they can drive dubbing or scene logic.
        if getattr(seg, "no_speech_prob", 0.0) > 0.6 or getattr(seg, "avg_logprob", 0.0) < -1.0:
            continue
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
        else:
            # Fallback: use segment-level timing
            words.append({"word": seg.text.strip(), "start": seg.start, "end": seg.end})
    return words


def _get_media_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except (ValueError, TypeError):
        return 0.0


def _transcribe_with_gemini(audio_path: Path, source_lang: str = None) -> tuple[str, list[dict]]:
    """Use Gemini Multimodal Audio transcription to capture full verbatim timestamped speech."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise PipelineError("GEMINI_API_KEY is empty")
    model = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
    client = genai.Client(api_key=api_key)

    api_audio = Path(audio_path).with_suffix(".mp3")
    compact = subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(audio_path), "-vn", "-ac", "1", "-b:a", "128k", str(api_audio)],
        capture_output=True, text=True,
    )
    if compact.returncode:
        raise PipelineError("Could not prepare audio for Gemini transcription.")

    uploaded = None
    try:
        uploaded = client.files.upload(file=str(api_audio))
        lang_hint = f" The spoken language is {source_lang}." if source_lang and source_lang.lower() != "auto" else ""
        prompt = (
            f"Listen to the entire audio carefully from beginning to end.{lang_hint}\n"
            "Generate a complete, verbatim timestamped transcription in standard SRT subtitle format.\n"
            "Do NOT summarize, do NOT omit or skip any dialogue or sentences, even if there is background noise or music.\n"
            "Ensure timestamp intervals reflect when each sentence or phrase is actually spoken.\n"
            "Output ONLY valid SRT subtitle text."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
        )
        text = (resp.text or "").strip()
        if "```srt" in text:
            text = text.split("```srt", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        cues = parse_srt(text)
        if not cues:
            raise PipelineError("Gemini transcription did not produce any valid SRT cues.")
        words = [
            {"word": c.text.replace("\n", " ").strip(),
             "start": c.start_ms / 1000.0,
             "end": c.end_ms / 1000.0}
            for c in cues
        ]
        canonical_srt = "\n\n".join(
            f"{c.id}\n{stamp(c.start_ms)} --> {stamp(c.end_ms)}\n{c.text}"
            for c in cues
        ) + "\n"
        return canonical_srt, words
    except Exception as exc:
        raise PipelineError(f"Gemini transcription failed: {exc}") from exc
    finally:
        api_audio.unlink(missing_ok=True)
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def _transcribe_with_openai(audio_path: Path, source_lang: str = None) -> list[dict]:
    """
    Use OpenAI Whisper transcription (whisper-1).
    Chunks audio into short windows to prevent Whisper's 30-second sliding-window
    hallucination filter from dropping code-switched or mixed-language speech.
    """
    import requests

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise PipelineError("OPENAI_API_KEY is empty")
    model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1").strip() or "whisper-1"

    dur = _get_media_duration(audio_path)
    # If audio is short (<= 10s), single call is fine; otherwise chunk into 7.5s segments.
    chunk_size = 7.5
    if dur <= 10.0:
        chunk_offsets = [(0.0, dur)]
    else:
        chunk_offsets = []
        cur = 0.0
        while cur < dur:
            chunk_offsets.append((cur, min(cur + chunk_size, dur)))
            cur += chunk_size

    words = []
    tmp_chunk = Path(audio_path).parent / f"chunk_{os.getpid()}.mp3"
    try:
        for offset, end_sec in chunk_offsets:
            chunk_len = end_sec - offset
            if chunk_len < 0.3:
                continue
            cmd = [
                "ffmpeg", "-y", "-nostdin", "-v", "error",
                "-ss", f"{offset:.3f}", "-to", f"{end_sec:.3f}",
                "-i", str(audio_path),
                "-vn", "-ac", "1", "-b:a", "64k",
                str(tmp_chunk),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not tmp_chunk.exists() or tmp_chunk.stat().st_size == 0:
                continue

            data = [
                ("model", model),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
            ]
            if source_lang and source_lang.lower() != "auto":
                data.append(("language", source_lang))

            try:
                with tmp_chunk.open("rb") as stream:
                    resp = requests.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        data=data,
                        files={"file": (tmp_chunk.name, stream, "audio/mpeg")},
                        timeout=(30, 90),
                    )
                if resp.status_code >= 400:
                    continue
                payload = resp.json()
            except requests.RequestException:
                continue

            if "words" in payload and isinstance(payload["words"], list) and payload["words"]:
                for w in payload["words"]:
                    wt = str(w.get("word", "")).strip()
                    if wt:
                        words.append({
                            "word": wt,
                            "start": round(offset + float(w.get("start", 0)), 3),
                            "end": round(offset + float(w.get("end", 0)), 3),
                        })
            elif "segments" in payload and isinstance(payload["segments"], list):
                for s in payload["segments"]:
                    st = str(s.get("text", "")).strip()
                    if st and s.get("end", 0) > s.get("start", 0):
                        words.append({
                            "word": st,
                            "start": round(offset + float(s.get("start", 0)), 3),
                            "end": round(offset + float(s.get("end", 0)), 3),
                        })
            elif payload.get("text", "").strip():
                words.append({
                    "word": payload["text"].strip(),
                    "start": round(offset, 3),
                    "end": round(end_sec, 3),
                })
    finally:
        tmp_chunk.unlink(missing_ok=True)

    if not words:
        raise PipelineError("OpenAI Whisper transcription returned no speech for this audio.")
    return words


def _words_to_srt(words: list[dict]) -> str:
    """Convert word list to SRT cue format (one cue per natural pause group)."""

    def _fmt(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int(round((secs % 1) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # Group words into sentences by gaps, natural pauses, or punctuation
    groups = []
    current: list[dict] = []
    for w in words:
        if current:
            gap = float(w["start"]) - float(current[-1]["end"])
            text = str(current[-1]["word"]).strip()
            ends_sentence = text.endswith((".", "!", "?", "।", ";"))
            duration = float(current[-1]["end"]) - float(current[0]["start"])
            # Break if:
            # 1. Clear pause (gap >= 0.35s)
            # 2. Sentence-ending punctuation
            # 3. Reasonable sentence duration reached (>= 3.2s) with any minor pause (gap >= 0.12s)
            # 4. Maximum block duration (>= 5.0s) or word limit (>= 12 words) to prevent runaways
            if (gap >= 0.35 or ends_sentence or
                (duration >= 3.2 and gap >= 0.12) or
                duration >= 5.0 or len(current) >= 12):
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)

    lines = []
    for i, group in enumerate(groups, 1):
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"] for w in group).strip()
        lines.append(f"{i}\n{_fmt(start)} --> {_fmt(end)}\n{text}\n")
    return "\n".join(lines)


def extract_transcript(
    video_path: Path,
    job_dir: Path,
    source_lang: str = "auto",
    supplied_srt: Path = None,
) -> tuple[Path, list[dict]]:
    """
    Main entry point for Stage 1.

    Returns:
        (srt_path, words)
        srt_path  — Path to the saved SRT file
        words     — List of {word, start, end} dicts
    """
    video_path = Path(video_path).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    lang_tag = source_lang if source_lang and source_lang != "auto" else "src"
    srt_path = job_dir / f"transcript_{lang_tag}.srt"
    words_path = job_dir / f"transcript_{lang_tag}.json"
    hash_path = job_dir / f"transcript_{lang_tag}.hash"

    # --- Use supplied SRT if provided ---
    if supplied_srt and Path(supplied_srt).exists():
        text = Path(supplied_srt).read_text(encoding="utf-8")
        cues = parse_srt(text)
        words = [
            {"word": c.text.replace("\n", " ").strip(),
             "start": c.start_ms / 1000.0,
             "end": c.end_ms / 1000.0}
            for c in cues
        ]
        srt_path.write_text(text, encoding="utf-8")
        words_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Transcriber] Using supplied SRT: {supplied_srt.name} ({len(cues)} cues)")
        return srt_path, words

    # --- Determine transcription provider ---
    provider = os.getenv("TRANSCRIPTION_PROVIDER", "openai").strip().lower()
    if provider == "gemini":
        model = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
    else:
        model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1").strip() or "whisper-1"

    vid_hash = hashlib.sha256(
        f"{_video_hash(video_path)}|{provider}|{model}|{source_lang}".encode("utf-8")
    ).hexdigest()[:16]
    if srt_path.exists() and words_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == vid_hash:
            words = json.loads(words_path.read_text(encoding="utf-8"))
            if not words:
                raise PipelineError("Cached transcript contains no speech; refusing to continue.")
            print(f"[Transcriber] Cache hit — {len(words)} cues loaded from {srt_path.name}")
            return srt_path, words

    # --- Transcribe ---
    wav_path = job_dir / "whisper_input.wav"
    _extract_audio_for_whisper(video_path, wav_path)

    total_media_dur = _get_media_duration(video_path)
    srt_text = ""
    words = []

    if provider == "gemini":
        print(f"[Transcriber] Running Gemini Audio transcription ({model}, language={source_lang or 'auto'})...")
        srt_text, words = _transcribe_with_gemini(wav_path, source_lang)
        print(f"[Transcriber] Gemini transcribed {len(words)} cues.")
    else:
        print(f"[Transcriber] Running OpenAI Whisper API transcription ({model}, language={source_lang or 'auto'})...")
        words = _transcribe_with_openai(wav_path, source_lang)
        print(f"[Transcriber] OpenAI Whisper transcribed {len(words)} items.")
        srt_text = _words_to_srt(words)

        # Check if Whisper output is severely truncated or failed to segment into cues
        max_end = max((float(w.get("end", 0)) for w in words), default=0.0)
        cues_count = len([c for c in srt_text.strip().split("\n\n") if c.strip()])
        if total_media_dur >= 10.0 and (max_end < 0.5 * total_media_dur or cues_count <= 1):
            print(f"[Transcriber] WARNING: OpenAI Whisper output unsegmented or truncated ({cues_count} cues, coverage: {max_end:.1f}s / {total_media_dur:.1f}s). Falling back to Gemini Audio...")
            try:
                gemini_srt, gemini_words = _transcribe_with_gemini(wav_path, source_lang)
                if gemini_words and max((float(w.get("end", 0)) for w in gemini_words), default=0.0) >= 0.5 * total_media_dur:
                    srt_text, words = gemini_srt, gemini_words
                    print(f"[Transcriber] Fallback Gemini transcribed {len(words)} cues successfully.")
            except Exception as e:
                print(f"[Transcriber] Fallback to Gemini failed: {e}; keeping Whisper output.")

    # Clean up temp audio
    wav_path.unlink(missing_ok=True)

    # --- Save ---
    if not any(float(w.get("end", 0)) > float(w.get("start", 0)) for w in words):
        raise PipelineError("Transcription returned text without usable timestamps; refusing to invent cue timing.")
    srt_path.write_text(srt_text, encoding="utf-8")
    words_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_path.write_text(vid_hash, encoding="utf-8")
    print(f"[Transcriber] Saved transcript → {srt_path.name}")
    if not words:
        raise PipelineError("Speech transcription produced no reliable words; refusing to continue with a fabricated transcript.")
    return srt_path, words
