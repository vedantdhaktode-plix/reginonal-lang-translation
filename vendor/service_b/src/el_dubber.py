"""
Stage 3 — ElevenLabs Audio Generation
Two modes depending on whether ElevenLabs Dubbing API supports the target language:

MODE A — Dubbing API (auto, supported languages only):
  POST /v1/dubbing → full video → dubbed audio + timed SRT
  Supported: hi, es, fr, de, pt, ja, ko, zh, ar, etc.
  NOT supported: mr (Marathi) — falls through to Mode B automatically.

MODE B — Voice-Clone TTS (for unsupported languages like Marathi):
  1. Extract voice sample from video (ffmpeg)
  2. Clone voice via ElevenLabs Instant Voice Clone
  3. Translate transcript using a supplied translated SRT or Google Translate
  4. TTS each cue using eleven_v4 / eleven_v3 (both support Marathi)
  5. Assemble time-fitted audio matching source video duration

Cache: per (video_hash, target_lang). Never re-charges on re-run.

Supported Dubbing API languages (partial, check EL docs for full list):
  af, ar, bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, id, it, ja, ko,
  lt, lv, ms, nl, no, pl, pt, ro, ru, sk, sl, sv, ta, tr, uk, vi, zh, hi
NOT supported by Dubbing API (use Voice-Clone TTS instead):
  mr, gu, bn, pa, ml, te, kn, ur, ne, si, my, km, lo, ...
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from .common import PipelineError
from .srt_parser import parse_srt, Cue

# Languages supported by ElevenLabs Dubbing API
# (based on validation_error testing — mr returns unsupported_target_language)
EL_DUBBING_SUPPORTED = {
    "af", "ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "es-419",
    "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
    "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl",
    "sq", "sr", "sv", "sw", "ta", "th", "tl", "tr", "uk", "vi", "zh",
}

# ElevenLabs TTS models that support Marathi and other Indian languages
EL_V4_MODEL = "eleven_v4"           # Best quality, Marathi supported
EL_V4_TURBO = "eleven_v4_turbo"     # Faster, Marathi supported
EL_V3_MODEL = "eleven_v3"           # Also Marathi supported
TTS_ASSEMBLY_VERSION = "cue-fit-v2"


def _video_hash(video_path: Path) -> str:
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
    h.update(str(video_path.stat().st_size).encode())
    return h.hexdigest()[:16]


def _extract_voice_sample(video_path: Path, out_wav: Path, duration_s: float = 20.0) -> Path:
    """Extract first N seconds of clean speech from video for voice cloning."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(video_path),
        "-vn", "-ss", "1.0", "-t", str(duration_s),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Voice sample extraction failed: {r.stderr}")
    return out_wav


def _ensure_voice_clone(sample_wav: Path, voice_name: str, api_key: str) -> str:
    """Get existing clone by name, or create one. Returns voice_id."""
    headers = {"xi-api-key": api_key}
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=30)
    if r.ok:
        for v in r.json().get("voices", []):
            if v["name"] == voice_name:
                print(f"[Dubber] Reusing voice clone '{voice_name}' (ID: {v['voice_id']})")
                return v["voice_id"]

    print(f"[Dubber] Creating voice clone '{voice_name}' from {sample_wav.name}...")
    with open(sample_wav, "rb") as f:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers=headers,
            data={"name": voice_name, "description": "Auto-cloned for dubbing"},
            files={"files": ("sample.wav", f, "audio/wav")},
            timeout=60,
        )
    if not resp.ok:
        raise PipelineError(f"Voice clone failed: HTTP {resp.status_code} — {resp.text[:300]}")
    vid = resp.json()["voice_id"]
    print(f"[Dubber] Created voice clone ID: {vid}")
    return vid


def _translate_cues_google(cues: list[Cue], target_lang: str) -> list[Cue]:
    """Translate cue texts using unofficial Google Translate (no API key needed)."""
    from urllib.parse import quote
    translated = []
    for cue in cues:
        text = cue.text.replace("\n", " ").strip()
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=auto&tl={target_lang}&dt=t&q={quote(text)}"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                data = r.json()
                translated_text = "".join(part[0] for part in data[0] if part[0])
            else:
                translated_text = text
        except Exception:
            translated_text = text
        new_cue = Cue(cue.id, cue.start_ms, cue.end_ms, translated_text)
        translated.append(new_cue)
        print(f"  [{cue.id}] {text[:40]} → {translated_text[:40]}")
    return translated


def _tts_cue(voice_id: str, text: str, out_path: Path, model: str, api_key: str) -> Path:
    """Synthesize a single cue to audio file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.85,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers=headers, json=payload, timeout=60,
    )
    if not r.ok:
        raise PipelineError(f"TTS failed for '{text[:30]}...': HTTP {r.status_code} — {r.text[:200]}")
    out_path.write_bytes(r.content)
    return out_path


def _prepare_tts_cues(
    cues: list[Cue],
    total_duration_s: float,
    merge_gap_ms: int = 350,
    max_block_ms: int = 15_000,
) -> list[Cue]:
    """
    Convert dense OCR/word-level subtitle cues into speakable, non-overlapping
    narration blocks.

    Visual subtitle extraction can produce 0-250 ms cues, including several cues
    with the same timestamp. A TTS voice cannot speak those independently without
    overlap. Consecutive cues are therefore grouped until a real pause is found
    (or max_block_ms is reached). The returned cue windows never overlap.
    """
    limit_ms = max(1, int(round(total_duration_s * 1000)))
    ordered: list[Cue] = []

    for cue in sorted(cues, key=lambda item: (int(item.start_ms), int(item.id))):
        text = " ".join(str(cue.text).replace("\n", " ").split())
        if not text:
            continue

        start_ms = max(0, min(limit_ms - 1, int(cue.start_ms)))
        end_ms = max(start_ms, min(limit_ms, int(cue.end_ms)))
        ordered.append(Cue(cue.id, start_ms, end_ms, text))

    if not ordered:
        raise PipelineError("No non-empty TTS cues were supplied.")

    groups: list[list[Cue]] = []
    current: list[Cue] = []

    for cue in ordered:
        if not current:
            current = [cue]
            continue

        block_start = int(current[0].start_ms)
        previous_end = max(int(current[-1].start_ms), int(current[-1].end_ms))
        proposed_end = max(int(cue.start_ms), int(cue.end_ms))
        gap_ms = max(0, int(cue.start_ms) - previous_end)

        if gap_ms <= merge_gap_ms and proposed_end - block_start <= max_block_ms:
            current.append(cue)
        else:
            groups.append(current)
            current = [cue]

    if current:
        groups.append(current)

    prepared: list[Cue] = []
    for index, group in enumerate(groups):
        start_ms = int(group[0].start_ms)
        next_start_ms = (
            int(groups[index + 1][0].start_ms)
            if index + 1 < len(groups)
            else limit_ms
        )
        end_ms = max(start_ms + 1, min(limit_ms, next_start_ms))

        parts: list[str] = []
        previous_text = None
        for cue in group:
            text = cue.text.strip()
            folded = text.casefold()
            if text and folded != previous_text:
                parts.append(text)
                previous_text = folded

        if parts and end_ms > start_ms:
            prepared.append(Cue(group[0].id, start_ms, end_ms, " ".join(parts)))

    if not prepared:
        raise PipelineError("TTS cue normalization produced no speakable windows.")

    if len(prepared) != len(ordered):
        print(
            f"[Dubber] Consolidated {len(ordered)} dense subtitle cues into "
            f"{len(prepared)} non-overlapping narration blocks."
        )
    return prepared


def _probe_audio_duration(audio_path: Path) -> float:
    """Return decoded audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(
            f"Could not read TTS cue duration for {audio_path.name}: {result.stderr}"
        )
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            f"Invalid TTS cue duration for {audio_path.name}: {result.stdout!r}"
        ) from exc
    if duration <= 0:
        raise PipelineError(f"TTS cue is empty: {audio_path}")
    return duration


def _atempo_filters(speed: float) -> list[str]:
    """
    Build an FFmpeg atempo chain. Each stage stays within the portable 0.5-2.0
    range, while the whole chain can fit unusually long translations.
    """
    if speed <= 1.0005:
        return []

    filters: list[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.8f}")
    return filters


def _assemble_tts_audio(
    cues: list[Cue], cue_audio_paths: list[Path],
    total_duration_s: float, out_path: Path
) -> Path:
    """
    Assemble one full-length, frame-aligned narration track.

    Every cue is time-fitted to its exclusive SRT window before it is delayed and
    mixed. This prevents a long Marathi utterance from continuing underneath the
    following cue. The limiter is only a final safety net; correct timing comes
    from the per-cue trim/pad operation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    filter_parts: list[str] = []
    mix_inputs: list[str] = []
    input_index = 0

    for cue, audio_path in zip(cues, cue_audio_paths):
        audio_path = Path(audio_path)
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            continue

        start_ms = max(0, int(cue.start_ms))
        end_ms = min(int(round(total_duration_s * 1000)), int(cue.end_ms))
        if end_ms <= start_ms:
            print(
                f"[Dubber] Skipping cue {cue.id}: invalid window "
                f"{start_ms}ms -> {end_ms}ms"
            )
            continue

        window_s = (end_ms - start_ms) / 1000.0
        guard_s = min(0.060, window_s * 0.15)
        speech_window_s = max(0.040, window_s - guard_s)
        audio_duration_s = _probe_audio_duration(audio_path)
        speed = max(1.0, audio_duration_s / speech_window_s)

        if speed > 1.20:
            print(
                f"[Dubber] Cue {cue.id}: fitting {audio_duration_s:.2f}s into "
                f"{speech_window_s:.2f}s (speed {speed:.2f}x)"
            )

        chain = [
            "aresample=48000",
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=mono",
            "asetpts=PTS-STARTPTS",
            *_atempo_filters(speed),
            f"atrim=duration={speech_window_s:.6f}",
            "asetpts=PTS-STARTPTS",
        ]

        fade_s = min(0.020, speech_window_s / 4.0)
        if fade_s > 0.001:
            fade_out_start = max(0.0, speech_window_s - fade_s)
            chain.extend([
                f"afade=t=in:st=0:d={fade_s:.6f}",
                f"afade=t=out:st={fade_out_start:.6f}:d={fade_s:.6f}",
            ])

        chain.extend([
            f"apad=pad_dur={window_s:.6f}",
            f"atrim=duration={window_s:.6f}",
            f"adelay={start_ms}:all=1",
        ])

        inputs.extend(["-i", str(audio_path)])
        label = f"a{input_index}"
        filter_parts.append(f"[{input_index}:a]{','.join(chain)}[{label}]")
        mix_inputs.append(f"[{label}]")
        input_index += 1

    if not mix_inputs:
        raise PipelineError("No valid cue audio files were generated.")

    mix_filter = (
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=longest:"
          "dropout_transition=0:normalize=0,"
          "alimiter=limit=0.95,"
        + f"apad=pad_dur={total_duration_s:.6f},"
          f"atrim=duration={total_duration_s:.6f},"
          "asetpts=PTS-STARTPTS[out]"
    )
    full_filter = ";".join(filter_parts + [mix_filter])

    cmd = [
        "ffmpeg", "-y", "-nostdin",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-ar", "48000", "-ac", "1",
        "-acodec", "libmp3lame", "-q:a", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(f"TTS audio assembly failed:\n{result.stderr}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — ElevenLabs Dubbing API
# ──────────────────────────────────────────────────────────────────────────────

class ElevenLabsDubber:
    BASE_URL = "https://api.elevenlabs.io/v1/dubbing"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")
        self.headers = {"xi-api-key": self.api_key}

    def create_job(self, video_path: Path, target_lang: str,
                   source_lang: str = "auto", num_speakers: int = 0) -> str:
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        content_type = "video/mp4" if suffix in (".mp4", ".mov", ".mkv") else "audio/mpeg"
        data = {
            "target_lang": target_lang,
            "source_lang": source_lang or "auto",
            "num_speakers": str(int(num_speakers)),
            "watermark": "false",
        }
        print(f"[Dubber] Submitting to ElevenLabs Dubbing ({source_lang} → {target_lang})...")
        with open(video_path, "rb") as f:
            r = requests.post(self.BASE_URL, headers=self.headers, data=data,
                              files={"file": (video_path.name, f, content_type)}, timeout=300)
        if not r.ok:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
            raise PipelineError(f"ElevenLabs Dubbing create failed: HTTP {r.status_code} — {detail}")
        dubbing_id = r.json().get("dubbing_id")
        if not dubbing_id:
            raise PipelineError("ElevenLabs returned no dubbing_id.")
        print(f"[Dubber] Job created: {dubbing_id}")
        return dubbing_id

    def wait_for_completion(self, dubbing_id: str, poll_interval: int = 12, timeout: int = 1800) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            r = requests.get(f"{self.BASE_URL}/{dubbing_id}", headers=self.headers, timeout=30)
            if not r.ok:
                raise PipelineError(f"ElevenLabs status check failed: HTTP {r.status_code}")
            data = r.json()
            status = data.get("status", "")
            print(f"  [Dubber] Status: {status} ({int(time.time()-start)}s)...")
            if status == "dubbed":
                return data
            if status == "failed":
                raise PipelineError(f"ElevenLabs Dubbing failed: {data.get('error','unknown')}")
            time.sleep(poll_interval)
        raise PipelineError(f"ElevenLabs Dubbing timed out after {timeout}s.")

    def download_audio(self, dubbing_id: str, target_lang: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/audio/{target_lang}"
        with requests.get(url, headers=self.headers, stream=True, timeout=120) as r:
            if not r.ok:
                raise PipelineError(f"Failed to download dubbed audio: HTTP {r.status_code}")
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
        print(f"[Dubber] Dubbed audio → {output_path.name} ({output_path.stat().st_size//1024}KB)")
        return output_path

    def download_transcript(self, dubbing_id: str, target_lang: str, srt_path: Path) -> list:
        srt_path = Path(srt_path)
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/transcript/{target_lang}?format_type=srt"
        r = requests.get(url, headers=self.headers, timeout=30)
        if not r.ok:
            raise PipelineError(f"Failed to download transcript: HTTP {r.status_code}")
        srt_path.write_text(r.text, encoding="utf-8")
        cues = parse_srt(r.text)
        print(f"[Dubber] Transcript → {srt_path.name} ({len(cues)} cues)")
        return cues


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_dubbing(
    video_path: Path,
    job_dir: Path,
    target_lang: str,
    source_lang: str = "auto",
    num_speakers: int = 0,
    source_cues: list = None,
    translated_srt: Path = None,
    tts_model: str = EL_V4_MODEL,
) -> tuple[Path, Path, list]:
    """
    Stage 3 unified entry point.
    Auto-selects Mode A (Dubbing API) or Mode B (Voice-Clone TTS) based on language support.

    Args:
        video_path      : Source video
        job_dir         : Output directory
        target_lang     : e.g. 'mr', 'hi', 'en'
        source_lang     : Source language or 'auto'
        num_speakers    : For Dubbing API
        source_cues     : Parsed source transcript cues (for Mode B translation)
        translated_srt  : Pre-translated SRT in target language (skips translation step)
        tts_model       : ElevenLabs TTS model for Mode B (default: eleven_v4)

    Returns:
        (dubbed_audio_path, dubbed_srt_path, cues)
    """
    video_path = Path(video_path).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")

    dubbed_audio = job_dir / f"dubbed_master_{target_lang}.mp3"
    dubbed_srt = job_dir / f"transcript_{target_lang}.srt"
    job_id_path = job_dir / f"el_dubbing_id_{target_lang}.txt"

    # Cache check — Mode B also requires the current timing-assembly version.
    # This invalidates master audio created by the old overlapping amix pipeline.
    REAL_AUDIO_MIN_BYTES = 500 * 1024  # 500KB
    assembly_version_path = job_dir / f"tts_assembly_version_{target_lang}.txt"
    mode_b_cache_current = (
        assembly_version_path.exists()
        and assembly_version_path.read_text(encoding="utf-8").strip()
        == TTS_ASSEMBLY_VERSION
    )
    if (dubbed_audio.exists()
            and dubbed_audio.stat().st_size >= REAL_AUDIO_MIN_BYTES
            and dubbed_srt.exists()
            and dubbed_srt.stat().st_size > 100
            and "PLACEHOLDER" not in dubbed_audio.name
            and (target_lang in EL_DUBBING_SUPPORTED or mode_b_cache_current)):
        print(f"[Dubber] Cache hit — reusing {dubbed_audio.name} ({dubbed_audio.stat().st_size // 1024}KB)")
        cues = parse_srt(dubbed_srt.read_text(encoding="utf-8")) if dubbed_srt.exists() else []
        return dubbed_audio, dubbed_srt, cues


    # ── Mode A: ElevenLabs Dubbing API (auto dub, voice-preserved) ────────────
    if target_lang in EL_DUBBING_SUPPORTED:
        print(f"[Dubber] Mode A: ElevenLabs Dubbing API ({target_lang} is supported)")
        client = ElevenLabsDubber(api_key)

        dubbing_id = None
        if job_id_path.exists():
            dubbing_id = job_id_path.read_text().strip()
            print(f"[Dubber] Resuming existing job: {dubbing_id}")

        if not dubbing_id:
            dubbing_id = client.create_job(video_path, target_lang, source_lang, num_speakers)
            job_id_path.write_text(dubbing_id, encoding="utf-8")

        client.wait_for_completion(dubbing_id)
        client.download_audio(dubbing_id, target_lang, dubbed_audio)
        cues = client.download_transcript(dubbing_id, target_lang, dubbed_srt)
        return dubbed_audio, dubbed_srt, cues

    # ── Mode B: Voice-Clone TTS (for unsupported languages like Marathi) ──────
    print(f"[Dubber] Mode B: Voice-Clone TTS ({target_lang} not supported by Dubbing API)")
    print(f"[Dubber] Using ElevenLabs TTS model: {tts_model}")

    # Step B1: Get translated cues
    target_cues: list[Cue] = []
    if translated_srt and Path(translated_srt).exists():
        print(f"[Dubber] Using supplied translated SRT: {Path(translated_srt).name}")
        target_cues = parse_srt(Path(translated_srt).read_text(encoding="utf-8"))
    elif source_cues:
        print(f"[Dubber] Translating {len(source_cues)} cues to {target_lang} via Google Translate...")
        target_cues = _translate_cues_google(source_cues, target_lang)
        # Save translated SRT
        srt_lines = []
        for i, cue in enumerate(target_cues, 1):
            def _fmt(ms):
                h, rem = divmod(int(ms), 3600000)
                m, rem = divmod(rem, 60000)
                s, ms_ = divmod(rem, 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"
            srt_lines.append(f"{i}\n{_fmt(cue.start_ms)} --> {_fmt(cue.end_ms)}\n{cue.text}\n")
        dubbed_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    else:
        raise PipelineError(
            f"Language '{target_lang}' is not supported by ElevenLabs Dubbing API.\n"
            f"For Mode B (Voice-Clone TTS), supply one of:\n"
            f"  --transcript <source_srt>  (will be auto-translated)\n"
            f"  --reviewed-srt <marathi_srt>  (pre-translated Marathi SRT)"
        )

    # Step B2: Extract voice sample & clone
    sample_wav = job_dir / "voice_sample.wav"
    if not sample_wav.exists() or sample_wav.stat().st_size < 1000:
        _extract_voice_sample(video_path, sample_wav, duration_s=20.0)

    voice_name = f"DubClone_{video_path.stem[:20]}"
    voice_id = _ensure_voice_clone(sample_wav, voice_name, api_key)

    # Step B3: Consolidate dense/zero-duration visual cues into narration blocks.
    from .scene_analyzer import get_video_info
    info = get_video_info(video_path)
    total_dur = info["duration"]
    tts_cues = _prepare_tts_cues(target_cues, total_dur)

    # Include the normalized text hash in the cache key. Old one-word TTS clips
    # must not be reused for a newly consolidated narration block.
    cue_dir = job_dir / f"tts_cues_{target_lang}_{TTS_ASSEMBLY_VERSION}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    cue_audio_paths = []
    print(f"[Dubber] Generating {len(tts_cues)} fitted TTS blocks in {target_lang}...")
    for cue in tts_cues:
        text = cue.text.strip()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        cue_out = cue_dir / f"cue_{cue.id:04d}_{text_hash}.mp3"
        if text and not cue_out.exists():
            _tts_cue(voice_id, text, cue_out, tts_model, api_key)
        cue_audio_paths.append(cue_out)

    # Step B4: Fit every narration block to its exclusive timeline window,
    # then assemble one master track with no simultaneous cue speech.
    print(f"[Dubber] Assembling {len(cue_audio_paths)} cues into {total_dur:.2f}s audio track...")
    _assemble_tts_audio(tts_cues, cue_audio_paths, total_dur, dubbed_audio)
    assembly_version_path.write_text(TTS_ASSEMBLY_VERSION, encoding="utf-8")
    print(f"[Dubber] Mode B complete → {dubbed_audio.name} ({dubbed_audio.stat().st_size//1024}KB)")

    return dubbed_audio, dubbed_srt, target_cues
