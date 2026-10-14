"""
Unified Dubbing Engine — Orchestrates all 5 stages:

  Stage 1: Transcript extraction (Whisper)
  Stage 2: Scene classification (YuNet + SFace)
  Stage 3: ElevenLabs Dubbing (full-video, speaker-consistent audio)
  Stage 4: Targeted lip-sync (Sync Labs, A-Roll only)
  Stage 5: Frame-locked master stitch

Entry point: run_dubbing_engine(...)
"""
import json
from pathlib import Path

from .common import PipelineError, load_env
from .scene_analyzer import get_video_info
from .transcriber import extract_transcript
from .classifier import classify_video
from .el_dubber import run_dubbing
from .lipsyncer import process_shots
from .stitcher import stitch_segments


ROOT = Path(__file__).resolve().parent.parent


def run_dubbing_engine(
    source_video_path: Path,
    target_lang: str,
    source_lang: str = "auto",
    job_dir: Path = None,
    sync_model: str = "lipsync-2",
    scene_threshold: float = 32.0,
    target_frame: float = None,
    target_point: tuple = None,
    reference_image: Path = None,
    supplied_transcript: Path = None,
    supplied_source_transcript: Path = None,
    reviewed_translation_srt: Path = None,
    num_speakers: int = 0,
    dry_run: bool = False,
    skip_stages: list[int] = None,
) -> Path:
    """
    Full 5-stage dubbing pipeline.

    Args:
        source_video_path   : Input video file
        target_lang         : BCP-47 target language code (e.g. 'mr', 'hi', 'en')
        source_lang         : Source language code or 'auto'
        job_dir             : Output directory (auto-named from video stem if None)
        sync_model          : Sync Labs model ('lipsync-2' default)
        scene_threshold     : Camera cut detection sensitivity
        target_frame        : Timestamp (s) in video showing the target speaker
        target_point        : (x, y) pixel inside her face for disambiguation
        reference_image     : Path to reference photo of speaker
        supplied_transcript : Pre-made SRT to use instead of transcribing
        num_speakers        : Expected speaker count (0 = auto-detect)
        dry_run             : Simulate pipeline without paid API calls
        skip_stages         : List of stage numbers to skip (e.g. [3] reuses existing dub)

    Returns:
        Path to final assembled video
    """
    load_env(ROOT / ".env")
    skip_stages = set(skip_stages or [])

    source_video = Path(source_video_path).resolve()
    if not source_video.exists():
        raise PipelineError(f"Source video not found: {source_video}")

    if job_dir is None:
        job_dir = ROOT / "output" / f"dub_{source_video.stem}_{target_lang}"
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(source_video)
    fps = info["fps"]
    total_frames = info["frame_count"]
    duration = info["duration"]

    print(f"\n{'='*60}")
    print(f"[Engine] Video: {source_video.name}")
    print(f"[Engine] Resolution: {info['width']}x{info['height']} @ {fps:.3f}fps")
    print(f"[Engine] Duration: {duration:.2f}s ({total_frames} frames)")
    print(f"[Engine] Target language: {target_lang} | Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Stage 1: Transcript
    # ---------------------------------------------------------------
    words = []
    srt_path = None
    if 1 not in skip_stages:
        print("[Engine] ── Stage 1: Transcript Extraction ──")
        srt_path, words = extract_transcript(
            video_path=source_video,
            job_dir=job_dir,
            source_lang=source_lang,
            supplied_srt=supplied_source_transcript or supplied_transcript,
        )
    else:
        lang_tag = source_lang if source_lang and source_lang != "auto" else "src"
        srt_path = job_dir / f"transcript_{lang_tag}.srt"
        words_path = job_dir / f"transcript_{lang_tag}.json"
        if words_path.exists():
            words = json.loads(words_path.read_text(encoding="utf-8"))
            print(f"[Engine] Stage 1 skipped — loaded {len(words)} words from cache")

    # ---------------------------------------------------------------
    # Stage 2: Scene Classification
    # ---------------------------------------------------------------
    classified_scenes = []
    if 2 not in skip_stages:
        print("\n[Engine] ── Stage 2: Scene Classification ──")
        scenes_path = job_dir / "scenes.json"
        # Force re-classify if no cache
        if scenes_path.exists():
            classified_scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
            print(f"[Engine] Stage 2 cached — {len(classified_scenes)} scenes loaded")
        else:
            classified_scenes = classify_video(
                video_path=source_video,
                job_dir=job_dir,
                words=words,
                scene_threshold=scene_threshold,
                target_time=target_frame,
                target_point=target_point,
                reference_image=reference_image,
            )
    else:
        scenes_path = job_dir / "scenes.json"
        if scenes_path.exists():
            classified_scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
            print(f"[Engine] Stage 2 skipped — {len(classified_scenes)} scenes loaded")
        else:
            raise PipelineError("Stage 2 skipped but scenes.json not found. Run without --skip 2.")

    aroll_count = sum(1 for s in classified_scenes if s.get("class") == "A_ROLL")
    broll_count = sum(1 for s in classified_scenes if s.get("class") == "B_ROLL")
    silent_count = sum(1 for s in classified_scenes if s.get("class") == "SILENT")
    print(f"\n[Engine] Scene breakdown: {aroll_count} A-Roll | {broll_count} B-Roll | {silent_count} Silent")

    # ---------------------------------------------------------------
    # Stage 3: ElevenLabs Dubbing
    # ---------------------------------------------------------------
    dubbed_audio = None
    dubbed_srt = None
    if 3 not in skip_stages:
        print("\n[Engine] ── Stage 3: ElevenLabs Dubbing ──")
        if dry_run:
            # In dry-run, check if dubbed audio already exists
            dubbed_audio = job_dir / f"dubbed_master_{target_lang}.mp3"
            dubbed_srt = job_dir / f"transcript_{target_lang}.srt"
            if dubbed_audio.exists():
                print(f"[Engine] DRY RUN — reusing existing dubbed audio: {dubbed_audio.name}")
            else:
                print(f"[Engine] DRY RUN — skipping ElevenLabs Dubbing API call")
                # Use source audio as placeholder
                dubbed_audio = job_dir / f"dubbed_master_{target_lang}_PLACEHOLDER.mp3"
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-nostdin", "-i", str(source_video),
                    "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                    str(dubbed_audio)
                ], capture_output=True)
        else:
            # Parse source cues for Mode B (Voice-Clone TTS for unsupported languages)
            from .srt_parser import parse_srt as _parse_srt
            source_cues = []
            if srt_path and Path(srt_path).exists():
                source_cues = _parse_srt(Path(srt_path).read_text(encoding="utf-8"))

            dubbed_audio, dubbed_srt, _ = run_dubbing(
                video_path=source_video,
                job_dir=job_dir,
                target_lang=target_lang,
                source_lang=source_lang,
                num_speakers=num_speakers,
                source_cues=source_cues,
                translated_srt=reviewed_translation_srt,
            )

    else:
        dubbed_audio = job_dir / f"dubbed_master_{target_lang}.mp3"
        dubbed_srt = job_dir / f"transcript_{target_lang}.srt"
        if not dubbed_audio.exists():
            raise PipelineError(f"Stage 3 skipped but dubbed audio not found: {dubbed_audio}")
        print(f"[Engine] Stage 3 skipped — using {dubbed_audio.name}")

    # ---------------------------------------------------------------
    # Stage 4: Targeted Lip-Sync
    # ---------------------------------------------------------------
    output_clips = []
    if 4 not in skip_stages:
        print("\n[Engine] ── Stage 4: Targeted Lip-Sync ──")
        output_clips = process_shots(
            video_path=source_video,
            dubbed_audio=dubbed_audio,
            classified_scenes=classified_scenes,
            job_dir=job_dir,
            sync_model=sync_model,
            dry_run=dry_run,
        )
    else:
        # Reconstruct clip list from existing segments
        seg_dir = job_dir / "segments"
        for i in range(len(classified_scenes)):
            prefix = f"seg_{i:03d}"
            label = classified_scenes[i].get("class", "B_ROLL")
            if label == "A_ROLL":
                candidates = sorted(seg_dir.glob(f"{prefix}_*_synced.mp4"))
                if not candidates:
                    raise PipelineError(f"Stage 4 skipped but synced segment missing: {prefix}")
                output_clips.append(candidates[0])
            elif label == "B_ROLL":
                candidates = sorted(seg_dir.glob(f"{prefix}_*broll.mp4"))
                if not candidates:
                    raise PipelineError(f"Stage 4 skipped but B-roll segment missing: {prefix}")
                output_clips.append(candidates[0])
            else:
                candidates = sorted(seg_dir.glob(f"{prefix}_*silent.mp4"))
                if not candidates:
                    raise PipelineError(f"Stage 4 skipped but silent segment missing: {prefix}")
                output_clips.append(candidates[0])
        print(f"[Engine] Stage 4 skipped — loaded {len(output_clips)} existing clips")

    # ---------------------------------------------------------------
    # Stage 5: Frame-Locked Stitch
    # ---------------------------------------------------------------
    print("\n[Engine] ── Stage 5: Frame-Locked Assembly ──")
    final_output = job_dir / f"final_dubbed_{target_lang}.mp4"
    stitch_segments(
        segments=output_clips,
        output_path=final_output,
        source_video=source_video,
        fps=fps,
        master_audio=dubbed_audio,
    )


    print(f"\n{'='*60}")
    print(f"[Engine] ✓ DONE — Final video: {final_output}")
    print(f"[Engine] A-Roll lip-synced: {aroll_count} shots")
    print(f"[Engine] B-Roll (audio + original video): {broll_count} shots")
    print(f"{'='*60}\n")
    return final_output
