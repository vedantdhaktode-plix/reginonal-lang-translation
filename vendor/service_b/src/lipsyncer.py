"""
Stage 4 — Targeted Lip-Sync
For each A-Roll shot:
  1. Extract video clip from source (exact frame bounds)
  2. Extract dubbed audio slice from master dubbed audio
  3. Submit to Sync Labs lipsync-2 (active_speaker_detection on)
  4. Cache by sha256(video_clip + audio_clip) — never re-calls if synced file exists

For each B-Roll shot:
  1. Extract video clip (untouched original visuals)
  2. Extract dubbed audio slice
  3. Mux audio onto original video clip

For SILENT shots:
  1. Copy source video clip untouched (no audio change)
"""
import hashlib
import subprocess
from pathlib import Path

from .common import PipelineError
from .scene_analyzer import get_video_info, extract_subclip
from .synclabs_client import SyncLabsClient
from .audio import ffmpeg


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))  # First 2MB sufficient
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _extract_audio_slice(
    audio_path: Path, start_s: float, duration_s: float, out_path: Path
) -> Path:
    """Cut an exact slice of audio to WAV for Sync Labs input."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-y", "-nostdin",
        "-ss", f"{start_s:.6f}", "-t", f"{duration_s:.6f}",
        "-i", str(audio_path),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio slice failed: {r.stderr}")
    return out_path


def _mux_audio_onto_video(
    video_path: Path, audio_path: Path, out_path: Path, duration_s: float
) -> Path:
    """Replace video's audio track with dubbed audio (for B-Roll shots)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-y", "-nostdin",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", f"atrim=duration={duration_s:.6f},asetpts=PTS-STARTPTS,apad,atrim=duration={duration_s:.6f}",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio mux failed: {r.stderr}")
    return out_path


def _copy_source_clip(
    video_path: Path, start_s: float, duration_s: float,
    fps: float, out_path: Path
) -> Path:
    """Copy source clip bit-for-bit (SILENT or untouched shots)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract_subclip(video_path, start_s, start_s + duration_s, out_path, target_fps=fps)
    return out_path


def process_shots(
    video_path: Path,
    dubbed_audio: Path,
    classified_scenes: list[dict],
    job_dir: Path,
    sync_model: str = "lipsync-2",
    dry_run: bool = False,
) -> list[Path]:
    """
    Stage 4 entry point.
    Processes all shots and returns ordered list of output video clip paths.

    A_ROLL → Sync Labs lip-sync (cached by content hash)
    B_ROLL → Original video + dubbed audio mux
    SILENT → Original video clip (untouched)
    """
    job_dir = Path(job_dir)
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    fps = info["fps"]
    source_duration = info["duration"]

    sync_client = None
    if not dry_run:
        sync_client = SyncLabsClient(default_model=sync_model)

    output_clips: list[Path] = []

    for i, scene in enumerate(classified_scenes):
        start_s = scene["start"]
        end_s = min(scene["end"], source_duration)
        duration_s = end_s - start_s
        label = scene.get("class", "B_ROLL")
        prefix = f"seg_{i:03d}"
        # Time-keyed suffix so files are strictly bound to their exact millisecond range
        time_key = f"{int(round(start_s * 1000))}_{int(round(end_s * 1000))}"

        print(f"\n[LipSync] Shot {i:02d}/{len(classified_scenes)-1} "
              f"[{start_s:.2f}s-{end_s:.2f}s] ({duration_s:.2f}s) → {label}")

        # --- Extract source video clip (strictly time-keyed) ---
        raw_clip = seg_dir / f"{prefix}_{time_key}_raw.mp4"
        if not raw_clip.exists():
            extract_subclip(video_path, start_s, end_s, raw_clip, target_fps=fps)

        # --- Extract dubbed audio slice (strictly time-keyed) ---
        audio_slice = seg_dir / f"{prefix}_{time_key}_dubbed.wav"
        if not audio_slice.exists():
            _extract_audio_slice(dubbed_audio, start_s, duration_s, audio_slice)

        # --------------------------------------------------------
        if label == "SILENT":
            out = seg_dir / f"{prefix}_{time_key}_silent.mp4"
            if not out.exists():
                _copy_source_clip(video_path, start_s, duration_s, fps, out)
            print(f"  [LipSync] SILENT → copied source clip ({duration_s:.2f}s)")
            output_clips.append(out)

        elif label == "B_ROLL":
            out = seg_dir / f"{prefix}_{time_key}_broll.mp4"
            if not out.exists():
                _mux_audio_onto_video(raw_clip, audio_slice, out, duration_s)
            print(f"  [LipSync] B-Roll → original video + dubbed audio muxed")
            output_clips.append(out)

        else:  # A_ROLL
            # Content-hash cache — reuse if already synced
            clip_hash = _file_hash(raw_clip) + _file_hash(audio_slice)
            synced_out = seg_dir / f"{prefix}_{time_key}_{clip_hash}_synced.mp4"

            # Check for existing synced clip:
            # 1. Exact hash match
            # 2. Or existing synced file for this exact shot prefix and time range
            synced_found = None
            if synced_out.exists():
                synced_found = synced_out
            else:
                candidates = sorted(seg_dir.glob(f"{prefix}_{time_key}_*_synced.mp4"))
                if candidates:
                    synced_found = candidates[0]

            if synced_found:
                print(f"  [LipSync] A-Roll → cached synced clip reused: {synced_found.name} (dur={duration_s:.2f}s)")
                output_clips.append(synced_found)
                continue

            if dry_run:
                # In dry-run: mux audio without lip-sync (for testing)
                dry_out = seg_dir / f"{prefix}_{time_key}_aroll_dryrun.mp4"
                if not dry_out.exists():
                    _mux_audio_onto_video(raw_clip, audio_slice, dry_out, duration_s)
                print(f"  [LipSync] A-Roll → DRY RUN, muxed without sync")
                output_clips.append(dry_out)
                continue

            # --- Guard: check actual clip duration ---
            try:
                actual_dur = get_video_info(raw_clip)["duration"]
            except Exception:
                actual_dur = duration_s

            # True sub-second microcuts (< 0.85s, e.g. 10-25 frames) produce visual popping and should not be warped
            if actual_dur < 0.85:
                print(f"  [LipSync] A-Roll → clip too short for neural sync ({actual_dur:.2f}s < 0.85s) "
                      f"— falling back to clean B-Roll mux")
                fallback_out = seg_dir / f"{prefix}_{time_key}_short_fallback.mp4"
                if not fallback_out.exists():
                    _mux_audio_onto_video(raw_clip, audio_slice, fallback_out, actual_dur)
                output_clips.append(fallback_out)
                continue

            # Clips between 0.85s and 1.02s (e.g. 0.97s from 30fps quantization of nominal 1s cuts):
            # Sync Labs API rejects inputs strictly under 1.0s. Auto-pad with repeated end frame so Sync Labs accepts it,
            # then trim back to the exact frame-locked duration upon download.
            padded_needed = (actual_dur < 1.02)
            send_clip = raw_clip
            send_audio = audio_slice

            if padded_needed:
                pad_target = 1.06
                padded_clip = seg_dir / f"{prefix}_{time_key}_padded.mp4"
                padded_audio = seg_dir / f"{prefix}_{time_key}_padded.wav"
                if not padded_clip.exists():
                    subprocess.run([
                        ffmpeg(), "-y", "-nostdin",
                        "-i", str(raw_clip),
                        "-vf", f"tpad=stop_mode=clone:stop_duration={pad_target - actual_dur:.4f}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        str(padded_clip)
                    ], capture_output=True)
                if not padded_audio.exists():
                    subprocess.run([
                        ffmpeg(), "-y", "-nostdin",
                        "-i", str(audio_slice),
                        "-af", f"apad=whole_dur={pad_target:.4f}",
                        "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
                        str(padded_audio)
                    ], capture_output=True)
                if padded_clip.exists() and padded_audio.exists():
                    send_clip = padded_clip
                    send_audio = padded_audio

            # --- Submit to Sync Labs ---
            print(f"  [LipSync] A-Roll → uploading to Sync Labs {sync_model} "
                  f"({actual_dur:.2f}s, {raw_clip.stat().st_size // 1024}KB)...")
            try:
                video_url = sync_client.upload_file(send_clip, content_type="video/mp4")
                audio_url = sync_client.upload_file(send_audio, content_type="audio/wav")

                options = {
                    "sync_mode": "cut_off",
                    "active_speaker_detection": {"auto_detect": True},
                }

                gen_data = sync_client.generate(video_url, audio_url,
                                                model=sync_model, options=options)
                gen_id = gen_data["id"]
                print(f"  [LipSync] Sync Labs Job: {gen_id} — waiting...")
                result = sync_client.wait_for_completion(gen_id, poll_interval=8)
                sync_client.download_result(result["outputUrl"], synced_out)

                # If clip was padded, trim back to exact original frame duration
                if padded_needed and synced_out.exists():
                    trimmed_tmp = seg_dir / f"{prefix}_{time_key}_trimmed.mp4"
                    subprocess.run([
                        ffmpeg(), "-y", "-nostdin",
                        "-ss", "0", "-t", f"{actual_dur:.6f}",
                        "-i", str(synced_out),
                        "-c:v", "libx264", "-c:a", "copy",
                        str(trimmed_tmp)
                    ], capture_output=True)
                    if trimmed_tmp.exists():
                        trimmed_tmp.replace(synced_out)

                print(f"  [LipSync] Downloaded → {synced_out.name}")
                output_clips.append(synced_out)

            except PipelineError as e:
                # Graceful fallback: if Sync Labs rejects (422, 429, etc.), mux audio only
                print(f"  [LipSync] ⚠ Sync Labs failed for shot {i}: {e}")
                print(f"  [LipSync] Falling back to B-Roll mux (original video + dubbed audio)")
                fallback_out = seg_dir / f"{prefix}_{time_key}_syncfail_fallback.mp4"
                if not fallback_out.exists():
                    _mux_audio_onto_video(raw_clip, audio_slice, fallback_out, actual_dur)
                output_clips.append(fallback_out)

    return output_clips
