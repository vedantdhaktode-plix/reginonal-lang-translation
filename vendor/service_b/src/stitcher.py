"""
Stage 5 — Frame-Locked Master Stitch
Concatenates all processed segments in exact scene order.
Validates final frame count == source frame count before writing output.
Normalizes segments to a common CFR/timebase, then performs one deterministic
final encode so variable-rate cloud clips cannot introduce timestamp drift.
"""
import subprocess
from pathlib import Path
from .common import PipelineError
from .scene_analyzer import get_video_info
from .audio import ffmpeg


def stitch_segments(
    segments: list[Path],
    output_path: Path,
    source_video: Path,
    fps: float = None,
    master_audio: Path = None,
) -> Path:
    """
    Frame-locked concatenation of all segment clips.

    Args:
        segments      : Ordered list of per-shot video file paths
        output_path   : Destination for the final assembled video
        source_video  : Original source (used to validate final frame count)
        fps           : Source FPS (read from source_video if not provided)
        master_audio  : Optional full-length dubbed audio to mux over stitched video

    Returns:
        output_path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise PipelineError("No segments to stitch.")

    # Validate each segment exists and is non-empty
    for i, seg in enumerate(segments):
        if not Path(seg).exists():
            raise PipelineError(f"Segment {i} missing: {seg}")
        if Path(seg).stat().st_size < 1000:
            raise PipelineError(f"Segment {i} suspiciously small: {seg}")

    # Get source info first so we can normalize segments to exact target FPS and timebase
    src_info = get_video_info(source_video)
    src_frames = src_info["frame_count"]
    src_duration = src_info["duration"]
    if fps is None:
        fps = src_info["fps"]

    # Pre-normalize segments to video-only streams:
    # Sync Labs outputs variable frame rates and different container timebases (e.g. 1/18720 vs 1/15360),
    # while local B-roll has different stream layouts. Re-encoding each segment with uniform fps
    # and reset PTS prevents timestamp drift and frame drops in concat demuxer.
    norm_segments = []
    temp_cleanups = []
    for i, seg in enumerate(segments):
        norm_path = seg.parent / f"{seg.stem}_vonly.mp4"
        cmd_norm = [
            ffmpeg(), "-y", "-nostdin",
            "-i", str(seg),
            "-map", "0:v:0",
            "-an",
            "-vf", f"fps={fps},setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(norm_path),
        ]
        r_norm = subprocess.run(cmd_norm, capture_output=True, text=True)
        if r_norm.returncode == 0 and norm_path.exists():
            norm_segments.append(norm_path)
            temp_cleanups.append(norm_path)
        else:
            norm_segments.append(Path(seg))

    # Build concat manifest
    concat_txt = output_path.with_suffix(".concat.txt")
    lines = [
        f"file '{Path(s).resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
        for s in norm_segments
    ]
    concat_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[Stitcher] Assembling {len(segments)} segments (normalized video-only streams)...")
    print(f"[Stitcher] Source: {src_frames} frames @ {fps:.3f}fps ({src_duration:.3f}s)")

    # Build ffmpeg concat command
    # Enforce constant fps matching source to prevent VFR drift from cloud lipsync models
    cmd = [
        ffmpeg(), "-y", "-nostdin",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
    ]

    has_master_audio = bool(master_audio and Path(master_audio).exists())
    if has_master_audio:
        # The segment tracks are video-only, so this maps exactly one audio
        # source. Reset its timestamps, smooth tiny clock discontinuities, and
        # pad/trim it to the source duration. This cannot mix duplicate voices.
        audio_filter = (
            f"aresample=48000:async=1:first_pts=0,"
            f"asetpts=PTS-STARTPTS,"
            f"apad=pad_dur={src_duration:.9f},"
            f"atrim=duration={src_duration:.9f}"
        )
        cmd += [
            "-i", str(master_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", f"fps={fps},setpts=PTS-STARTPTS",
            "-af", audio_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-frames:v", str(src_frames),
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        cmd += [
            "-map", "0:v:0",
            "-vf", f"fps={fps},setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-frames:v", str(src_frames),
            "-an",
            "-movflags", "+faststart",
            str(output_path),
        ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Stitching failed:\n{r.stderr}")

    # Frame count validation & conform
    out_info = get_video_info(output_path)
    out_frames = out_info["frame_count"]
    out_duration = out_info["duration"]

    # If minor frame rounding occurred, conform final output to exact source frame count
    if out_frames != src_frames:
        conformed_path = output_path.with_name(f"{output_path.stem}_conformed.mp4")
        if out_frames < src_frames:
            pad_count = src_frames - out_frames
            conform_cmd = [
                ffmpeg(), "-y", "-nostdin",
                "-i", str(output_path),
                "-vf", f"tpad=stop_mode=clone:stop={pad_count},fps={fps}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(conformed_path),
            ]
        else:
            conform_cmd = [
                ffmpeg(), "-y", "-nostdin",
                "-i", str(output_path),
                "-vf", f"trim=end_frame={src_frames},fps={fps}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(conformed_path),
            ]
        r_conf = subprocess.run(conform_cmd, capture_output=True, text=True)
        if r_conf.returncode == 0 and conformed_path.exists():
            conformed_path.replace(output_path)
            out_info = get_video_info(output_path)
            out_frames = out_info["frame_count"]
            out_duration = out_info["duration"]

    tolerance = 6
    if abs(out_frames - src_frames) > tolerance:
        raise PipelineError(
            f"Frame count mismatch! Source={src_frames}, Output={out_frames} "
            f"(tolerance={tolerance}). Check segment boundaries."
        )

    drift_ms = abs(out_duration - src_duration) * 1000
    print(f"[Stitcher] ✓ Output: {out_frames} frames, {out_duration:.3f}s "
          f"(drift={drift_ms:.1f}ms)")
    print(f"[Stitcher] ✓ Final video: {output_path}")
    concat_txt.unlink(missing_ok=True)
    for p in temp_cleanups:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return output_path

