import json
import hashlib
import re
import os
from pathlib import Path
import shutil
import subprocess
import time
from .audio import ffmpeg
from .common import PipelineError, read_json, write_json
from .dubbing_client import ElevenLabsDubbingClient
from .face_tracker import (
    auto_detect_creator_anchor,
    classify_shot_voiceover_aware,
    extract_reference_signature_from_image,
)
from .scene_analyzer import (
    concat_segments,
    detect_scenes,
    extract_subclip,
    get_video_info,
    mux_audio_video,
)
from .srt_parser import parse_srt, stamp
from .synclabs_client import SyncLabsClient
from .dub_assembly import route_model, assemble_segment, join_segments
from .face_id_tracker import FaceIDTracker
import cv2


def find_repeated_phrase(cues):
    """Find an accidentally tripled phrase in a provider transcript."""
    for cue in cues:
        words = re.findall(r"\b[\w']+\b", cue.text.lower())
        for size in range(3, min(9, len(words) // 3 + 1)):
            for offset in range(len(words) - 3 * size + 1):
                phrase = words[offset:offset + size]
                if phrase == words[offset + size:offset + 2 * size] == words[offset + 2 * size:offset + 3 * size]:
                    return " ".join(phrase), cue.start_ms / 1000
    return None


def audio_only_path(media_path):
    """ElevenLabs may return an MP4 from its audio endpoint; extract its audio."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=index", "-of", "csv=p=0", str(media_path)],
                           capture_output=True, text=True)
    if probe.returncode or not probe.stdout.strip():
        return Path(media_path)
    result_path = Path(media_path).with_name(Path(media_path).stem + "_audio_only.mp3")
    if not result_path.exists():
        result = subprocess.run([ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(media_path),
                                 "-map", "0:a:0", "-vn", "-c:a", "libmp3lame", "-q:a", "2",
                                 str(result_path)], capture_output=True, text=True)
        if result.returncode:
            raise PipelineError(f"Could not extract dubbed audio: {result.stderr}")
    return result_path


def extract_source_audio(video_path, output_audio_path):
    """
    Extracts audio track from video to 48kHz WAV.
    """
    output_audio_path = Path(output_audio_path).resolve()
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        str(output_audio_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Failed to extract source audio: {res.stderr}")
    return output_audio_path


def extract_audio_segment(audio_path, start_sec, end_sec, output_path):
    """
    Slices an exact segment from an audio file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(audio_path),
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        str(output_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Failed to extract audio segment [{start_sec:.2f}-{end_sec:.2f}]: {res.stderr}")
    return output_path


def standardize_video_stream(src_path, dst_path, expected_frames=None):
    """
    Re-encodes a video-only clip with PTS reset to zero.

    Using ``-c:v copy`` with the concat demuxer preserves the container's
    start_time offset, which accumulates into visible A/V drift.  This function
    transcodes instead so that every segment fed to the concat demuxer has
    start_time=0 and consecutive PTS values beginning at 0.

    Parameters
    ----------
    src_path : path-like
        Input video file (may have audio; audio is dropped).
    dst_path : path-like
        Output video-only file with PTS starting at 0.
    expected_frames : int or None
        When provided, a warning is printed if the output contains a different
        number of frames.  The segment is still kept; callers decide what to do.
    """
    dst_path = Path(dst_path).resolve()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(src_path),
        "-map", "0:v:0",
        # Re-encode so that PTS is recalculated from scratch.
        # setpts=PTS-STARTPTS shifts the first frame to pts=0 and makes every
        # subsequent frame contiguous, eliminating the start_time gap.
        "-vf", "setpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(
            f"Failed to standardize video segment {Path(src_path).name}: {res.stderr}"
        )

    if expected_frames is not None:
        # Quick probe to verify frame count.
        import json as _json
        from pathlib import Path as _Path
        _ffmpeg_path = _Path(ffmpeg())
        # ffprobe lives in the same bin directory as ffmpeg.
        _ffprobe_name = _ffmpeg_path.name.lower().replace("ffmpeg", "ffprobe")
        _ffprobe = str(_ffmpeg_path.parent / _ffprobe_name)
        probe_cmd = [
            _ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_streams", str(dst_path),
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe_res.returncode == 0:
            try:
                info = _json.loads(probe_res.stdout)
                vs = next(
                    (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
                    {},
                )
                actual = int(vs.get("nb_frames", expected_frames))
                if actual != expected_frames:
                    print(
                        f"  [Warning] {Path(src_path).name}: expected {expected_frames} "
                        f"frames but got {actual} after standardize."
                    )
            except Exception:
                pass

    return dst_path


def select_speech_intervals(cues, scenes, total_duration_sec, merge_gap_sec=0.5, min_interval_sec=0.2):
    """
    Derives candidate speech intervals from transcript cues, merges nearby utterances (< merge_gap_sec),
    splits intervals at scene cuts, and constructs full timeline intervals (speech vs passthrough).
    """
    if not cues:
        # No cues: entire video is passthrough
        return [{
            "index": 0,
            "start": 0.0,
            "end": round(total_duration_sec, 3),
            "duration": round(total_duration_sec, 3),
            "is_speech": False,
            "action": "passthrough"
        }]

    # Step 1: Convert cues to raw intervals in seconds
    raw_spans = []
    for c in cues:
        s = getattr(c, "start_ms", None)
        e = getattr(c, "end_ms", None)
        if s is None:
            s = c.get("start_ms", 0)
            e = c.get("end_ms", 0)
        s_sec = max(0.0, s / 1000.0)
        e_sec = min(total_duration_sec, e / 1000.0)
        if e_sec > s_sec:
            raw_spans.append((s_sec, e_sec))

    raw_spans.sort(key=lambda x: x[0])

    # Step 2: Merge cues with small gaps (< merge_gap_sec)
    merged_spans = []
    for span in raw_spans:
        if not merged_spans:
            merged_spans.append(list(span))
        else:
            prev = merged_spans[-1]
            if span[0] - prev[1] <= merge_gap_sec:
                prev[1] = max(prev[1], span[1])
            else:
                merged_spans.append(list(span))

    # Step 3: Collect scene cut points
    cut_points = set()
    for s in scenes:
        start_cut = round(s["start"], 3)
        end_cut = round(s["end"], 3)
        if 0 < start_cut < total_duration_sec:
            cut_points.add(start_cut)
        if 0 < end_cut < total_duration_sec:
            cut_points.add(end_cut)
    sorted_cuts = sorted(cut_points)

    # Step 4: Split speech spans at scene boundaries
    split_speech_spans = []
    for span in merged_spans:
        curr_start = span[0]
        curr_end = span[1]

        # Find any cuts that fall strictly within this interval
        intersecting_cuts = [
            c for c in sorted_cuts
            if (curr_start + 0.08) < c < (curr_end - 0.08)
        ]

        if not intersecting_cuts:
            if curr_end - curr_start >= min_interval_sec:
                split_speech_spans.append((round(curr_start, 3), round(curr_end, 3)))
        else:
            points = [curr_start] + intersecting_cuts + [curr_end]
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                if p2 - p1 >= min_interval_sec:
                    split_speech_spans.append((round(p1, 3), round(p2, 3)))

    split_speech_spans.sort(key=lambda x: x[0])

    # Step 5: Fill timeline gaps with passthrough intervals
    timeline = []
    cursor = 0.0
    for span in split_speech_spans:
        s, e = span
        if s > cursor + 0.05:
            # Passthrough gap before speech
            timeline.append({
                "start": round(cursor, 3),
                "end": round(s, 3),
                "duration": round(s - cursor, 3),
                "is_speech": False,
                "action": "passthrough"
            })
        timeline.append({
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "is_speech": True,
            "action": "sync"
        })
        cursor = e

    if cursor < total_duration_sec - 0.05:
        timeline.append({
            "start": round(cursor, 3),
            "end": round(total_duration_sec, 3),
            "duration": round(total_duration_sec - cursor, 3),
            "is_speech": False,
            "action": "passthrough"
        })

    # Assign sequential indices
    for idx, item in enumerate(timeline):
        item["index"] = idx

    return timeline


def run_video_dubbing_pipeline(
    source_video_path,
    target_lang,
    job_dir=None,
    source_lang="auto",
    num_speakers=1,
    auto_approve=False,
    dry_run=False,
    dubbing_client=None,
    synclabs_client=None,
    sync_model=None,
    broll_model="lipsync-2",
    broll_audio=None,
    target_frame=None,
    target_point=None,
    reviewed_audio=None,
    reviewed_transcript=None,
    scene_threshold=32.0,
    poll_interval=5,
    timeout=1200,
    broll_filter=True,
    reference_image=None,
    motion_threshold=5.0,
    match_threshold=0.45
):
    """
    Executes the end-to-end video dubbing pipeline:
    1. Inspects video & extracts source audio.
    2. Submits to ElevenLabs Dubbing API (or resumes existing job).
    3. Retrieves dubbed master audio & transcript cues.
    4. Detects scenes & selects speech intervals (splitting at camera cuts).
    5. Applies voiceover-aware B-roll filtering (leaves product shots & demonstrations untouched).
    6. Requests CLI confirmation (unless auto_approve is True).
    7. Submits verified talking-head intervals to Sync Labs (default: lipsync-2, resumable per-interval).
    8. Assembles frame-bounded segments retaining returned Sync audio.
    9. Validates output and records final manifest.
    """
    source_video = Path(source_video_path).resolve()
    if not source_video.exists():
        raise PipelineError(f"Source video not found: {source_video}")
    if target_point and target_frame is None and not reference_image:
        raise PipelineError("--target-point requires --target-frame or --reference.")

    if not job_dir:
        job_dir = Path("output") / f"dub_{source_video.stem}_{target_lang}"
    job_dir = Path(job_dir).resolve()
    if dry_run:
        job_dir = job_dir / "dry_run"
    job_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = job_dir / "dub_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    # Initial manifest state
    manifest.setdefault("source_video", str(source_video))
    manifest.setdefault("target_lang", target_lang)
    manifest.setdefault("source_lang", source_lang)
    manifest.setdefault("created_at", time.time())
    manifest.setdefault("status", "initialized")

    if bool(reviewed_audio) != bool(reviewed_transcript):
        raise PipelineError("Provide both --reviewed-audio and --reviewed-transcript so words match the audio.")
    dub_suffix = "_speakers_auto" if num_speakers == 0 else f"_speakers_{num_speakers}"
    dubbed_audio_path = Path(reviewed_audio).resolve() if reviewed_audio else job_dir / f"master_dubbed_audio_{target_lang}{dub_suffix}.mp3"
    transcript_path = Path(reviewed_transcript).resolve() if reviewed_transcript else job_dir / f"transcript_{target_lang}{dub_suffix}.srt"
    if reviewed_audio and (not dubbed_audio_path.is_file() or not transcript_path.is_file()):
        raise PipelineError("Reviewed audio or transcript file is missing.")
    dubbing_request = {"source": str(source_video), "source_size": source_video.stat().st_size,
                       "source_mtime_ns": source_video.stat().st_mtime_ns, "target_lang": target_lang,
                       "source_lang": source_lang, "num_speakers": num_speakers}
    if not reviewed_audio and manifest.get("dubbing_request") != dubbing_request:
        manifest.pop("elevenlabs_dubbing_id", None)
        manifest.pop("timeline", None)
        manifest.pop("broll_filtered", None)
        manifest["dubbing_request"] = dubbing_request

    # Historical mixed manifests are retained; never delete paid assets on resume.
    if broll_audio is not None:
        broll_audio = Path(broll_audio).resolve()
        if not broll_audio.is_file():
            raise PipelineError(f"B-roll audio source not found: {broll_audio}")

    manifest["dry_run"] = dry_run
    write_json(manifest_path, manifest)

    # Step 1: Video inspection
    video_info = get_video_info(source_video)
    total_duration = video_info["duration"]
    target_fps = video_info["fps"]
    if reviewed_audio:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(dubbed_audio_path)],
                               capture_output=True, text=True)
        try:
            audio_duration = float(probe.stdout.strip())
        except ValueError:
            raise PipelineError("Cannot read reviewed audio duration.") from None
        if audio_duration + 1 / target_fps < total_duration:
            raise PipelineError(f"Reviewed audio ends {total_duration - audio_duration:.2f}s before the video.")
    identity_key = None
    identity_tracker = None
    if reference_image or target_frame is not None:
        identity_tracker = FaceIDTracker()
        if reference_image:
            reference_image = Path(reference_image).resolve()
            anchor = cv2.imread(str(reference_image))
            if anchor is None:
                raise PipelineError(f"Cannot read reference image: {reference_image}")
            identity_bytes = reference_image.read_bytes() + repr(target_point).encode()
        else:
            cap = cv2.VideoCapture(str(source_video))
            cap.set(cv2.CAP_PROP_POS_MSEC, float(target_frame) * 1000)
            ok, anchor = cap.read()
            cap.release()
            if not ok:
                raise PipelineError(f"Cannot read target frame at {target_frame}s")
            identity_bytes = f"{target_frame}:{target_point}".encode()
        try:
            if target_point:
                identity_tracker.set_anchor_from_point(anchor, *target_point)
            else:
                height, width = anchor.shape[:2]
                identity_tracker.detector.setInputSize((width, height))
                _, faces = identity_tracker.detector.detect(anchor)
                if faces is not None and len(faces) > 1:
                    raise PipelineError("Reference frame contains multiple faces; provide --target-point X Y.")
                identity_tracker.set_anchor_from_frame(anchor)
        except ValueError as exc:
            raise PipelineError(f"Target speaker selection failed: {exc}") from exc
        identity_key = hashlib.sha256(identity_bytes).hexdigest()
    print(f"[Pipeline] Source video: {source_video.name} ({video_info['width']}x{video_info['height']} @ {target_fps}fps, {total_duration:.2f}s)")
    if reviewed_audio:
        manifest["dubbed_audio_path"] = str(dubbed_audio_path)
        manifest["transcript_path"] = str(transcript_path)
        manifest["dubbing_source"] = "reviewed"
        write_json(manifest_path, manifest)
    elif dry_run:
        print("[Pipeline] *** DRY-RUN MODE ACTIVATED: Local simulation without paid API calls ***")

    # Step 2: ElevenLabs Dubbing
    if reviewed_audio:
        pass
    elif dry_run:
        elevenlabs_dubbing_id = "dry_run_dub_id"
        manifest["elevenlabs_dubbing_id"] = elevenlabs_dubbing_id
        if not dubbed_audio_path.exists():
            extract_source_audio(source_video, dubbed_audio_path.with_suffix(".wav"))
            dubbed_audio_path = dubbed_audio_path.with_suffix(".wav")
        if not transcript_path.exists():
            cue_start_ms = int(min(total_duration * 0.1, 0.5) * 1000)
            cue_end_ms = int(min(total_duration * 0.8, 4.0) * 1000)
            if cue_end_ms <= cue_start_ms:
                cue_end_ms = int(total_duration * 1000)
            transcript_path.write_text(
                f"1\n{stamp(cue_start_ms)} --> {stamp(cue_end_ms)}\nSimulated speech for dry run.\n",
                encoding="utf-8"
            )
        manifest["status"] = "elevenlabs_completed"
        manifest["dubbed_audio_path"] = str(dubbed_audio_path)
        manifest["transcript_path"] = str(transcript_path)
        write_json(manifest_path, manifest)
    elif not (dubbed_audio_path.exists() and transcript_path.exists() and dubbed_audio_path.stat().st_size > 0):
        dub_client = dubbing_client or ElevenLabsDubbingClient()
        elevenlabs_dubbing_id = manifest.get("elevenlabs_dubbing_id")

        # Validate existing dubbing ID if present
        if elevenlabs_dubbing_id:
            if "dry_run" in str(elevenlabs_dubbing_id):
                elevenlabs_dubbing_id = None
            else:
                chk = dub_client.get_status(elevenlabs_dubbing_id)
                if chk.get("status") == "not_found":
                    print(f"[Pipeline] Stored dubbing ID '{elevenlabs_dubbing_id}' was not found on ElevenLabs. Creating new job...")
                    elevenlabs_dubbing_id = None

        if not elevenlabs_dubbing_id:
            print(f"[Pipeline] Submitting source video to ElevenLabs Dubbing ({source_lang} -> {target_lang})...")
            elevenlabs_dubbing_id = dub_client.create_dub_job(
                source_video,
                target_lang=target_lang,
                source_lang=source_lang,
                num_speakers=num_speakers
            )
            manifest["elevenlabs_dubbing_id"] = elevenlabs_dubbing_id
            manifest["status"] = "elevenlabs_submitted"
            write_json(manifest_path, manifest)
            print(f"[Pipeline] ElevenLabs Dubbing ID: {elevenlabs_dubbing_id}")
        else:
            print(f"[Pipeline] Resuming existing ElevenLabs Dubbing ID: {elevenlabs_dubbing_id}")

        if not (dubbed_audio_path.exists() and transcript_path.exists() and dubbed_audio_path.stat().st_size > 0):
            print("[Pipeline] Waiting for ElevenLabs Dubbing completion...")
            dub_client.wait_for_completion(elevenlabs_dubbing_id, poll_interval=poll_interval, timeout=timeout)
            print("[Pipeline] Dubbing complete. Downloading dubbed master audio & transcript...")
            dub_client.download_dubbed_audio(elevenlabs_dubbing_id, target_lang, dubbed_audio_path)
            dub_client.download_transcript(elevenlabs_dubbing_id, target_lang, transcript_path, format_type="srt")
            manifest["status"] = "elevenlabs_completed"
            manifest["dubbed_audio_path"] = str(dubbed_audio_path)
            manifest["transcript_path"] = str(transcript_path)
            write_json(manifest_path, manifest)

    if not reviewed_audio and not dry_run:
        dubbed_audio_path = audio_only_path(dubbed_audio_path)
        manifest["dubbed_audio_path"] = str(dubbed_audio_path)
        write_json(manifest_path, manifest)

    # Parse cues
    cues = parse_srt(transcript_path.read_text(encoding="utf-8"))
    if not cues:
        raise PipelineError("Dubbed transcript has no speech cues; review it before lip-sync.")
    if not reviewed_transcript:
        repetition = find_repeated_phrase(cues)
        if repetition:
            phrase, at_sec = repetition
            raise PipelineError(
                f"Dubbed script repeats '{phrase}' three times near {at_sec:.2f}s. "
                "Review the audio and supply matching --reviewed-audio and --reviewed-transcript."
            )
    print(f"[Pipeline] Loaded {len(cues)} transcript cues from ElevenLabs.")

    # Step 3: Scene detection & Speech interval selection
    scenes = manifest.get("scenes")
    if not scenes:
        print("[Pipeline] Detecting camera cuts and scene transitions...")
        scenes = detect_scenes(source_video, threshold=scene_threshold)
        manifest["scenes"] = scenes
        write_json(manifest_path, manifest)
    print(f"[Pipeline] Video divided into {len(scenes)} camera scenes.")

    timeline = manifest.get("timeline")
    if not timeline:
        timeline = select_speech_intervals(cues, scenes, total_duration)

    needs_broll_filtering = (broll_filter or identity_tracker is not None) and (
        not manifest.get("broll_filtered") or any("classification" not in it for it in timeline)
        or manifest.get("identity_key") != identity_key
    )

    if needs_broll_filtering:
        print("[Pipeline] Analyzing shots for voiceover B-roll passthrough vs talking head lip-sync...")
        ref_sig = None
        if identity_tracker:
            print("[Pipeline] Using face embeddings to identify the selected woman.")
        elif reference_image:
            ref_sig = extract_reference_signature_from_image(reference_image)
            print(f"[Pipeline] Using custom speaker reference image: {reference_image}")
        else:
            ref_sig = auto_detect_creator_anchor(source_video, scenes)
            if ref_sig is not None:
                print("[Pipeline] Speaker facial identity anchor extracted for character locking.")
            else:
                print("[Pipeline] No distinct facial anchor found; using mouth motion and skin texture analysis.")

        for item in timeline:
            # Older manifests changed is_speech to False for rejected faces.
            # Preserve the original transcript interval when identity changes.
            speech_candidate = item.get("speech_candidate", item.get("is_speech", False) or item.get("classification") is not None)
            item["speech_candidate"] = speech_candidate
            if speech_candidate:
                if identity_tracker:
                    match = identity_tracker.match_interval(source_video, item["start"], item["end"])
                    item["classification"] = "a_roll" if match["is_target"] else "b_roll"
                    item["is_speech"] = match["is_target"]
                    item["action"] = "sync" if match["is_target"] else "passthrough"
                    item["target_ratio"] = match["target_ratio"]
                    item["target_face"] = match["best"] if match["is_target"] else None
                    item["reason"] = "Selected speaker verified" if match["is_target"] else "Selected speaker absent or uncertain"
                    continue
                c_res = classify_shot_voiceover_aware(
                    source_video,
                    item["start"],
                    item["end"],
                    reference_sig=ref_sig,
                    motion_threshold=motion_threshold,
                    match_threshold=match_threshold
                )
                item["face_ratio"] = c_res.get("face_ratio", 0.0)
                item["classification"] = c_res["type"]
                item["reason"] = c_res["reason"]
                item["match"] = c_res.get("match", 0.0)
                item["motion"] = c_res.get("motion", 0.0)
                if c_res["type"] == "b_roll":
                    item["is_speech"] = False
                    item["action"] = "passthrough"
                else:
                    item["is_speech"] = True
                    item["action"] = "sync"

        manifest["broll_filtered"] = True
        manifest["identity_key"] = identity_key
        manifest["timeline"] = timeline
        write_json(manifest_path, manifest)
    elif not manifest.get("timeline"):
        manifest["timeline"] = timeline
        write_json(manifest_path, manifest)

    effective_sync_model = sync_model or os.environ.get("SYNC_MODEL", "lipsync-2")
    for item in timeline:
        item["selected_model"] = route_model(item, effective_sync_model, broll_model)
    speech_segments = [s for s in timeline if s["selected_model"]]
    passthrough_segments = [s for s in timeline if not s["selected_model"]]
    total_speech_sec = sum(s["duration"] for s in speech_segments)
    total_pass_sec = sum(s["duration"] for s in passthrough_segments)
    savings_pct = (1.0 - (total_speech_sec / total_duration)) * 100.0 if total_duration > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"PROPOSED SPEECH INTERVALS FOR LIP-SYNC ({len(speech_segments)} segments):")
    for s in speech_segments:
        extra = f" - Talking Head (Motion: {s.get('motion', 0):.1f})" if "motion" in s else ""
        print(f"  Segment #{s['index']:02d}: [{s['start']:6.2f}s - {s['end']:6.2f}s] ({s['duration']:5.2f}s){extra}")
    print("-" * 65)
    print(f"VOICEOVER B-ROLL PASSTHROUGH PRESERVED ({len(passthrough_segments)} segments):")
    for s in passthrough_segments:
        reason = s.get("reason", "Silence / non-speech pause")
        print(f"  Segment #{s['index']:02d}: [{s['start']:6.2f}s - {s['end']:6.2f}s] ({s['duration']:5.2f}s) - {reason}")
    print("-" * 65)
    print(f"Total Video: {total_duration:6.2f}s | Lip-sync: {total_speech_sec:6.2f}s | Passthrough: {total_pass_sec:6.2f}s")
    print(f"B-roll / Silent Footage Preserved: {savings_pct:.1f}%")
    print("=" * 65 + "\n")

    # Step 4: CLI Approval before Paid Sync Labs Requests
    effective_sync_model = sync_model or os.environ.get("SYNC_MODEL", "lipsync-2")
    if not auto_approve and not dry_run:
        confirm = input(f"Proceed with paid Sync generation for verified presenter shots using {effective_sync_model}? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("[Pipeline] Operation cancelled by user. Manifest saved; run with --yes to resume.")
            manifest["status"] = "pending_user_approval"
            write_json(manifest_path, manifest)
            return None, manifest

    sync_client = None
    if not dry_run:
        sync_client = synclabs_client or SyncLabsClient(default_model=effective_sync_model)
    final_segments = []

    print("[Pipeline] Processing video segments...")
    manifest["routing"] = {"presenter": effective_sync_model, "b_roll": broll_model,
                           "broll_audio": str(broll_audio) if broll_audio else None}
    source_hash = hashlib.sha256(source_video.read_bytes()).hexdigest()
    audio_hashes = {}
    cursor = 0
    for item in timeline:
        idx = item["index"]
        start_frame = cursor
        end_frame = round(item["end"] * target_fps)
        if item is timeline[-1]:
            end_frame = video_info["frame_count"]
        cursor = end_frame
        frames = end_frame - start_frame
        if frames <= 0:
            continue
        start_s, end_s = start_frame / target_fps, end_frame / target_fps
        selected_model = route_model(item, effective_sync_model, broll_model)
        audio_source = broll_audio if broll_audio and item.get("classification") == "b_roll" else dubbed_audio_path
        audio_source = Path(audio_source)
        if str(audio_source) not in audio_hashes:
            audio_hashes[str(audio_source)] = hashlib.sha256(audio_source.read_bytes()).hexdigest()
        key = hashlib.sha256(json.dumps([source_hash, audio_hashes[str(audio_source)], start_frame,
                                        end_frame, selected_model, identity_key, item.get("target_face"),
                                        dry_run, "paired-v2"]).encode()).hexdigest()[:16]
        raw_clip = seg_dir / f"seg_{idx:03d}_{key}_raw.mp4"
        if not raw_clip.exists():
            extract_subclip(source_video, start_s, end_s, raw_clip, target_fps=target_fps)
        video_clip, paired_audio, audio_start = raw_clip, audio_source, start_s
        if selected_model and not dry_run:
            synced_clip = seg_dir / f"seg_{idx:03d}_{key}_synced.mp4"
            if not synced_clip.exists():
                audio_clip = seg_dir / f"seg_{idx:03d}_{key}_audio.wav"
                extract_audio_segment(audio_source, start_s, end_s, audio_clip)
                gen_id = item.get("synclabs_generation_id") if item.get("request_key") == key else None
                if not gen_id:
                    video_url = sync_client.upload_file(raw_clip, content_type="video/mp4")
                    audio_url = sync_client.upload_file(audio_clip, content_type="audio/wav")
                    try:
                        options = {"sync_mode": "cut_off"}
                        if identity_tracker:
                            face = item.get("target_face")
                            if not face:
                                raise PipelineError(f"No selected face for segment {idx}")
                            options["active_speaker_detection"] = {
                                "auto_detect": False, "frame_number": face["frame_number"],
                                "coordinates": face["coordinates"]}
                        gen_id = sync_client.generate(video_url=video_url, audio_url=audio_url,
                                                      model=selected_model, options=options)["id"]
                        item.update(request_key=key, synclabs_generation_id=gen_id, sync_model=selected_model)
                        write_json(manifest_path, manifest)
                    except PipelineError as e:
                        if "422" in str(e):
                            print(f"  [Segment #{idx:02d}] WARNING: Sync Labs rejected segment with HTTP 422 (no face detected). Falling back to untouched B-roll cut.")
                            video_clip = raw_clip
                            paired_audio = audio_source
                            audio_start = start_s
                            item["action"] = "passthrough"
                            item["is_speech"] = False
                            gen_id = None
                        else:
                            raise
                if gen_id:
                    result = sync_client.wait_for_completion(gen_id, poll_interval=poll_interval, timeout=timeout)
                    pending = synced_clip.with_suffix(".download.mp4")
                    sync_client.download_result(result["outputUrl"], pending)
                    pending.replace(synced_clip)
                    video_clip, paired_audio, audio_start = synced_clip, synced_clip, 0
                    item["synced_clip"] = str(synced_clip)
        item.update(start_frame=start_frame, end_frame=end_frame, selected_model=selected_model)
        standardized = seg_dir / f"seg_{idx:03d}_{key}_paired.mov"
        assemble_segment(video_clip, paired_audio, standardized, frames, target_fps, audio_start)
        final_segments.append(standardized)
        write_json(manifest_path, manifest)

    final_output = job_dir / f"final_dual_model_{source_video.stem}_{target_lang}.mp4"
    join_segments(final_segments, final_output, video_info["frame_count"], target_fps)

    # Step 7: Validation — duration and per-segment boundary check
    import json as _json
    _ffmpeg_path = Path(ffmpeg())
    _ffprobe = str(_ffmpeg_path.parent / _ffmpeg_path.name.lower().replace("ffmpeg", "ffprobe"))

    probe_cmd = [
        _ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_streams", str(final_output),
    ]
    probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
    actual_frames = None
    actual_dur = None
    if probe_res.returncode == 0:
        info = _json.loads(probe_res.stdout)
        vs = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        actual_frames = int(vs.get("nb_frames", 0)) or None
        actual_dur = float(vs.get("duration", 0)) or None

    expected_total_frames = round(total_duration * target_fps)
    one_frame_sec = 1.0 / target_fps  # ~41.7 ms tolerance

    if actual_frames is not None:
        frame_diff = abs(actual_frames - expected_total_frames)
        if frame_diff <= 2:
            print(
                f"[Pipeline] Validation passed: {actual_frames} frames "
                f"(expected {expected_total_frames}, diff {frame_diff:+d})."
            )
        else:
            print(
                f"[Pipeline] WARNING: Frame count mismatch — got {actual_frames} "
                f"frames, expected {expected_total_frames} (diff {frame_diff:+d} frames = "
                f"{frame_diff * one_frame_sec * 1000:+.0f} ms)."
            )
    elif actual_dur is not None:
        dur_diff = actual_dur - total_duration
        if abs(dur_diff) <= one_frame_sec * 2:
            print(
                f"[Pipeline] Validation passed: output duration {actual_dur:.4f}s "
                f"(expected {total_duration:.4f}s, diff {dur_diff*1000:+.1f} ms)."
            )
        else:
            print(
                f"[Pipeline] WARNING: Duration mismatch — output {actual_dur:.4f}s, "
                f"expected {total_duration:.4f}s, diff {dur_diff*1000:+.1f} ms."
            )
    else:
        out_info = get_video_info(final_output)
        dur_diff = out_info["duration"] - total_duration
        if abs(dur_diff) > 0.1:
            print(
                f"[Pipeline] Warning: Duration discrepancy (Source: {total_duration:.2f}s, "
                f"Output: {out_info['duration']:.2f}s, Diff: {dur_diff:.2f}s)"
            )
        else:
            print(f"[Pipeline] Validation passed: Duration matches ({out_info['duration']:.2f}s).")

    manifest["status"] = "completed"
    manifest["final_output"] = str(final_output)
    manifest["completed_at"] = time.time()
    # Correct the dry_run flag: only mark true when this pipeline run was actually
    # a dry run.  Previous manifests could be left with dry_run=true even after
    # real Sync Labs results were stored, which would confuse future resume logic.
    manifest["dry_run"] = dry_run
    manifest["stats"] = {
        "total_duration": total_duration,
        "expected_total_frames": expected_total_frames,
        "actual_total_frames": actual_frames,
        "speech_duration": total_speech_sec,
        "passthrough_duration": total_pass_sec,
        "speech_segments_count": len(speech_segments),
        "passthrough_segments_count": len(passthrough_segments),
        "broll_filter_enabled": broll_filter,
    }
    write_json(manifest_path, manifest)

    print(f"\n[Pipeline] Successfully produced dubbed & lip-synced video!")
    print(f"Output: {final_output}\n")
    return final_output, manifest
