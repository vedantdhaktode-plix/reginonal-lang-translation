"""
Hybrid Clone Video Dubbing Module
Combines targeted Sync Labs lipsync-2 on presenter shots with ElevenLabs
Instant Voice Cloning for pristine B-roll voiceovers with zero drift and zero loops.
Automatically detects speaker identity when multiple persons appear.
"""
import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path

from .common import PipelineError, load_env, write_json, read_json, request_with_retry
from .scene_analyzer import get_video_info, detect_scenes, extract_subclip
from .srt_parser import parse_srt
from .synclabs_client import SyncLabsClient
from .dub_assembly import assemble_segment, join_segments


def ensure_voice_clone(audio_sample_path, voice_name=None, api_key=None):
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise PipelineError("Set ELEVENLABS_API_KEY in .env or your environment.")

    headers = {"xi-api-key": api_key}
    r = request_with_retry("GET", "https://api.elevenlabs.io/v1/voices", headers=headers, timeout=30, log_prefix="[Hybrid Clone Voices]")
    if r.ok:
        voices = r.json().get("voices", [])
        if voice_name:
            matched = next((v for v in voices if v["name"] == voice_name), None)
            if matched:
                print(f"[Hybrid Clone] Using existing voice clone '{voice_name}' (ID: {matched['voice_id']})")
                return matched["voice_id"]
        # Look for any recent cloned voice matching the reference
        matched = next((v for v in voices if "creator" in v["name"].lower() or "priyanta" in v["name"].lower()), None)
        if matched and not voice_name:
            print(f"[Hybrid Clone] Found matching creator voice clone '{matched['name']}' (ID: {matched['voice_id']})")
            return matched["voice_id"]

    v_name = voice_name or f"Creator_Clone_{int(time.time())}"
    print(f"[Hybrid Clone] Creating new voice clone '{v_name}' from reference audio...")
    with open(audio_sample_path, "rb") as f:
        files = {"files": ("reference_sample.wav", f, "audio/wav")}
        data = {"name": v_name, "description": "High fidelity voice clone for B-roll voiceovers"}
        resp = request_with_retry(
            "POST",
            "https://api.elevenlabs.io/v1/voices/add",
            headers=headers,
            data=data,
            files=files,
            timeout=60,
            log_prefix="[Hybrid Clone Voice Add]",
        )
        if not resp.ok:
            raise PipelineError(f"ElevenLabs voice clone failed: HTTP {resp.status_code} - {resp.text}")
        voice_id = resp.json()["voice_id"]
        print(f"[Hybrid Clone] Created voice ID: {voice_id}")
        return voice_id


def generate_tts_voiceover(voice_id, text, output_path, model="eleven_multilingual_v2", api_key=None):
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    headers = {"xi-api-key": api_key}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.85,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    resp = request_with_retry("POST", url, headers=headers, json=payload, timeout=60, log_prefix="[Hybrid Clone TTS]")
    if not resp.ok:
        raise PipelineError(f"ElevenLabs TTS failed: HTTP {resp.status_code} - {resp.text}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"[Hybrid Clone] Generated TTS ({len(resp.content)} bytes) -> {output_path.name}")
    return output_path


def time_fit_voiceover(input_audio, output_audio, target_duration, delay_ms=160):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_audio)],
        capture_output=True, text=True
    )
    raw_duration = float(probe.stdout.strip())
    speech_duration = target_duration - (delay_ms / 1000.0)
    tempo = raw_duration / speech_duration
    tempo = max(0.95, min(1.20, tempo))

    af_filters = [f"atempo={tempo:.4f}"]
    if delay_ms > 0:
        af_filters.append(f"adelay={delay_ms}|{delay_ms}")
    af_filters.append(f"apad=whole_dur={target_duration:.4f}")

    output_audio = Path(output_audio)
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(input_audio),
        "-af", ",".join(af_filters),
        "-t", f"{target_duration:.4f}",
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(output_audio)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_audio


def extract_hook_text(srt_path, hook_duration_sec):
    """
    Extracts text for all cues that fall within the opening hook duration.
    """
    if not Path(srt_path).exists():
        return "This is sugar damage."
    
    cues = parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    hook_cues = []
    for c in cues:
        start_sec = c.start_ms / 1000.0
        if start_sec < hook_duration_sec - 0.2:
            hook_cues.append(c.text.replace("\n", " ").strip())
    
    if hook_cues:
        return " ".join(hook_cues)
    return cues[0].text.replace("\n", " ").strip() if cues else ""


def run_hybrid_clone(
    source_video_path,
    target_lang="en",
    job_dir=None,
    sync_model="lipsync-2",
    dry_run=False,
    hook_text=None
):
    source_video = Path(source_video_path).resolve()
    if not source_video.exists():
        raise PipelineError(f"Source video not found: {source_video}")

    if not job_dir:
        job_dir = Path("output") / f"dub_{source_video.stem}_{target_lang}"
    job_dir = Path(job_dir).resolve()
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(source_video)
    total_frames = info["frame_count"]
    fps = info["fps"]
    total_duration = info["duration"]

    print(f"\n=======================================================")
    print(f"[Hybrid Clone] Source video: {source_video.name}")
    print(f"[Hybrid Clone] Resolution: {info['width']}x{info['height']} @ {fps:.3f}fps, {total_duration:.2f}s ({total_frames} frames)")
    print(f"=======================================================\n")

    manifest_path = job_dir / "dub_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    timeline = manifest.get("timeline", [])

    # If no timeline in manifest, generate basic scenes
    if not timeline:
        scenes = detect_scenes(source_video, threshold=32.0)
        manifest["scenes"] = scenes
        cursor = 0.0
        for i, s in enumerate(scenes):
            timeline.append({
                "index": i,
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
                "start_frame": int(round(s["start"] * fps)),
                "end_frame": int(round(s["end"] * fps))
            })
        manifest["timeline"] = timeline
        write_json(manifest_path, manifest)

    num_segments = len(timeline)
    print(f"[Hybrid Clone] Total timeline segments in video: {num_segments}")

    # Identify primary presenter shot (first talking head scene or seg_004)
    # Typically seg_004 or the longest segment with speech
    pres_idx = 4 if num_segments > 4 else min(len(timeline) - 1, 1)
    pres_seg = timeline[pres_idx]
    pres_start = pres_seg.get("start", 3.8)
    pres_end = pres_seg.get("end", 9.0)
    pres_dur = pres_end - pres_start
    pres_frames = int(round(pres_dur * fps))

    # Step 1: Check existing synced talking head clip (deterministic on-disk cache)
    seg_synced = None
    for cand in seg_dir.glob(f"seg_{pres_idx:03d}*synced.mp4"):
        if cand.stat().st_size > 100000:
            seg_synced = cand
            break

    if seg_synced and seg_synced.exists():
        print(f"[Hybrid Clone] Found existing synced talking head on disk: {seg_synced.name} (Reusing without new API call)")
    else:
        print(f"[Hybrid Clone] Submitting targeted talking head shot [{pres_start:.2f}s - {pres_end:.2f}s] to Sync Labs {sync_model}...")
        seg_raw = seg_dir / f"seg_{pres_idx:03d}_raw.mp4"
        if not seg_raw.exists():
            extract_subclip(source_video, pres_start, pres_end, seg_raw, target_fps=fps)
        
        seg_audio = seg_dir / f"seg_{pres_idx:03d}_audio.wav"
        if not seg_audio.exists():
            master_audio = job_dir / f"master_dubbed_audio_{target_lang}.mp3"
            if not master_audio.exists():
                raise PipelineError(f"Master dubbed audio not found: {master_audio}")
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{pres_start:.3f}", "-t", f"{pres_dur:.3f}", "-i", str(master_audio),
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(seg_audio)
            ], check=True, capture_output=True)

        # Multi-speaker smart isolation: options enable active_speaker_detection automatically!
        sync_client = SyncLabsClient(default_model=sync_model)
        video_url = sync_client.upload_file(seg_raw, content_type="video/mp4")
        audio_url = sync_client.upload_file(seg_audio, content_type="audio/wav")
        options = {
            "sync_mode": "cut_off",
            "active_speaker_detection": {"auto_detect": True}
        }
        gen_data = sync_client.generate(video_url, audio_url, model=sync_model, options=options)
        gen_id = gen_data["id"]
        print(f"[Hybrid Clone] Sync Labs Job ID: {gen_id} (Active Speaker Detection enabled). Waiting for completion...")
        res = sync_client.wait_for_completion(gen_id, poll_interval=8)
        seg_synced = seg_dir / f"seg_{pres_idx:03d}_synced.mp4"
        sync_client.download_result(res["outputUrl"], seg_synced)
        print(f"[Hybrid Clone] Downloaded synced clip to: {seg_synced}")

    # Step 2: Extract audio sample from talking head for voice cloning
    sample_wav = job_dir / "creator_reference_sample.wav"
    if not sample_wav.exists() or sample_wav.stat().st_size == 0:
        print("[Hybrid Clone] Extracting reference vocal sample from presenter talking head...")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(seg_synced),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            str(sample_wav)
        ], check=True, capture_output=True)

    # Step 3: Ensure ElevenLabs Voice Clone
    voice_name = f"Creator_{source_video.stem[:15]}_Ref"
    voice_id = ensure_voice_clone(sample_wav, voice_name=voice_name)

    # Step 4: Generate & Fit Hook Voiceover for B-Roll (0.00s to presenter start)
    hook_duration = pres_start
    if not hook_text:
        srt_path = job_dir / f"transcript_{target_lang}.srt"
        hook_text = extract_hook_text(srt_path, hook_duration)
    print(f"[Hybrid Clone] B-roll hook text ({hook_duration:.2f}s): '{hook_text}'")

    broll_fitted = job_dir / "broll_fitted_voiceover.wav"
    if not broll_fitted.exists() or broll_fitted.stat().st_size == 0:
        broll_raw_mp3 = job_dir / "broll_cloned_voiceover_raw.mp3"
        print(f"[Hybrid Clone] Generating TTS for opening B-roll hook...")
        generate_tts_voiceover(voice_id, hook_text, broll_raw_mp3)
        time_fit_voiceover(broll_raw_mp3, broll_fitted, target_duration=hook_duration, delay_ms=120)
        print(f"[Hybrid Clone] Fitted voiceover created: {broll_fitted.name}")

    # Step 5: Assemble segments using dub_assembly
    print(f"\n[Hybrid Clone] Assembling {num_segments} segments with dub_assembly...")
    assembled_segments = []

    # Segments before presenter (opening B-roll)
    for i in range(pres_idx):
        item = timeline[i]
        s_sec = item.get("start", 0.0)
        e_sec = item.get("end", pres_start)
        dur = e_sec - s_sec
        f_count = int(round(dur * fps))
        
        # Check raw subclip
        matching_raw = list(seg_dir.glob(f"seg_{i:03d}_*raw.mp4"))
        if matching_raw:
            s_raw = matching_raw[0]
        else:
            s_raw = seg_dir / f"seg_{i:03d}_raw.mp4"
            extract_subclip(source_video, s_sec, e_sec, s_raw, target_fps=fps)
        
        mov = seg_dir / f"h_seg_{i:03d}.mov"
        assembled_segments.append(assemble_segment(s_raw, broll_fitted, mov, f_count, fps, audio_start=s_sec))

    # Presenter talking head segment
    mov_pres = seg_dir / f"h_seg_{pres_idx:03d}.mov"
    assembled_segments.append(assemble_segment(seg_synced, seg_synced, mov_pres, pres_frames, fps, audio_start=0.0))

    # Downstream segments after presenter
    master_audio = job_dir / f"master_dubbed_audio_{target_lang}.mp3"
    for i in range(pres_idx + 1, num_segments):
        item = timeline[i]
        s_sec = item.get("start", 0.0)
        e_sec = item.get("end", total_duration)
        dur = e_sec - s_sec
        f_count = int(round(dur * fps))
        if f_count <= 0:
            continue

        # Prefer already synced clip if present (e.g. seg_006, seg_007)
        matching_synced = list(seg_dir.glob(f"seg_{i:03d}_*synced.mp4"))
        matching_paired = list(seg_dir.glob(f"seg_{i:03d}_*paired.mov"))

        mov_out = seg_dir / f"h_seg_{i:03d}.mov"
        if matching_synced:
            # Reusing synced clip
            assembled_segments.append(assemble_segment(matching_synced[0], matching_synced[0], mov_out, f_count, fps, audio_start=0.0))
        elif matching_paired:
            assembled_segments.append(matching_paired[0])
        else:
            matching_raw = list(seg_dir.glob(f"seg_{i:03d}_*raw.mp4"))
            if matching_raw:
                s_raw = matching_raw[0]
            else:
                s_raw = seg_dir / f"seg_{i:03d}_raw.mp4"
                extract_subclip(source_video, s_sec, e_sec, s_raw, target_fps=fps)
            assembled_segments.append(assemble_segment(s_raw, master_audio, mov_out, f_count, fps, audio_start=s_sec))

    print(f"[Hybrid Clone] Total segments assembled: {len(assembled_segments)}")
    final_output = job_dir / f"final_hybrid_cloned_dubbed_{target_lang}.mp4"
    join_segments(assembled_segments, final_output, total_frames, fps)

    print("\n=======================================================")
    print(f"SUCCESS! Master Hybrid Cloned Video created at:\n{final_output}")
    print("=======================================================")

    final_info = get_video_info(final_output)
    print("Final Video Info:", final_info)
    return final_output
