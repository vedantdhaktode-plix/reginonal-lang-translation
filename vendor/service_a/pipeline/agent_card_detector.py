import io
import json
import logging
import math
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from models.subtitle import BoundingBox, SubtitleSegment
from pipeline.gemini_models import model_chain, note_failure

logger = logging.getLogger(__name__)

@dataclass
class AgentCardResult:
    card_detected: bool
    bbox: BoundingBox | None
    bg_hex_color: str | None
    text_hex_color: str | None
    description: str = ''

class AgentCardDetector:
    """Uses a vision model to locate subtitle cards and read their text,
    extracting its precise bounding box, background color, and text styling.
    """
    def __init__(self, config):
        self.config = config
        self.vision_provider = getattr(config, 'ocr_provider', 'gemini').lower()
        if self.vision_provider == 'groq':
            self.groq_api_key = getattr(config, 'groq_api_key', '') or os.getenv('GROQ_API_KEY', '')
            if not self.groq_api_key:
                raise ValueError('Set GROQ_API_KEY in .env to use Groq vision card detection.')
            self.groq_model = getattr(config, 'groq_vision_model', '') or 'qwen/qwen3.8-27b'
            return
        api_key = getattr(config, 'gemini_api_key', '') or os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            raise ValueError('GEMINI_API_KEY is required for AgentCardDetector')
        from google import genai
        self.client = genai.Client(api_key=api_key)
        pref = getattr(config, 'gemini_model', 'gemini-3.5-flash-lite') or 'gemini-3.5-flash-lite'
        if pref in ('gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash-lite'):
            pref = 'gemini-3.5-flash-lite'
        self.preferred_model = pref

    def extract_keyframe(self, video_path: str | Path, timestamp: float = 0.5) -> Image.Image:
        ffmpeg_bin = getattr(self.config, 'ffmpeg', 'ffmpeg')
        cmd = [
            ffmpeg_bin,
            '-loglevel', 'error',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            'pipe:1'
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f'Failed to extract keyframe at {timestamp}s: {proc.stderr.decode(errors="replace")}')
        return Image.open(io.BytesIO(proc.stdout))

    def detect_master_card(self, video_path: str | Path, metadata, sample_timestamps: list[float] | None = None) -> dict:
        """Analyzes 1 or more keyframes from the video to locate the stable master title card."""
        from google.genai import types

        timestamps = sample_timestamps or [1.0, 2.5, 4.0]
        prompt = (
            "Analyze this video keyframe. Detect the on-screen spoken dialogue or subtitle card/pill container.\n"
            "The subtitle/card container can appear anywhere: lower area (subtitles), middle, or top.\n"
            "CRITICAL: The bounding box MUST encompass the ENTIRE card/pill container from its very top boundary "
            "to its bottom boundary, and full width from left curve/edge to right curve/edge. Do not restrict the box to just the inner text.\n"
            "Return a JSON object with:\n"
            "- card_detected: boolean\n"
            "- box_2d: [ymin, xmin, ymax, xmax] normalized to 1000 (integers between 0 and 1000)\n"
            "- bg_hex_color: hex color of the card background (e.g. #000000 or #1C1C1E)\n"
            "- text_hex_color: hex color of the title text (e.g. #FFFFFF)\n"
            "- description: short description\n"
        )
        gen_config = types.GenerateContentConfig(
            response_mime_type='application/json',
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        for ts in timestamps:
            try:
                img = self.extract_keyframe(video_path, timestamp=ts)
                w, h = img.size

                # Attempt with preferred model, fallback if busy
                response = None
                for model_name in model_chain(self.preferred_model):
                    try:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[img, prompt],
                            config=gen_config
                        )
                        if response and response.text:
                            break
                    except Exception as exc:
                        note_failure(model_name, exc)
                        logger.warning('Gemini model %s failed: %s', model_name, exc)
                        continue

                if not response or not response.text:
                    continue

                raw = json.loads(response.text)
                if isinstance(raw, list) and len(raw) > 0:
                    raw = raw[0]
                if not isinstance(raw, dict) or not raw.get('card_detected'):
                    continue

                box = raw.get('box_2d', [])
                if len(box) == 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in box]
                    abs_x = int(xmin / 1000.0 * w)
                    abs_y = int(ymin / 1000.0 * h)
                    abs_w = int((xmax - xmin) / 1000.0 * w)
                    abs_h = int((ymax - ymin) / 1000.0 * h)

                    # Normalize color to 0xRRGGBB format
                    raw_bg = raw.get('bg_hex_color', '#F4D9B9') or '#F4D9B9'
                    clean_hex = raw_bg.lstrip('#')
                    patch_color = f"0x{clean_hex.upper()}"

                    text_color = raw.get('text_hex_color', '#1F1916')
                    if not text_color.startswith('#'):
                        text_color = f"#{text_color.lstrip('0x')}"

                    bbox = BoundingBox(abs_x, abs_y, abs_w, abs_h, patch_color=patch_color)
                    logger.info('Agent detected master title card: %r (bg=%s, text=%s)', bbox, patch_color, text_color)
                    return {
                        'card_detected': True,
                        'bbox': bbox,
                        'bg_hex_color': patch_color,
                        'text_hex_color': text_color,
                        'description': raw.get('description', '')
                    }
            except Exception as exc:
                logger.warning('Agent detection attempt at %ss failed: %s', ts, exc)

        logger.info('Agent did not detect a distinct title card pill.')
        return {'card_detected': False, 'bbox': None, 'bg_hex_color': None, 'text_hex_color': None}

    def detect_subtitles(self, video_path: str | Path, metadata) -> list[SubtitleSegment]:
        """Detects dialogue subtitle cards across the video using video transcript extraction
        combined with batched multimodal vision for exact per-card bounding box tracking.
        """
        if self.vision_provider == 'groq':
            return self._detect_subtitles_groq(video_path, metadata)

        from google.genai import types
        import tempfile, time, shutil
        from difflib import SequenceMatcher

        w = getattr(metadata, 'width', 1080) or 1080
        h = getattr(metadata, 'height', 1920) or 1920
        dur = float(getattr(metadata, 'duration', 0.0) or 9999.0)
        ffmpeg_bin = getattr(self.config, 'ffmpeg', 'ffmpeg')
        video_path = Path(video_path)

        # Step 1: Extract all subtitle card events across the full video timeline in 1 API call
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            upload_path = td_path / "proxy_timecode.mp4"
            font_opt = "fontfile='C\\:/Windows/Fonts/arial.ttf':" if Path("C:/Windows/Fonts/arial.ttf").exists() else ""
            vf_filter = f"scale=-2:480,drawtext={font_opt}text='%{{pts\\:hms}}':x=(w-tw)/2:y=20:fontsize=32:fontcolor=yellow:box=1:boxcolor=black@0.9:boxborderw=5"
            cmd = [
                ffmpeg_bin, '-loglevel', 'error', '-y', '-i', str(video_path),
                '-vf', vf_filter, '-c:v', 'libx264',
                '-preset', 'ultrafast', '-crf', '28',
                '-c:a', 'aac', '-b:a', '64k',
                str(upload_path)
            ]
            try:
                subprocess.run(cmd, check=True, timeout=90)
            except Exception as e:
                logger.warning('Failed to render timecode proxy, falling back to clean proxy: %s', e)
                subprocess.run([
                    ffmpeg_bin, '-loglevel', 'error', '-y', '-i', str(video_path),
                    '-vf', 'scale=-2:480', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '28',
                    '-c:a', 'aac', '-b:a', '64k',
                    str(upload_path)
                ], check=True, timeout=60)

            upload = self.client.files.upload(file=str(upload_path))
            try:
                while self.client.files.get(name=upload.name).state.name == "PROCESSING":
                    time.sleep(1)

                prompt_video = (
                    "Watch this video carefully. A yellow HH:MM:SS timecode is burned in a black box at the top center.\n"
                    "CHECK FOR ALL VISUAL ON-SCREEN SUBTITLE PILLS, CAPTIONS, PROMPTS, ANSWERS, LABELS, OR CALL-TO-ACTION CARDS (e.g. 'Comment DM', 'Comment now').\n"
                    "CRITICAL RULES:\n"
                    "1. ONLY extract text that is ACTUALLY VISIBLE ON SCREEN in the video frames.\n"
                    "2. DO NOT transcribe spoken audio dialogue if that text is NOT displayed visually on screen! If there are no on-screen subtitle cards while the person speaks, DO NOT output any cards for that time.\n"
                    "3. DO NOT extract physical text printed on product packaging, bottles, clothing graphics, or permanent channel watermarks.\n"
                    "4. Extract every distinct text-bearing panel even when it remains unchanged for the entire video. For a multi-section card (for example a dark question/header above a light answer), return EACH section as a separate item with its own exact text and timing. Never omit an adjacent panel merely because another panel looks more like spoken dialogue.\n"
                    "5. If on-screen subtitle pills or call-to-action cards appear, record their exact start and end timestamps in seconds using the yellow timecode.\n"
                    "If no visual subtitle cards or call-to-action text appear on screen, return an empty array [].\n"
                    "Return a JSON array of objects with:\n"
                    "  - start: float (seconds)\n"
                    "  - end: float (seconds)\n"
                    "  - text: string (exact words on that card)\n"
                )
                schema = {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["start", "end", "text"]
                    }
                }
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
                # Default Gemini video sampling is 1 fps, too coarse for cards that change
                # every 1-2 s. Ask for denser sampling; fall back to the plain file if refused.
                video_fps = float(os.getenv('GEMINI_VIDEO_FPS', '4'))
                dense_video = types.Part(
                    file_data=types.FileData(file_uri=upload.uri, mime_type=upload.mime_type),
                    video_metadata=types.VideoMetadata(fps=video_fps),
                )
                resp_video = None
                for m_name in model_chain(self.preferred_model):
                    for video_part in (dense_video, upload):
                        try:
                            resp_video = self.client.models.generate_content(
                                model=m_name,
                                contents=[video_part, prompt_video],
                                config=gen_config
                            )
                            if resp_video and resp_video.text:
                                break
                        except Exception as exc:
                            note_failure(m_name, exc)
                            logger.warning('Gemini model %s failed in video detection (%s): %s', m_name,
                                           'dense sampling' if video_part is dense_video else 'default sampling', exc)
                    if resp_video and resp_video.text:
                        break

                clean_json = (resp_video.text if resp_video else '[]').strip()
                raw_cards = json.loads(clean_json)
            finally:
                try:
                    self.client.files.delete(name=upload.name)
                except Exception:
                    pass

            if not isinstance(raw_cards, list) or not raw_cards:
                logger.warning('Video subtitle card extraction returned no cards')
                return []

            # Preferred path: frame-accurate local tracking of dark caption cards.
            # Gemini's text is kept, but timing and boxes come from the real frames.
            gemini_cards = []
            for item in raw_cards:
                try:
                    s_t = max(0.0, float(item.get('start', 0.0)))
                    e_t = min(dur, float(item.get('end', s_t + 1.0)))
                except (TypeError, ValueError, AttributeError):
                    continue
                txt = (item.get('text') or '').strip()
                if txt and e_t > s_t:
                    gemini_cards.append({'start': s_t, 'end': e_t, 'text': txt})
            tracked_preferred = []
            tracked_card_indices = set()
            try:
                from pipeline.card_tracker import track_cards, build_segments
                intervals = track_cards(video_path, metadata)
                tracked, coverage = build_segments(intervals, gemini_cards, self.config, dur)
                min_coverage = float(os.getenv('CARD_TRACK_MIN_COVERAGE', '0.6'))
                logger.info('Card tracker: %d intervals, %d segments, %.0f%% of Gemini dialogue time covered',
                            len(intervals), len(tracked), coverage * 100)
                if tracked and coverage >= min_coverage:
                    return tracked
                # Keep locally tracked geometry for individual cards that it
                # covers reliably. A low aggregate score can simply mean that a
                # few other captions were missed; discarding every accurate box
                # makes the vision fallback draw a second panel beside the source.
                for card_index, card in enumerate(gemini_cards):
                    local = [segment for segment in tracked if segment.event_index == card_index]
                    if not local:
                        continue
                    clipped = sorted(
                        (max(card['start'], segment.start), min(card['end'], segment.end))
                        for segment in local
                        if min(card['end'], segment.end) > max(card['start'], segment.start)
                    )
                    merged = []
                    for interval_start, interval_end in clipped:
                        if merged and interval_start <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
                        else:
                            merged.append((interval_start, interval_end))
                    local_coverage = sum(end - start for start, end in merged) / max(
                        0.01, card['end'] - card['start']
                    )
                    if local_coverage >= 0.75:
                        tracked_card_indices.add(card_index)
                        tracked_preferred.extend(local)
                logger.warning(
                    'Card tracker coverage %.0f%% below %.0f%%; preserving local geometry for %d/%d cards and grounding the rest from keyframes',
                    coverage * 100, min_coverage * 100,
                    len(tracked_card_indices), len(gemini_cards),
                )
            except Exception as exc:
                logger.warning('Card tracker failed, using Gemini keyframe boxes: %s', exc)

            # Sample densely through both timing boundaries.  Vision timestamps
            # are often rounded by 100-200 ms, and short-form edits can move a
            # floating label on the very last frame of a cue.  Three interior
            # samples miss that move and leave the original word exposed.
            card_items = []
            for item in raw_cards:
                s_t = max(0.0, float(item.get('start', 0.0)))
                e_t = min(dur, float(item.get('end', s_t + 1.0)))
                txt = (item.get('text') or '').strip()
                if not txt or e_t <= s_t:
                    continue
                sample_start = max(0.0, s_t - 0.25)
                sample_end = min(dur, e_t + 0.25)
                sample_span = sample_end - sample_start
                sample_count = min(16, max(3, int(math.ceil(sample_span / 0.15)) + 1))
                sample_times = sorted({
                    round(sample_start + sample_span * point / (sample_count - 1), 3)
                    for point in range(sample_count)
                })
                images = []
                image_times = []
                for sample_idx, sample_t in enumerate(sample_times):
                    kf_path = td_path / f"kf_{sample_t}_{len(card_items)}_{sample_idx}.jpg"
                    cmd = [
                        ffmpeg_bin, '-loglevel', 'error', '-y', '-ss', str(sample_t),
                        '-i', str(video_path), '-vframes', '1',
                        '-vf', 'scale=-2:720', '-q:v', '3', str(kf_path)
                    ]
                    try:
                        subprocess.run(cmd, check=True, timeout=15)
                        img = Image.open(kf_path)
                        img.load()
                        images.append(img)
                        image_times.append(sample_t)
                    except Exception as exc:
                        logger.debug('Could not extract keyframe at %.2fs: %s', sample_t, exc)
                sample_step = min(
                    (later - earlier for earlier, later in zip(image_times, image_times[1:])),
                    default=0.15,
                )
                card_items.append({'start': s_t, 'end': e_t, 'text': txt,
                                   'images': images, 'image_times': image_times,
                                   'sample_step': sample_step})

            if not card_items:
                return []

            # Step 3: Batch image inspection with targeted text grounding
            BATCH_SIZE = 8
            card_boxes = {}
            vision_items = [(card_idx, sample_idx, card['text'], img)
                            for card_idx, card in enumerate(card_items)
                            for sample_idx, img in enumerate(card['images'])]
            for i in range(0, len(vision_items), BATCH_SIZE):
                valid_batch = vision_items[i:i + BATCH_SIZE]
                if not valid_batch:
                    continue
                imgs = [entry[3] for entry in valid_batch]
                image_descriptions = []
                for sub_idx, (card_idx, sample_idx, text, _) in enumerate(valid_batch, 1):
                    image_descriptions.append(f"- Image {sub_idx}: Subtitle card {card_idx + 1}, sample {sample_idx + 1}, dialogue is \"{text}\"")
                img_desc_text = "\n".join(image_descriptions)

                prompt_batch = (
                    f"You are given {len(imgs)} video keyframes in order (Image 1 to Image {len(imgs)}).\n"
                    "For EACH image, find the exact on-screen panel displaying the specified visible text:\n"
                    f"{img_desc_text}\n\n"
                    "CRITICAL RULES:\n"
                    "1. First decide whether the matching text is printed on a solid rectangular panel/card. If yes, panel_box_2d MUST be the full OUTER boundary of that panel including all background padding, not merely the glyphs. If no panel exists, tightly enclose all glyphs and outlines in box_2d.\n"
                    "2. NEVER select speaker name badges / lower thirds, text printed on products or packaging, prop stands, diagram labels, or brand logos.\n"
                    "   If the matching dialogue text is not visible as a subtitle card, set card_detected to false.\n"
                    "3. If the dialogue subtitle is at the bottom, return the bottom box. If it moved up into the middle or top, return the middle/top box where that dialogue appears.\n"
                    f"Return a JSON array of {len(imgs)} objects in exact matching order:\n"
                    f"- image_index: integer (1 to {len(imgs)})\n"
                    "- card_detected: boolean (true if the matching dialogue subtitle card is visible)\n"
                    "- is_solid_panel: boolean\n"
                    "- panel_box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000 for the complete panel, or [] when text floats directly over video\n"
                    "- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000 around the complete text glyphs\n"
                    "- bg_hex_color: hex color sampled from the panel background away from its text (not the glyph color)\n"
                )
                try:
                    gen_cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                    resp_b = None
                    for m_name in model_chain(self.preferred_model):
                        try:
                            resp_b = self.client.models.generate_content(
                                model=m_name,
                                contents=[prompt_batch, *imgs],
                                config=gen_cfg
                            )
                            if resp_b and resp_b.text:
                                break
                        except Exception as m_exc:
                            note_failure(m_name, m_exc)
                            logger.warning('Model %s failed in batch keyframe detection: %s', m_name, m_exc)
                    b_res = json.loads(resp_b.text or '[]') if resp_b else []
                    if isinstance(b_res, list):
                        for res_item, (card_idx, sample_idx, _, _) in zip(b_res, valid_batch):
                            card_boxes.setdefault(card_idx, []).append((sample_idx, res_item))
                except Exception as exc:
                    logger.warning('Batch card detection failed for batch %d: %s', i // BATCH_SIZE, exc)
                time.sleep(0.5)

            # Step 4: Construct SubtitleSegments with verified per-card bounding boxes
            segments = []
            for idx, item in enumerate(card_items):
                boxes = []
                colors = []
                for sample_idx, box_info in card_boxes.get(idx, []):
                    solid_panel = bool(box_info.get('is_solid_panel'))
                    panel_box = box_info.get('panel_box_2d', [])
                    box = panel_box if solid_panel and isinstance(panel_box, list) and len(panel_box) == 4 else box_info.get('box_2d', [])
                    if isinstance(box, list) and len(box) > 0 and isinstance(box[0], list):
                        box = box[0]
                    if box_info.get('card_detected') and isinstance(box, list) and len(box) == 4:
                        ymin, xmin, ymax, xmax = [float(v) for v in box]
                        pad_x, pad_y = (0, 0) if solid_panel else (20, 5)
                        left = max(0, int(xmin / 1000.0 * w) - pad_x)
                        top = max(0, int(ymin / 1000.0 * h) - pad_y)
                        right = min(w, int(xmax / 1000.0 * w) + pad_x)
                        bottom = min(h, int(ymax / 1000.0 * h) + pad_y)
                        bw_, bh_ = right - left, bottom - top
                        # A subtitle card is a short, wide strip. Reject boxes that Gemini
                        # grounded onto packaging, props or large regions.
                        if bh_ > 0 and bh_ <= h * 0.14 and bw_ >= w * 0.12 and bw_ / bh_ >= 2.0:
                            sample_time = item['image_times'][sample_idx]
                            sample_image = item['images'][sample_idx]
                            boxes.append((sample_time, left, top, right, bottom, sample_image))
                            colors.append((box_info.get('bg_hex_color') or '').strip().lstrip('#'))
                        elif right > left and bottom > top:
                            logger.info('Rejected non-caption box %r for segment %d', (left, top, right, bottom), idx + 1)

                if boxes:
                    # Discard isolated geometry hallucinations. Boundary frames
                    # can contain an adjacent short label while the requested cue
                    # is absent; treating that box as the same cue combines a tiny
                    # top label with a wide bottom caption and inflates every mask.
                    if len(boxes) >= 3:
                        median_w = statistics.median(box[3] - box[1] for box in boxes)
                        median_h = statistics.median(box[4] - box[2] for box in boxes)
                        keep = [
                            sample_idx for sample_idx, box in enumerate(boxes)
                            if 0.55 * median_w <= box[3] - box[1] <= 1.80 * median_w
                            and 0.55 * median_h <= box[4] - box[2] <= 1.80 * median_h
                        ]
                        if len(keep) >= 2:
                            rejected = len(boxes) - len(keep)
                            if rejected:
                                logger.info('Rejected %d geometry outlier(s) for segment %d', rejected, idx + 1)
                            boxes = [boxes[sample_idx] for sample_idx in keep]
                            colors = [colors[sample_idx] for sample_idx in keep]

                    # Keep spatially distant samples: short-form videos commonly
                    # jump between shots while the same label remains visible.
                    # The renderer follows these points stepwise instead of leaving
                    # one fixed mask behind at the first/median location.
                    max_w = max(b[3] - b[1] for b in boxes)
                    max_h = max(b[4] - b[2] for b in boxes)
                    abs_x, abs_y = boxes[0][1], boxes[0][2]
                    raw_bg = next((c for c in colors if len(c) == 6), '')
                    if len(raw_bg) == 6:
                        try:
                            r, g, b = int(raw_bg[0:2], 16), int(raw_bg[2:4], 16), int(raw_bg[4:6], 16)
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            # Preserve the detected panel colour. The renderer chooses
                            # a contrasting text colour, so light answer panels remain
                            # light instead of becoming an unrelated black rectangle.
                            patch_color = f"0x{raw_bg.upper()}"
                        except Exception:
                            patch_color = getattr(self.config, 'overlay_patch_color', '0x000000')
                    else:
                        patch_color = getattr(self.config, 'overlay_patch_color', '0x000000')
                    bbox = BoundingBox(abs_x, abs_y, max_w, max_h, patch_color=patch_color)

                    # A solid caption panel has one dominant quantized colour.
                    # Floating outlined text does not; only those cues receive a
                    # small frame-cadence guard to prevent English edge flicker.
                    import numpy as np
                    dominance = []
                    dominant_rgb = []
                    for _, left, top, right, bottom, sample_image in boxes:
                        sx, sy = sample_image.width / float(w), sample_image.height / float(h)
                        crop = sample_image.crop((max(0, int(left * sx)), max(0, int(top * sy)),
                                                  min(sample_image.width, int(right * sx)),
                                                  min(sample_image.height, int(bottom * sy))))
                        pixels = np.asarray(crop.convert('RGB'), dtype=np.uint8)
                        if pixels.size:
                            flat_pixels = pixels.reshape(-1, 3)
                            quantized = (flat_pixels // 32)
                            values, counts = np.unique(quantized, axis=0, return_counts=True)
                            dominance.append(float(counts.max() / counts.sum()))
                            winning_bin = values[int(counts.argmax())]
                            members = flat_pixels[np.all(quantized == winning_bin, axis=1)]
                            dominant_rgb.append(tuple(int(value) for value in np.median(members, axis=0)))
                    floating_text = bool(dominance and sum(dominance) / len(dominance) < 0.55)
                    if not floating_text and dominant_rgb:
                        panel_rgb = tuple(int(statistics.median(color[channel] for color in dominant_rgb))
                                          for channel in range(3))
                        bbox.patch_color = f'0x{panel_rgb[0]:02X}{panel_rgb[1]:02X}{panel_rgb[2]:02X}'
                    if floating_text and len(boxes) >= 2:
                        # Confirmed boundary samples are more accurate than the
                        # rounded video-level timestamps.  Half a sampling step
                        # covers the frames between the last negative and first
                        # positive observation without adding a visible second.
                        timing_pad = min(0.12, max(0.04, item['sample_step'] * 0.6))
                        start = max(0.0, min(box[0] for box in boxes) - timing_pad)
                        end = min(dur, max(box[0] for box in boxes) + timing_pad)
                    else:
                        start = item['start']
                        end = item['end']
                    bbox_track = [(sample_time, left, top, max_w, max_h)
                                  for sample_time, left, top, _, _, _ in sorted(boxes)]
                else:
                    logger.warning('No subtitle-card box was confirmed for segment %d; falling back to the next OCR detector', idx + 1)
                    return []

                segments.append(SubtitleSegment(
                    len(segments) + 1,
                    start,
                    end,
                    item['text'],
                    bbox=bbox,
                    ocr_confidence=1.0,
                    bbox_track=bbox_track,
                    bbox_track_mode='step',
                    timing_guarded=floating_text,
                    event_index=idx
                ))

            # Step 5: Post-processing to ensure seamless coverage without gaps
            if segments and segments[0].start <= 0.8:
                segments[0].start = 0.0

            # Bridge small gaps between consecutive segments (<= 0.8s) so background mask never flickers off
            for i in range(len(segments) - 1):
                gap = segments[i + 1].start - segments[i].end
                if 0 <= gap <= 0.8:
                    segments[i].end = segments[i + 1].start
                elif gap < 0 and not (
                    segments[i].timing_guarded or segments[i + 1].timing_guarded
                ):
                    segments[i].end = segments[i + 1].start

            segments = [s for s in segments if s.end > s.start + 0.15]
            if tracked_preferred:
                segments = tracked_preferred + [
                    segment for segment in segments
                    if segment.event_index not in tracked_card_indices
                ]
                segments.sort(key=lambda segment: (segment.start, segment.end, segment.event_index or -1))
                for segment_id, segment in enumerate(segments, 1):
                    segment.id = segment_id
                logger.info(
                    'Hybrid geometry: %d local tracked segment(s), %d keyframe-grounded segment(s)',
                    len(tracked_preferred), len(segments) - len(tracked_preferred),
                )
            return segments

    def _detect_subtitles_groq(self, video_path: str | Path, metadata) -> list[SubtitleSegment]:
        """Track card geometry locally, then read up to three card crops per Groq vision call."""
        import base64
        import cv2
        import requests
        from pipeline.card_tracker import track_cards

        intervals = track_cards(video_path, metadata)
        if not intervals:
            logger.warning('Local card tracker found no subtitle cards for Groq vision OCR')
            return []

        segments = []
        batch_size = 3  # Groq vision currently accepts at most three images per request.
        duration = float(getattr(metadata, 'duration', 0.0) or max(iv.end for iv in intervals))
        for offset in range(0, len(intervals), batch_size):
            batch = [iv for iv in intervals[offset:offset + batch_size]
                     if iv.crop is not None and iv.crop.size > 0]
            if not batch:
                continue
            content = [{
                'type': 'text',
                'text': (
                    'Read the complete visible spoken dialogue subtitle text in each numbered image. '
                    'Each image is a horizontal crop showing the subtitle line in a video. '
                    'CRITICAL: Read ALL words from the very far left to the very far right. '
                    'Never omit brand names (e.g. Plix, Hydrangea, Shampoo), pronouns, or starting words. '
                    'Return the exact visible words in the original language and script. '
                    'If no readable dialogue subtitle is visible, return an empty string. '
                    f'Return only JSON: {{"cards":[{{"index":1,"text":"..."}}]}} for all {len(batch)} images. '
                    'The index is the image position in this request, starting at 1.'
                ),
            }]
            for i, interval in enumerate(batch, 1):
                ok, encoded = cv2.imencode('.jpg', interval.crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError(f'Could not encode subtitle card crop {offset + i}')
                data_url = 'data:image/jpeg;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')
                content.append({'type': 'text', 'text': f'Image {i} is subtitle card {offset + i + 1}.'})
                content.append({'type': 'image_url', 'image_url': {'url': data_url}})

            payload = {
                'model': self.groq_model,
                'temperature': 0,
                'max_tokens': 350,
                'response_format': {'type': 'json_object'},
                'messages': [
                    {'role': 'system', 'content': 'You are a careful OCR reader for video subtitle text. Return valid JSON only.'},
                    {'role': 'user', 'content': content},
                ],
            }
            import time
            response = None
            for attempt in range(6):
                try:
                    response = requests.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers={'Authorization': f'Bearer {self.groq_api_key}', 'Content-Type': 'application/json'},
                        json=payload,
                        timeout=self.config.api_timeout,
                    )
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
                    if attempt < 5:
                        wait_sec = 2.0 * (attempt + 1)
                        logger.warning('Groq vision OCR connection dropped (%s, attempt %d/6), waiting %.1fs...', conn_err, attempt + 1, wait_sec)
                        time.sleep(wait_sec)
                        continue
                    raise
                if response.status_code == 429:
                    import re as _re
                    match = _re.search(r'try again in ([\d\.]+)s', response.text)
                    wait_sec = float(match.group(1)) + 1.0 if match else 2.0 * (attempt + 1)
                    logger.warning('Groq vision OCR hit 429 rate limit (attempt %d/6), waiting %.1fs...', attempt + 1, wait_sec)
                    time.sleep(wait_sec)
                    continue
                break
            if response is None or response.status_code >= 400:
                err_text = response.text[:800] if response is not None else 'No response'
                raise RuntimeError(f'Groq vision OCR request failed ({getattr(response, "status_code", 0)}): {err_text}')
            time.sleep(0.5)
            try:
                result = json.loads(response.json()['choices'][0]['message']['content'])
                card_texts = {
                    int(item['index']) - 1: str(item.get('text', '')).strip()
                    for item in result.get('cards', [])
                    if isinstance(item, dict) and 'index' in item
                }
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError('Groq vision OCR returned invalid card JSON') from exc

            for local_idx, interval in enumerate(batch):
                text = card_texts.get(local_idx, '')
                if not text:
                    logger.info('Groq vision OCR found no text on card %d', offset + local_idx + 1)
                    continue
                start = max(0.0, float(interval.start))
                end = min(duration, float(interval.end))
                if end <= start + 0.15:
                    continue
                raw_bb = interval.bbox()
                # Ensure the replacement overlay banner spans the full video width (x=0 to frame_w)
                # across the detected subtitle vertical band, with slight vertical padding.
                # This guarantees 100% masking coverage — no original text can peek out on edges.
                frame_w = int(getattr(metadata, 'width', 1080) or 1080)
                pad_y = max(3, int(raw_bb.height * 0.15))
                new_y = max(0, raw_bb.y - pad_y)
                new_h = raw_bb.height + 2 * pad_y
                patch_color = raw_bb.patch_color or '0x121212'
                padded_bb = BoundingBox(0, new_y, frame_w, new_h, patch_color=patch_color)
                segments.append(SubtitleSegment(
                    len(segments) + 1, start, end, text,
                    bbox=padded_bb, ocr_confidence=0.95,
                ))

        logger.info('Groq vision OCR read %d subtitle cards from %d locally tracked intervals', len(segments), len(intervals))
        return segments
