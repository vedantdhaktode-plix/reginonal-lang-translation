import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from models.subtitle import BoundingBox, SubtitleSegment
from pipeline.subtitle_extractor import normalize

@dataclass
class OCRDetection:
    text: str
    bbox: BoundingBox
    confidence: float

class OCRProvider(ABC):
    @abstractmethod
    def detect(self, frame) -> list[OCRDetection]: pass

class TesseractOCRProvider(OCRProvider):
    def __init__(self, language):
        try:
            import os
            import shutil
            from pathlib import Path
            import pytesseract
            if not shutil.which('tesseract'):
                candidates = [
                    Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Tesseract-OCR/tesseract.exe',
                    Path('C:/Program Files/Tesseract-OCR/tesseract.exe'),
                    Path('C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'),
                ]
                for c in candidates:
                    if c.is_file():
                        pytesseract.pytesseract.tesseract_cmd = str(c)
                        break
            pytesseract.get_tesseract_version()
        except (ImportError, OSError) as exc:
            raise RuntimeError('OCR unavailable. Install requirements-ocr.txt and the Tesseract executable/language packs; add Tesseract to PATH.') from exc
        self.engine=pytesseract
        self.language=language

    def detect(self, frame, psm=11):
        data=self.engine.image_to_data(frame,lang=self.language,config=f'--psm {psm}',output_type=self.engine.Output.DICT,timeout=30)
        groups={}
        for i,text in enumerate(data['text']):
            confidence=float(data['conf'][i])/100
            if confidence<0.50 or not normalize(text): continue
            key=(data['block_num'][i],data['par_num'][i])
            groups.setdefault(key,[]).append((i,text,confidence))
        result=[]
        for words in groups.values():
            x=min(data['left'][i] for i,_,_ in words); y=min(data['top'][i] for i,_,_ in words)
            right=max(data['left'][i]+data['width'][i] for i,_,_ in words)
            bottom=max(data['top'][i]+data['height'][i] for i,_,_ in words)
            raw = ' '.join(t for _,t,_ in words)
            cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', raw).strip()
            norm = normalize(cleaned) or normalize(raw)
            if norm:
                result.append(OCRDetection(norm,BoundingBox(x,y,right-x,bottom-y),sum(c for _,_,c in words)/len(words)))
        return result

def _has_light_background(frame, box, min_brightness):
    import numpy as np
    pad = max(8, int(box.height * 0.4))
    px1, py1 = max(0, box.x - pad), max(0, box.y - pad)
    px2 = min(frame.shape[1], box.x + box.width + pad)
    py2 = min(frame.shape[0], box.y + box.height + pad)
    region = frame[py1:py2, px1:px2]
    mask = np.ones(region.shape[:2], dtype=bool)
    ix, iy = box.x - px1, box.y - py1
    mask[iy:iy + box.height, ix:ix + box.width] = False
    border = region[mask]
    if len(border) < 10: return False
    return float(np.mean(border)) > min_brightness

def find_beige_box(frame, max_y_ratio=0.35):
    """Detects a beige/peach title pill/box in the upper portion of the frame.
    Returns BoundingBox with patch_color or None.
    """
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    crop_h = int(h * max_y_ratio)
    top_crop = frame[:crop_h, :]
    hsv = cv2.cvtColor(top_crop, cv2.COLOR_BGR2HSV)
    lower_beige = np.array([10, 25, 160], dtype=np.uint8)
    upper_beige = np.array([38, 130, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_beige, upper_beige)
    
    # Horizontal closing to bridge letter gaps without vertical bleeding
    k_w = max(31, int(w * 0.06))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1)))
    
    row_spans = []
    min_y_start = int(h * 0.02) # Skip very top margin
    for y in range(crop_h):
        if y < min_y_start:
            row_spans.append((0, 0, 0))
            continue
        row = (closed[y, :] > 0).astype(np.int32)
        diff = np.diff(np.pad(row, (1, 1), 'constant'))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            n = min(len(starts), len(ends))
            lengths = ends[:n] - starts[:n]
            max_idx = np.argmax(lengths)
            row_spans.append((int(lengths[max_idx]), int(starts[max_idx]), int(ends[max_idx])))
        else:
            row_spans.append((0, 0, 0))
            
    min_span_w = int(w * 0.35)
    valid_rows = [i for i, span in enumerate(row_spans) if span[0] >= min_span_w]
    if not valid_rows:
        # Fallback to contour detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 2))
        cnt_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cnt_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            aspect = bw / float(bh) if bh > 0 else 0
            if y >= min_y_start and bw > (w * 0.35) and int(h * 0.02) <= bh <= int(h * 0.12) and aspect >= 2.2:
                box_region = top_crop[y:y+bh, x:x+bw]
                box_mask = mask[y:y+bh, x:x+bw]
                if np.sum(box_mask > 0) > 0.35 * area:
                    bgr = np.mean(box_region[box_mask > 0], axis=0).astype(int)
                    hex_color = f'0x{bgr[2]:02X}{bgr[1]:02X}{bgr[0]:02X}'
                    candidates.append((area, BoundingBox(int(x), int(y), int(bw), int(bh), patch_color=hex_color)))
        if not candidates: return None
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]
        
    blocks = []
    current_block = [valid_rows[0]]
    for r in valid_rows[1:]:
        if r - current_block[-1] <= 4:
            current_block.append(r)
        else:
            blocks.append(current_block)
            current_block = [r]
    blocks.append(current_block)
    
    min_h = int(h * 0.02)
    max_h = int(h * 0.12)
    valid_blocks = [b for b in blocks if min_h <= (b[-1] - b[0]) <= max_h]
    if not valid_blocks:
        return None
        
    best_block = max(valid_blocks, key=lambda b: len(b))
    by1, by2 = best_block[0], best_block[-1]
    bh = int(by2 - by1)
    bx1 = int(min(row_spans[y][1] for y in best_block))
    bx2 = int(max(row_spans[y][2] for y in best_block))
    bw = int(bx2 - bx1)
    
    box_region = top_crop[by1:by2, bx1:bx2]
    box_mask = mask[by1:by2, bx1:bx2]
    if np.sum(box_mask > 0) > 0.30 * (bw * bh):
        bgr = np.mean(box_region[box_mask > 0], axis=0).astype(int)
        hex_color = f'0x{bgr[2]:02X}{bgr[1]:02X}{bgr[0]:02X}'
        return BoundingBox(bx1, by1, bw, bh, patch_color=hex_color)
    return None


def _split_into_single_sentence_segments(raw_items: list, meta_dur: float) -> list[tuple[float, float, str]]:
    split_items = []
    pattern = r'(?<=[.!?])\s+|,\s*(?=(?:warna|otherwise|or|isme|ismein|it\s+has|it\s+contains|contains)\b)'
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        start = float(item.get('start', 0.0))
        end = min(float(item.get('end', start + 2.0)), meta_dur)
        txt = str(item.get('text', '')).strip()
        if not txt:
            continue
        parts = re.split(pattern, txt, flags=re.IGNORECASE)
        cleaned = []
        for p in parts:
            p = p.strip()
            if p.startswith(','):
                p = p[1:].strip()
            if not p:
                continue
            p = p[0].upper() + p[1:] if len(p) > 1 else p.upper()
            if not p.endswith(('.', '!', '?', ',', ':')):
                p += '!'
            cleaned.append(p)
        if len(cleaned) <= 1:
            split_items.append((start, end, txt))
        else:
            total_len = sum(len(c) for c in cleaned)
            total_dur = max(0.2, end - start)
            cur_start = start
            for i, c in enumerate(cleaned):
                if i == len(cleaned) - 1:
                    cur_end = end
                else:
                    share = (len(c) / total_len) * total_dur
                    cur_end = round(cur_start + share, 2)
                split_items.append((cur_start, cur_end, c))
                cur_start = cur_end
    return split_items

class GeminiVideoSubtitleExtractor:
    def __init__(self, config):
        self.config = config
        api_key = getattr(config, 'gemini_api_key', '') or os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            raise ValueError('GEMINI_API_KEY is required for Gemini subtitle extraction')
        from google import genai
        self.client = genai.Client(api_key=api_key)
        pref = getattr(config, 'gemini_model', 'gemini-3.5-flash-lite') or 'gemini-3.5-flash-lite'
        if pref in ('gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash-lite'):
            pref = 'gemini-3.5-flash-lite'
        self.model = pref

    def extract_subtitles(self, video_path, metadata) -> list[SubtitleSegment]:
        import tempfile, shutil, time, json
        from pathlib import Path
        video_path = Path(video_path)
        with tempfile.TemporaryDirectory() as td:
            upload_path = Path(td) / "upload.mp4"
            if video_path.stat().st_size > 15 * 1024 * 1024:
                ffmpeg_bin = getattr(self.config, 'ffmpeg', 'ffmpeg')
                cmd = [
                    ffmpeg_bin, '-y', '-i', str(video_path),
                    '-vf', 'scale=-2:480', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '28',
                    '-c:a', 'aac', '-b:a', '64k',
                    str(upload_path)
                ]
                import subprocess
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                shutil.copy2(video_path, upload_path)
            upload = self.client.files.upload(file=str(upload_path))
            try:
                while self.client.files.get(name=upload.name).state.name == "PROCESSING":
                    time.sleep(1)
                prompt = (
                    "Analyze this video ONLY for VISUAL ON-SCREEN SUBTITLE CARDS, CAPTIONS, OR CALL-TO-ACTION TEXT (e.g. 'Comment DM', 'Comment now', etc.).\n\n"
                    "CRITICAL RULES:\n"
                    "1. ONLY extract on-screen subtitle cards, burned captions, or call-to-action text overlays.\n"
                    "2. DO NOT extract spoken audio if there is NO text on screen! If the person is speaking but NO subtitle card appears on screen, DO NOT output any item for that time.\n"
                    "3. DO NOT extract physical text printed on product bottles, boxes, packaging, labels, or clothing graphics.\n"
                    "4. For each VISUAL text overlay/card:\n"
                    "   - start: exact timestamp in seconds when the text appears on screen\n"
                    "   - end: exact timestamp in seconds when the text disappears\n"
                    "   - text: exact visible on-screen text\n"
                    "   - box_2d: [ymin, xmin, ymax, xmax] outer bounding box of the text container normalized to 0-1000\n\n"
                    "If no subtitle cards or text overlays appear on screen, return an empty array []."
                )
                from google.genai import types
                schema = {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER", "description": "Start time in seconds"},
                            "end": {"type": "NUMBER", "description": "End time in seconds"},
                            "text": {"type": "STRING", "description": "Exact visible on-screen text"},
                            "box_2d": {
                                "type": "ARRAY",
                                "items": {"type": "INTEGER"},
                                "description": "[ymin, xmin, ymax, xmax] normalized to 0-1000"
                            }
                        },
                        "required": ["start", "end", "text", "box_2d"]
                    }
                }
                gen_config = types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[upload, prompt],
                    config=gen_config
                )
                if not resp or not resp.text:
                    return []
                clean_json = resp.text.replace('```json', '').replace('```', '').strip()
                raw = json.loads(clean_json)
                if isinstance(raw, dict):
                    for k in ('segments', 'subtitles', 'cards', 'dialogue', 'transcript', 'items', 'results'):
                        if k in raw and isinstance(raw[k], list):
                            raw = raw[k]
                            break
                    else:
                        for v in raw.values():
                            if isinstance(v, list):
                                raw = v
                                break
                if not isinstance(raw, list) or not raw:
                    return []
                segments = []
                meta_h = getattr(metadata, 'height', None)
                if meta_h is None and isinstance(metadata, dict):
                    meta_h = metadata.get('height') or next((s.get('height') for s in metadata.get('streams', []) if s.get('height')), None)
                meta_h = int(meta_h or 1080)

                meta_w = getattr(metadata, 'width', None)
                if meta_w is None and isinstance(metadata, dict):
                    meta_w = metadata.get('width') or next((s.get('width') for s in metadata.get('streams', []) if s.get('width')), None)
                meta_w = int(meta_w or 1080)

                meta_dur = getattr(metadata, 'duration', None)
                if meta_dur is None and isinstance(metadata, dict):
                    meta_dur = metadata.get('duration') or metadata.get('format', {}).get('duration')
                meta_dur = float(meta_dur or 9999.0)

                patch_color = getattr(self.config, 'overlay_patch_color', '0x000000')

                for i, item in enumerate(raw, 1):
                    start = max(0.0, float(item.get('start', 0.0)))
                    end = float(item.get('end', start + 1.0))
                    txt = str(item.get('text', '')).strip()
                    if not txt or end <= start:
                        continue
                    if end >= meta_dur - 1.2:
                        end = meta_dur
                    box_2d = item.get('box_2d')
                    if box_2d and len(box_2d) == 4:
                        ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in box_2d]
                        pad_y = max(4, int((ymax - ymin) * 0.12))
                        pad_x = max(8, int((xmax - xmin) * 0.10))
                        ymin = max(0, ymin - pad_y)
                        ymax = min(1000, ymax + pad_y)
                        xmin = max(0, xmin - pad_x)
                        xmax = min(1000, xmax + pad_x)
                        bx = int(xmin * meta_w / 1000)
                        by = int(ymin * meta_h / 1000)
                        bw = int((xmax - xmin) * meta_w / 1000)
                        bh = int((ymax - ymin) * meta_h / 1000)
                    else:
                        bx = int(meta_w * 0.075)
                        bw = int(meta_w * 0.85)
                        by = int(meta_h * 0.881)
                        bh = int(meta_h * 0.080)
                    bbox = BoundingBox(bx, by, bw, bh, patch_color=patch_color)
                    segments.append(SubtitleSegment(i, start, min(end, meta_dur), txt, bbox=bbox, ocr_confidence=1.0))
                return segments
            finally:
                try:
                    self.client.files.delete(name=upload.name)
                except Exception:
                    pass

def extract_burned(video, metadata, config, provider=None):
    if not config.ocr_enabled: raise RuntimeError('No text subtitle stream. Enable OCR_ENABLED and install OCR dependencies.')
    try: import cv2
    except ImportError as exc: raise RuntimeError('Install requirements-ocr.txt for burned-in subtitle OCR.') from exc
    import logging
    target_box = getattr(config, 'ocr_target_box', '').lower()
    # Agent runs first when requested — it produces accurate per-segment bboxes
    if target_box == 'agent':
        try:
            from pipeline.agent_card_detector import AgentCardDetector
            detector = AgentCardDetector(config)
            agent_segs = detector.detect_subtitles(video, metadata)
            if agent_segs:
                return agent_segs
            if config.ocr_provider == 'groq':
                raise RuntimeError('Groq vision found no readable subtitle cards')
        except Exception as exc:
            fallback = 'Groq vision fallback disabled' if config.ocr_provider == 'groq' else 'falling back to Gemini OCR'
            logging.getLogger(__name__).warning('AgentCardDetector failed, %s: %s', fallback, exc)
            if config.ocr_provider == 'groq':
                raise RuntimeError(f'Groq vision card detection failed: {exc}') from exc
    if config.ocr_provider == 'groq':
        from pipeline.agent_card_detector import AgentCardDetector
        segments = AgentCardDetector(config).detect_subtitles(video, metadata)
        if not segments:
            raise RuntimeError('Groq vision found no readable subtitle cards')
        return segments
    if config.ocr_provider == 'gemini':
        try:
            extractor = GeminiVideoSubtitleExtractor(config)
            gemini_segs = extractor.extract_subtitles(video, metadata)
            if gemini_segs:
                return gemini_segs
        except Exception as exc:
            logging.getLogger(__name__).warning('GeminiVideoSubtitleExtractor failed, falling back to Tesseract: %s', exc)
    if provider is None:
        if config.ocr_provider not in ('tesseract', 'gemini', 'groq'): raise ValueError('Unsupported OCR_PROVIDER; implement OCRProvider for another engine')
        provider=TesseractOCRProvider(config.ocr_language)
    region=(0,0,metadata.width,metadata.height)
    if config.ocr_region:
        region=tuple(int(n) for n in config.ocr_region.split(','))
        if len(region)!=4: raise ValueError('OCR_REGION must be x,y,width,height')
        x,y,w,h=region
        if min(x,y)<0 or min(w,h)<=0 or x+w>metadata.width or y+h>metadata.height: raise ValueError('OCR_REGION is outside frame')
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError('OpenCV cannot open video for OCR')
    segments=[]; active=[]; timestamp=0.0
    try:
        while timestamp<metadata.duration:
            capture.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000)
            success,frame=capture.read()
            if not success: raise RuntimeError(f'OCR frame decode failed at {timestamp:.3f}s')

            if target_box == 'beige':
                beige_box = find_beige_box(frame)
                if beige_box is None:
                    active = []
                    timestamp += config.ocr_interval
                    continue
                pad_y = max(2, int(beige_box.height * 0.05))
                pad_x = max(2, int(beige_box.width * 0.03))
                crop = frame[beige_box.y + pad_y : beige_box.y + beige_box.height - pad_y,
                             beige_box.x + pad_x : beige_box.x + beige_box.width - pad_x]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                detections = provider.detect(crop_rgb, psm=6)
                if not detections:
                    detections = provider.detect(crop_rgb, psm=11)
                if detections:
                    text = ' '.join(d.text for d in detections)
                    confidence = sum(d.confidence for d in detections) / len(detections)
                    detections = [OCRDetection(text, beige_box, confidence)]
            else:
                x,y,w,h=region
                detections=provider.detect(cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB))

            next_active=[]; matched=set()
            for detection in detections:
                box=detection.bbox
                if target_box != 'beige':
                    box.x+=x; box.y+=y
                text=normalize(detection.text)
                if not text or sum(c.isalnum() for c in text) < 4: continue
                if config.ocr_solid_bg_only and not target_box:
                    if not _has_light_background(frame, box, config.ocr_bg_variance_threshold): continue

                def is_match(s, t, b1, b2):
                    s_l, t_l = s.casefold(), t.casefold()
                    sim = (SequenceMatcher(None, s_l, t_l).ratio() > 0.60 or s_l in t_l or t_l in s_l)
                    pos = (abs(b1.x - b2.x) < max(50, b2.width * 0.35) and abs(b1.y - b2.y) < max(40, b2.height * 0.5))
                    return sim and pos

                previous=next((s for s in active if s.id not in matched and is_match(s.source_text, text, s.bbox, box)), None)
                if previous:
                    previous.end=min(timestamp+config.ocr_interval,metadata.duration)
                    # Retain the union of observed positions for this subtitle. A
                    # moving card then gets one conservative mask spanning its path.
                    old_box = previous.bbox
                    left = min(old_box.x, box.x)
                    top = min(old_box.y, box.y)
                    right = max(old_box.x + old_box.width, box.x + box.width)
                    bottom = max(old_box.y + old_box.height, box.y + box.height)
                    previous.bbox = BoundingBox(left, top, right - left, bottom - top,
                                                patch_color=getattr(old_box, 'patch_color', None) or getattr(box, 'patch_color', None))
                    if sum(c.isalnum() for c in text) > sum(c.isalnum() for c in previous.source_text):
                        previous.source_text = text
                    if getattr(box, 'patch_color', None) and not getattr(previous.bbox, 'patch_color', None):
                        previous.bbox.patch_color = box.patch_color
                    matched.add(previous.id); next_active.append(previous)
                else:
                    segment=SubtitleSegment(len(segments)+1,timestamp,min(timestamp+config.ocr_interval,metadata.duration),text,bbox=box,ocr_confidence=detection.confidence)
                    segments.append(segment); next_active.append(segment)
            active=next_active
            timestamp+=config.ocr_interval
    finally: capture.release()

    # Post-process: bridge small gaps and merge consecutive similar cards so patch never shuts off prematurely
    merged = []
    for s in segments:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        gap = s.start - prev.end
        sim = SequenceMatcher(None, prev.source_text.casefold(), s.source_text.casefold()).ratio()
        is_same = sim > 0.60 or prev.source_text.casefold() in s.source_text.casefold() or s.source_text.casefold() in prev.source_text.casefold()
        if is_same and gap <= 2.0:
            prev.end = max(prev.end, s.end)
            if prev.bbox and s.bbox:
                left = min(prev.bbox.x, s.bbox.x)
                top = min(prev.bbox.y, s.bbox.y)
                right = max(prev.bbox.x + prev.bbox.width, s.bbox.x + s.bbox.width)
                bottom = max(prev.bbox.y + prev.bbox.height, s.bbox.y + s.bbox.height)
                prev.bbox = BoundingBox(left, top, right - left, bottom - top,
                                        patch_color=prev.bbox.patch_color or s.bbox.patch_color)
            if sum(c.isalnum() for c in s.source_text) > sum(c.isalnum() for c in prev.source_text):
                prev.source_text = s.source_text
        elif 0 <= gap <= 1.5:
            # Seamless transition: extend previous card patch up to start of next card
            prev.end = s.start
            merged.append(s)
        else:
            merged.append(s)
    segments = merged

    min_dur = getattr(config, 'min_subtitle_duration', 0.5)
    filtered = [s for s in segments if (s.end - s.start) >= min_dur or sum(c.isalnum() for c in s.source_text) >= 6]
    if filtered:
        segments = filtered
        for i, s in enumerate(segments, 1): s.id = i
    if not segments: raise RuntimeError('OCR found no subtitles. Check OCR_LANGUAGE, OCR_REGION, and video quality.')
    return segments

class SubtitleRemoval(ABC):
    @abstractmethod
    def remove_original_subtitle(self, bbox, time_range): pass

class OverlayRemoval(SubtitleRemoval):
    def __init__(self, default_color='0x000000'):
        self.default_color = default_color

    def remove_original_subtitle(self, bbox, time_range, color=None):
        start,end=time_range
        patch_color = color or getattr(bbox, 'patch_color', None) or self.default_color
        return f"drawbox=x={bbox.x}:y={bbox.y}:w={bbox.width}:h={bbox.height}:color={patch_color}:t=fill:enable='between(t,{start:.6f},{end:.6f})'"

class InpaintingRemoval(SubtitleRemoval):
    def remove_original_subtitle(self, bbox, time_range):
        # Video inpainting is performed frame-by-frame by subtitle_renderer, where
        # the timed mask and optional tracked coordinates are available.
        return None

class DelogoRemoval(SubtitleRemoval):
    def remove_original_subtitle(self, bbox, time_range):
        start, end = time_range
        return f"delogo=x={bbox.x}:y={bbox.y}:w={bbox.width}:h={bbox.height}:enable='between(t,{start:.6f},{end:.6f})'"
