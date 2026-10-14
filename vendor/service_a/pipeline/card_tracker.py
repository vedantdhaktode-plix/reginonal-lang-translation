"""Frame-accurate tracking of solid dark subtitle cards.

Gemini timestamps are sampled at about 1 fps and its boxes are coarse, so cards that
change every 1-2 s end up masked at the wrong time or place. This module scans every
decoded frame locally, finds the dark caption card with OpenCV, and splits the timeline
wherever the card's text pixels change. Gemini (or Tesseract) then only supplies text.
"""
import logging
import math
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from models.subtitle import BoundingBox, SubtitleSegment

logger = logging.getLogger(__name__)

DETECT_WIDTH = 540          # detection runs on a downscaled copy of each frame
SIG_SIZE = (96, 24)         # text-pixel signature resolution (w, h)
SIG_CHANGE = 0.12           # fraction of differing signature pixels that means "new card"
MERGE_GAP = 0.30            # same card missing for <= this many seconds is one interval
MIN_INTERVAL = 0.20         # shorter intervals (pop-in animation frames) join the next card


@dataclass
class CardInterval:
    start: float
    end: float
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in source pixels
    signature: object
    colors: list = field(default_factory=list)
    crop: object = None
    ocr_crop: object = None
    frames: int = 0
    positions: list = field(default_factory=list)  # (time, x, y, width, height)

    def bbox(self) -> BoundingBox:
        x1, y1, x2, y2 = self.box
        b, g, r = (int(c) for c in _median_color(self.colors))
        return BoundingBox(x1, y1, x2 - x1, y2 - y1, patch_color=f'0x{r:02X}{g:02X}{b:02X}')


def _median_color(colors):
    import numpy as np
    return np.median(np.array(colors), axis=0) if colors else (17, 17, 17)


def detect_dark_cards(small, max_luma=None):
    """Returns [(x, y, w, h, bgr)] of dark, caption-shaped rectangles that contain bright text."""
    import cv2
    import numpy as np
    max_luma = max_luma or int(os.getenv('CARD_MAX_LUMA', '60'))
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark = gray < max_luma
    # Cards are found from their text: dark-box contours merge with dark scenery
    # (hair, TV frames, split-screen seams), but bright text lines stay separate.
    bright = np.where(gray > 185, 255, 0).astype(np.uint8)
    # Keep only letter-sized bright pieces with dark directly around them, so caption
    # text cannot fuse with bright scenery (skin, walls) that touches it.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return []
    integ = cv2.integral(dark.astype(np.uint8))
    sx, sy, sw, sh, area = (stats[1:, i] for i in range(5))
    r = 2
    ox1, oy1 = np.clip(sx - r, 0, w), np.clip(sy - r, 0, h)
    ox2, oy2 = np.clip(sx + sw + r, 0, w), np.clip(sy + sh + r, 0, h)
    outer = integ[oy2, ox2] - integ[oy1, ox2] - integ[oy2, ox1] + integ[oy1, ox1]
    inner = integ[sy + sh, sx + sw] - integ[sy, sx + sw] - integ[sy + sh, sx] + integ[sy, sx]
    ring_px = (ox2 - ox1) * (oy2 - oy1) - sw * sh
    ring_dark = (outer - inner) / np.maximum(ring_px, 1)
    letter = (sh <= h * 0.06) & (sw <= w * 0.25) & (area >= 4) & (ring_dark >= 0.6)
    keep = np.concatenate([[False], letter])
    bright = np.where(keep[labels], 255, 0).astype(np.uint8)
    lines = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(w * 0.03)), 3)))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, tw, th = cv2.boundingRect(cnt)
        if not (h * 0.010 <= th <= h * 0.08 and tw >= w * 0.06 and tw / float(th) >= 1.5):
            continue
        # A caption line is wrapped in a dark margin; packaging or wall text is not.
        p = max(2, int(th * 0.25))
        x1, y1, x2, y2 = max(0, x - p), max(0, y - p), min(w, x + tw + p), min(h, y + th + p)
        ring = np.ones((y2 - y1, x2 - x1), bool)
        ring[y - y1:y - y1 + th, x - x1:x - x1 + tw] = False
        if ring.sum() < 10 or float(dark[y1:y2, x1:x2][ring].mean()) < 0.75:
            continue
        # Grow outward to the card edges while the next row/column is still mostly dark.
        limit_y, limit_x = max(3, int(th * 0.8)), max(4, int(th * 2.0))
        for _ in range(limit_y):
            if y1 > 0 and dark[y1 - 1, x1:x2].mean() >= 0.85: y1 -= 1
            else: break
        for _ in range(limit_y):
            if y2 < h and dark[y2, x1:x2].mean() >= 0.85: y2 += 1
            else: break
        for _ in range(limit_x):
            if x1 > 0 and dark[y1:y2, x1 - 1].mean() >= 0.85: x1 -= 1
            else: break
        for _ in range(limit_x):
            if x2 < w and dark[y1:y2, x2].mean() >= 0.85: x2 += 1
            else: break
        boxes.append([x1, y1, x2, y2])

    card_or_text = dark | (gray > 150)

    def same_card(a, b):
        if a[0] <= b[2] + 2 and b[0] <= a[2] + 2 and a[1] <= b[3] + 2 and b[1] <= a[3] + 2:
            return True  # touching or overlapping
        # Words on one row separated by a wide space: merge if the gap is card colour.
        overlap_y = min(a[3], b[3]) - max(a[1], b[1])
        if overlap_y < 0.7 * min(a[3] - a[1], b[3] - b[1]):
            return False
        gx1, gx2 = min(a[2], b[2]), max(a[0], b[0])
        if gx2 - gx1 > 1.5 * (a[3] - a[1]):
            return False
        gap = card_or_text[max(a[1], b[1]):min(a[3], b[3]), gx1:gx2]
        return gap.size > 0 and float(gap.mean()) >= 0.85

    # Words or lines of one card (separate text blobs) are merged into one box.
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if same_card(a, b):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break

    def card_column(x, y1, y2):
        # Inside a card a column has card colour in BOTH the top and bottom margins;
        # its middle may be anything (letters, or a solid anti-aliased letter edge).
        # Scenery on a split-screen seam (dark above, bright below) and walls fail.
        m = max(1, (y2 - y1) // 5)
        return float(dark[y1:y1 + m, x].mean()) >= 0.6 and float(dark[y2 - m:y2, x].mean()) >= 0.6

    def text_column(x, y1, y2):
        return card_column(x, y1, y2) and float((gray[y1:y2, x] > 150).mean()) >= 0.15

    def extend(x, step, y1, y2, limit):
        # Walks across text columns, tolerating the dark spacing between letters/words.
        edge, gap = x, 0
        probe = x + step
        for _ in range(limit):
            if not 0 <= probe < w:
                break
            if text_column(probe, y1, y2):
                edge, gap = probe, 0
            elif card_column(probe, y1, y2) and gap < 6:
                gap += 1
            else:
                break
            probe += step
        return edge

    cards = []
    for x1, y1, x2, y2 in boxes:
        # End words whose letters touch the card edge fail the letter test above;
        # keep growing sideways across their text so they are masked too.
        limit = 4 * (y2 - y1)
        x1 = min(x1, extend(x1, -1, y1, y2, limit))
        x2 = max(x2, extend(x2 - 1, 1, y1, y2, limit) + 1)
        # Then take in the card's plain margin beyond the outermost letters.
        for _ in range(max(2, (y2 - y1) // 2)):
            if x1 > 0 and card_column(x1 - 1, y1, y2): x1 -= 1
            else: break
        for _ in range(max(2, (y2 - y1) // 2)):
            if x2 < w and card_column(x2, y1, y2): x2 += 1
            else: break
        bw, bh = x2 - x1, y2 - y1
        if not (h * 0.018 <= bh <= h * 0.14 and bw >= w * 0.08 and bw / float(bh) >= 1.8):
            continue
        region = gray[y1:y2, x1:x2]
        # A solid card is almost entirely card colour or text colour.
        if float(card_or_text[y1:y2, x1:x2].mean()) < 0.80:
            continue
        bgr = small[y1:y2, x1:x2][region < max_luma].mean(axis=0)
        cards.append((x1, y1, bw, bh, bgr))
    return cards


def detect_caption_cards(small):
    """Detect both bright-on-dark and dark-on-light caption panels."""
    import cv2
    import numpy as np

    cards = list(detect_dark_cards(small))
    # The existing detector is contrast-based. Inverting the frame lets the same
    # conservative geometry find black text on white/light cards without a second
    # unrelated detection algorithm.
    inverted = 255 - small
    light_max_luma = int(os.getenv('CARD_LIGHT_MAX_INVERTED_LUMA', '55'))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    light_mask = np.where(gray > 235, 255, 0).astype(np.uint8)
    light_mask = cv2.morphologyEx(
        light_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 15)))
    _, _, light_stats, _ = cv2.connectedComponentsWithStats(light_mask, connectivity=8)

    for x, y, w, h, inverted_bgr in detect_dark_cards(inverted, max_luma=light_max_luma):
        # Expand a text-level detection to the large, uniform light panel around
        # it, then inset slightly so a rectangular overlay stays inside rounded
        # card corners. This gives text a correctly centered full-width canvas.
        best_panel = None
        best_overlap = 0
        for px, py, pw, ph, area in light_stats[1:]:
            px, py, pw, ph, area = map(int, (px, py, pw, ph, area))
            if pw < w or ph < h or pw > small.shape[1] * 0.95 or ph > small.shape[0] * 0.25:
                continue
            ix = max(0, min(x + w, px + pw) - max(x, px))
            iy = max(0, min(y + h, py + ph) - max(y, py))
            overlap = ix * iy
            if overlap > best_overlap and overlap >= w * h * 0.45:
                best_panel, best_overlap = (px, py, pw, ph), overlap
        if best_panel is not None:
            px, py, pw, ph = best_panel
            inset_x = max(3, int(ph * 0.08))
            inset_y = max(3, int(ph * 0.08))
            x, y = px + inset_x, py + inset_y
            w, h = pw - 2 * inset_x, ph - 2 * inset_y

        original_bgr = np.array([255 - int(c) for c in inverted_bgr], dtype=float)
        candidate = (x, y, w, h, original_bgr)

        def overlap_ratio(existing):
            ex, ey, ew, eh, _ = existing
            ix = max(0, min(x + w, ex + ew) - max(x, ex))
            iy = max(0, min(y + h, ey + eh) - max(y, ey))
            intersection = ix * iy
            return intersection / max(1, min(w * h, ew * eh))

        if not any(overlap_ratio(existing) >= 0.65 for existing in cards):
            cards.append(candidate)
    return cards


def _signature(gray_crop):
    import cv2
    import numpy as np
    text = gray_crop > 170
    # Crop to the text itself so a box that wobbles by a pixel or two at the card
    # margin does not change the signature and split one card into two intervals.
    rows, cols = np.where(text.any(axis=1))[0], np.where(text.any(axis=0))[0]
    if rows.size and cols.size:
        text = text[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return cv2.resize(text.astype('uint8') * 255, SIG_SIZE, interpolation=cv2.INTER_AREA) > 127


def _sig_diff(a, b):
    return float((a != b).mean())


def track_cards(video, metadata):
    """Scans every frame and returns CardInterval objects with frame-accurate timing."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('OpenCV cannot open video for card tracking')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_dur = 1.0 / fps
    intervals: list[CardInterval] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index * frame_dur
            index += 1
            fh, fw = frame.shape[:2]
            scale = DETECT_WIDTH / float(fw)
            small = cv2.resize(frame, (DETECT_WIDTH, max(1, int(fh * scale))), interpolation=cv2.INTER_AREA)
            # Several cards can be visible at once (crossfades, stacked captions), so each
            # detection continues its own track rather than only the largest card.
            claimed = set()
            for x, y, bw, bh, bgr in detect_caption_cards(small):
                box = (int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale))
                sig = _signature(cv2.cvtColor(small[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY))
                cy, ch = (box[1] + box[3]) / 2, box[3] - box[1]
                current = next((iv for iv in reversed(intervals)
                                if id(iv) not in claimed and t - iv.end <= MERGE_GAP
                                and _sig_diff(sig, iv.signature) <= SIG_CHANGE
                                and abs(cy - (iv.box[1] + iv.box[3]) / 2) <= ch * 0.6), None)
                if current is not None:
                    current.end = t + frame_dur
                    current.box = box
                    current.signature = sig  # follow slow fades/animation
                else:
                    current = CardInterval(t, t + frame_dur, box, sig)
                    intervals.append(current)
                claimed.add(id(current))
                current.frames += 1
                current.positions.append((t, box[0], box[1], box[2] - box[0], box[3] - box[1]))
                current.colors.append(bgr)
                if current.frames <= 5:  # keep a crop taken after any pop-in animation settles
                    pad_y = max(4, int((box[3] - box[1]) * 0.25))
                    pad_x = max(6, int((box[2] - box[0]) * 0.08))
                    c_y1 = max(0, box[1] - pad_y)
                    c_y2 = min(fh, box[3] + pad_y)
                    c_x1 = max(0, box[0] - pad_x)
                    c_x2 = min(fw, box[2] + pad_x)
                    # Extract the full horizontal row across the frame width so vision OCR
                    # sees the complete subtitle line, not just the part over dark pixels.
                    current.crop = frame[c_y1:c_y2, 0:fw].copy()
                    # Tesseract is substantially more reliable on a tight crop. Keep
                    # this separately so vision providers can still use the contextual row.
                    current.ocr_crop = frame[c_y1:c_y2, c_x1:c_x2].copy()
    finally:
        cap.release()

    # Pop-in/out animation frames form tiny intervals; fold each into the card that
    # follows (or precedes) it in the same screen area.
    def near(a, b):
        return abs((a.box[1] + a.box[3]) / 2 - (b.box[1] + b.box[3]) / 2) <= (a.box[3] - a.box[1]) * 0.6

    long_ivs = [iv for iv in intervals if iv.end - iv.start >= MIN_INTERVAL]
    for short in (iv for iv in intervals if iv.end - iv.start < MIN_INTERVAL):
        nxt = min((iv for iv in long_ivs if near(short, iv) and 0 <= iv.start - short.end <= MERGE_GAP),
                  key=lambda iv: iv.start, default=None)
        if nxt is not None:
            nxt.start = min(nxt.start, short.start)
            nxt.box = _union(nxt.box, short.box)
            continue
        prev = max((iv for iv in long_ivs if near(short, iv) and 0 <= short.start - iv.end <= MERGE_GAP),
                   key=lambda iv: iv.end, default=None)
        if prev is not None:
            prev.end = max(prev.end, short.end)
            prev.box = _union(prev.box, short.box)

    # A card whose detection briefly split or flickered can leave two overlapping
    # intervals in the same place; keep them as one.
    # A short crossfade (a few frames) between consecutive cards is not a duplicate.
    def is_twin(r, iv):
        ov = min(r.end, iv.end) - max(r.start, iv.start)
        shorter = min(r.end - r.start, iv.end - iv.start)
        return (near(r, iv) and ov >= 0.15 and ov >= 0.3 * shorter
                and min(r.box[2], iv.box[2]) > max(r.box[0], iv.box[0]))

    result: list[CardInterval] = []
    for iv in sorted(long_ivs, key=lambda iv: iv.start):
        twin = next((r for r in result if is_twin(r, iv)), None)
        if twin is not None:
            twin.end = max(twin.end, iv.end)
            twin.box = _union(twin.box, iv.box)
            twin.frames += iv.frames
        else:
            result.append(iv)
    return result


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _ocr_crop(crop, language):
    try:
        import cv2
        import pytesseract
    except ImportError:
        return ''
    if crop is None or crop.size == 0:
        return ''
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(binary, lang=language, config='--psm 6', timeout=15)
    except Exception as exc:
        logger.debug('Card OCR failed: %s', exc)
        return ''
    return ' '.join(text.split())


def _sim(a, b):
    a, b = re.sub(r'\W+', ' ', a.casefold()).strip(), re.sub(r'\W+', ' ', b.casefold()).strip()
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def _overlap(s1, e1, s2, e2):
    return max(0.0, min(e1, e2) - max(s1, s2))


def _split_words(text, weights):
    words = text.split()
    if len(weights) <= 1 or len(words) < len(weights):
        return [text] + [''] * (len(weights) - 1)
    total = sum(weights)
    parts, pos = [], 0
    for i, wgt in enumerate(weights):
        remaining_slots = len(weights) - i - 1
        if i == len(weights) - 1:
            take = len(words) - pos
        else:
            take = max(1, round(len(words) * wgt / total))
            take = min(take, len(words) - pos - remaining_slots)
        parts.append(' '.join(words[pos:pos + take]))
        pos += take
    return parts


def build_segments(intervals, gemini_cards, config, duration):
    """Attaches text to tracked intervals. Returns (segments, coverage of Gemini dialogue time)."""
    language = getattr(config, 'ocr_language', 'eng') or 'eng'
    # A crossfade can split one card into two back-to-back intervals in the same
    # place; when OCR reads the same text on both, they are one card.
    read = []
    for iv in intervals:
        ocr_source = iv.ocr_crop if iv.ocr_crop is not None else iv.crop
        ocr = _ocr_crop(ocr_source, language)
        prev = read[-1] if read else None
        if (prev is not None and sum(ch.isalnum() for ch in ocr) >= 3 and _sim(ocr, prev[1]) >= 0.9
                and 0 <= iv.start - prev[0].end <= MERGE_GAP + 0.05
                and abs((iv.box[1] + iv.box[3]) / 2 - (prev[0].box[1] + prev[0].box[3]) / 2) <= (iv.box[3] - iv.box[1]) * 0.6):
            prev[0].end = max(prev[0].end, iv.end)
            prev[0].box = _union(prev[0].box, iv.box)
            prev[0].colors.extend(iv.colors)
            continue
        read.append((iv, ocr))

    # (interval, text, matched Gemini index or None). A Gemini index is retained
    # only when local OCR confirms that the tracked geometry contains that text.
    # Temporal overlap alone is insufficient: two panels in one card commonly
    # remain visible for the same duration.
    assigned = []
    for iv, ocr in read:
        ocr_ok = sum(ch.isalnum() for ch in ocr) >= 3
        best, best_score = None, -999.0
        iv_center = (iv.start + iv.end) / 2.0
        iv_dur = max(0.01, iv.end - iv.start)

        for gi, card in enumerate(gemini_cards):
            c_start, c_end = card['start'], card['end']
            c_center = (c_start + c_end) / 2.0
            c_dur = max(0.01, c_end - c_start)
            ov = _overlap(iv.start, iv.end, c_start, c_end)

            is_indic = any(ord(c) > 0x0590 for c in card['text'])
            sim_score = (3.0 * _sim(ocr, card['text'])) if (ocr_ok and not is_indic) else 0.0

            if ov > 0:
                score = (ov / iv_dur) * 2.0 - abs(iv_center - c_center) / c_dur + sim_score
            else:
                dist = max(0.0, max(c_start - iv.end, iv.start - c_end))
                if dist <= 1.2:
                    score = 0.5 - (dist / 1.2) + sim_score
                else:
                    score = -999.0

            if score > best_score:
                best, best_score = gi, score

        if best_score <= -500:
            best = None

        gem_text = gemini_cards[best]['text'] if best is not None else ''
        same_length = 0.8 <= len(gem_text) / max(1, len(ocr)) <= 1.25
        is_indic = any(ord(c) > 0x0590 for c in gem_text) if gem_text else False

        ocr_is_indic = any(ord(c) > 0x0590 for c in ocr) if ocr else False
        punctuation_ratio = (sum(not ch.isalnum() and not ch.isspace() for ch in ocr)
                             / max(1, len(ocr)))
        confirmed = (ocr_ok and best is not None and same_length
                     and ocr_is_indic == is_indic and _sim(ocr, gem_text) >= 0.8)

        if confirmed:
            assigned.append((iv, gem_text, best))
        elif ocr_ok and punctuation_ratio < 0.35:
            # This is real text in a different simultaneous panel. Keep it as a
            # local segment, but do not let it claim coverage of the Gemini card.
            assigned.append((iv, ocr, None))
            if best is not None:
                logger.info('Tracked card OCR %r did not match target text %r; geometry will be grounded from keyframes',
                            ocr, gem_text)
        elif best is not None:
            # Retain a provisional segment for diagnostics. Because the match is
            # unconfirmed it deliberately contributes no coverage, forcing the
            # caller to use targeted Gemini keyframe geometry.
            assigned.append((iv, gem_text, None))
        else:
            logger.info('Dropping tracked card %.2f-%.2fs: no matching dialogue text', iv.start, iv.end)
            assigned.append((iv, '', None))

    # Without OCR, several tracked cards may map to one merged Gemini line; split its words.
    groups = {}
    for idx, (_, text, gi) in enumerate(assigned):
        if text is None:
            groups.setdefault(gi, []).append(idx)
    for gi, idxs in groups.items():
        parts = _split_words(gemini_cards[gi]['text'], [assigned[i][0].end - assigned[i][0].start for i in idxs])
        for i, part in zip(idxs, parts):
            assigned[i] = (assigned[i][0], part, gi)

    segments = []
    for iv, text, gi in assigned:
        if not text or not text.strip():
            continue
        # Round outward to whole milliseconds; rounding to nearest can start the
        # overlay one frame after the card appears.
        start = math.floor(iv.start * 1000) / 1000
        end = math.ceil(min(iv.end, duration) * 1000) / 1000
        segments.append(SubtitleSegment(len(segments) + 1, start, end,
                                        text.strip(), bbox=iv.bbox(), ocr_confidence=1.0,
                                        bbox_track=iv.positions or None,
                                        event_index=gi))

    # Outward rounding can overlap back-to-back cards by a millisecond or so, which
    # shows two cues at once in the SRT; the earlier card's last frame ends before.
    for prev, nxt in zip(segments, segments[1:]):
        if 0 < prev.end - nxt.start <= 0.02:
            prev.end = nxt.start

    gemini_total = sum(max(0.0, c['end'] - c['start']) for c in gemini_cards)
    confirmed_intervals = {}
    for iv, text, gi in assigned:
        if text and gi is not None:
            confirmed_intervals.setdefault(gi, []).append(iv)
    covered = sum(
        min(max(0.0, card['end'] - card['start']),
            sum(_overlap(iv.start, iv.end, card['start'], card['end'])
                for iv in confirmed_intervals.get(gi, [])))
        for gi, card in enumerate(gemini_cards)
    )
    coverage = covered / gemini_total if gemini_total else (1.0 if segments else 0.0)
    return segments, coverage
