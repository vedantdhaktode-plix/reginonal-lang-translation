import shutil
from models.subtitle import BoundingBox
from pipeline.ffmpeg_utils import run
from pipeline.burned_subtitle_ocr import OverlayRemoval, InpaintingRemoval, DelogoRemoval


def detect_line_regions(video, segments, config=None):
    """Find separate text-line bands using OCR geometry, then image analysis."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return segments
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return segments
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for seg in segments:
            box = seg.bbox
            if not box or box.width < 30 or box.height < 10:
                continue
            sample_t = max(seg.start, min(seg.end - 0.03, (seg.start + seg.end) / 2))
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_t * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            x1, y1 = max(0, box.x), max(0, box.y)
            x2, y2 = min(width, box.x + box.width), min(height, box.y + box.height)
            roi = frame[y1:y2, x1:x2]
            rh, rw = roi.shape[:2]
            if rh < 10 or rw < 30:
                continue

            rects = []
            # Word boxes from Tesseract are resilient to textured or non-uniform
            # caption backgrounds that defeat a single background-color threshold.
            try:
                import pytesseract
                from pytesseract import Output
                data = pytesseract.image_to_data(roi, lang=getattr(config, 'ocr_language', 'eng'),
                                                  config='--psm 6', output_type=Output.DICT)
                grouped = {}
                for i, word in enumerate(data.get('text', [])):
                    if not str(word).strip():
                        continue
                    try:
                        if float(data['conf'][i]) < 15:
                            continue
                    except (ValueError, TypeError):
                        continue
                    key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                    x, y = int(data['left'][i]), int(data['top'][i])
                    w, h = int(data['width'][i]), int(data['height'][i])
                    if w > 0 and h > 0:
                        grouped.setdefault(key, []).append((x, y, w, h))
                for words in grouped.values():
                    left = min(item[0] for item in words)
                    top = min(item[1] for item in words)
                    right = max(item[0] + item[2] for item in words)
                    bottom = max(item[1] + item[3] for item in words)
                    if right - left >= 4 and bottom - top >= 3:
                        rects.append((left, top, right, bottom))
            except Exception:
                pass

            # Adaptive threshold fallback catches stylized text where OCR provides
            # no word boxes. Row groups remain separate so inter-line gaps survive.
            if not rects:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                block = max(15, min(51, (min(rh, rw) // 3) | 1))
                for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        threshold_type, block, 9)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                    active = (binary > 0).sum(axis=1) >= max(3, int(rw * 0.006))
                    rows = np.flatnonzero(active)
                    runs = []
                    if len(rows):
                        start = previous = int(rows[0])
                        for row in rows[1:]:
                            row = int(row)
                            if row > previous + 3:
                                runs.append((start, previous + 1))
                                start = row
                            previous = row
                        runs.append((start, previous + 1))
                    candidates = []
                    for top, bottom in runs:
                        if bottom - top < max(3, int(rh * 0.025)) or bottom - top > rh * 0.65:
                            continue
                        cols = np.flatnonzero((binary[top:bottom] > 0).sum(axis=0) >= 2)
                        if len(cols) >= max(4, int(rw * 0.02)):
                            candidates.append((int(cols[0]), top, int(cols[-1]) + 1, bottom))
                    if candidates and len(candidates) <= 4:
                        rects = candidates
                        break

            # Last fallback uses OCR-preserved newlines when available; for a
            # single line, mask only a narrow centered band, never the whole tall card.
            if not rects:
                text_lines = [line.strip() for line in (seg.source_text or '').splitlines() if line.strip()]
                if len(text_lines) > 1:
                    band_h = max(4, int(rh / len(text_lines) * 0.68))
                    for index in range(len(text_lines)):
                        center = (index + 0.5) * rh / len(text_lines)
                        rects.append((0, max(0, round(center - band_h / 2)), rw,
                                      min(rh, round(center + band_h / 2))))
                else:
                    band_h = max(6, min(int(rh * 0.48), round(rw / max(8, len(seg.source_text or '')) * 1.15)))
                    top = max(0, (rh - band_h) // 2)
                    rects = [(0, top, rw, min(rh, top + band_h))]

            regions = []
            for left, top, right, bottom in sorted(rects, key=lambda r:r[1]):
                line_h = max(1, bottom - top)
                pad_x, pad_y = max(2, round((right-left)*0.025)), max(1, round(line_h*0.12))
                left, right = max(0,left-pad_x), min(rw,right+pad_x)
                top, bottom = max(0,top-pad_y), min(rh,bottom+pad_y)
                regions.append(((x1 + left - box.x) / box.width,
                                (y1 + top - box.y) / box.height,
                                (right-left) / box.width,
                                (bottom-top) / box.height))
            if regions:
                seg.line_regions = regions
    finally:
        capture.release()
    return segments


def _inpaint_source(video, segments, config, folder):
    """Remove each timed subtitle mask from decoded frames with OpenCV inpainting."""
    import cv2
    import numpy as np
    from pipeline.ffmpeg_utils import run

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('OpenCV cannot open video for inpainting')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = folder / 'inpainted_silent.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError('OpenCV could not create inpainted video')
    index = 0
    tracks = [(s, sorted(s.bbox_track or [], key=lambda p: p[0])) for s in segments if s.bbox]

    def position(seg, track, t):
        if not track:
            b = seg.bbox
            return b.x, b.y, b.width, b.height
        if t <= track[0][0]: return tuple(track[0][1:])
        if t >= track[-1][0]: return tuple(track[-1][1:])
        for a, b in zip(track, track[1:]):
            if a[0] <= t <= b[0]:
                f = (t - a[0]) / max(1e-6, b[0] - a[0])
                return tuple(round(a[i] + f * (b[i] - a[i])) for i in range(1, 5))

    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            t = index / fps
            index += 1
            mask = np.zeros((height, width), dtype=np.uint8)
            for seg, track in tracks:
                if not (seg.start <= t < seg.end): continue
                x, y, w, h = position(seg, track, t)
                regions = seg.line_regions or [(0.0, 0.0, 1.0, 1.0)]
                for rx, ry, rw, rh in regions:
                    x1, y1 = max(0, round(x + rx * w)), max(0, round(y + ry * h))
                    x2, y2 = min(width, round(x + (rx + rw) * w)), min(height, round(y + (ry + rh) * h))
                    if x2 > x1 and y2 > y1: cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            if mask.any():
                frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    cleaned = folder / 'inpainted.mp4'
    run([config.ffmpeg, '-nostdin', '-y', '-v', 'warning', '-i', str(raw), '-i', str(video),
         '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
         '-c:a', 'copy', '-shortest', str(cleaned)], config.command_timeout, cwd=folder)
    raw.unlink(missing_ok=True)
    return cleaned

def render_pill_overlays(video, segments, config, folder):
    from PIL import Image, ImageDraw, ImageFont
    from pipeline.subtitle_style import choose_font, _LANGUAGE_FONTS
    
    font_file = None
    target_lang = getattr(config, 'target_language', '').lower()
    pref_stem = _LANGUAGE_FONTS.get(target_lang, '').lower()
    candidates = [*config.font_dir.glob('*.ttf'), *config.font_dir.glob('*.otf')]
    if pref_stem:
        for c in candidates:
            if c.stem.lower() == pref_stem:
                font_file = str(c)
                break
    if not font_file and target_lang:
        for c in candidates:
            if target_lang in c.stem.lower():
                font_file = str(c)
                break
    if not font_file:
        font_name = choose_font(config)
        for c in candidates:
            if c.stem.casefold() == font_name.casefold() or font_name.casefold() in c.stem.casefold():
                font_file = str(c)
                break
    if not font_file and candidates:
        font_file = str(candidates[0])

    inputs = ['-i', str(video)]
    filter_parts = []
    last_v = '0:v'
    card_idx = 0

    def track_expression(track, axis, fallback, start, end, mode='linear'):
        if not track or len(track) < 2:
            return str(fallback)
        samples = sorted(track, key=lambda p: p[0])
        values = [(max(start, min(end, float(p[0]))), int(p[1 if axis == 'x' else 2])) for p in samples]
        compact = []
        for t, value in values:
            if compact and t <= compact[-1][0]:
                compact[-1] = (t, value)
            elif not compact or value != compact[-1][1]:
                compact.append((t, value))
        if len(compact) < 2:
            return str(compact[0][1] if compact else fallback)
        expr = str(compact[-1][1])
        for (t0, v0), (t1, v1) in reversed(list(zip(compact, compact[1:]))):
            if t1 <= t0:
                continue
            if mode == 'step':
                # A step sample confirms the new location at t1. Switching at
                # the midpoint can move the mask into the next shot hundreds of
                # milliseconds early and expose the source caption in this shot.
                boundary = t1
                expr = f"if(lt(t,{boundary:.6f}),{v0},{expr})"
            else:
                linear = f'({v0}+({v1}-{v0})*(t-{t0:.6f})/{t1-t0:.6f})'
                expr = f"if(lt(t,{t1:.6f}),{linear},{expr})"
        return expr

    for seg in segments:
        if not seg.bbox or not seg.translated_text:
            continue
        card_idx += 1
        box = seg.bbox
        bw, bh = box.width, box.height
        bx, by = box.x, box.y

        patch_hex = box.patch_color or getattr(config, 'overlay_patch_color', '0x000000')
        clean = patch_hex.lstrip('#').replace('0x', '')
        if len(clean) == 6:
            r, g, b = int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)
        else:
            r, g, b = 0, 0, 0
        fill_color = (r, g, b, 255)

        txt_hex = getattr(config, 'box_text_color', '#FFFFFF')
        clean_t = txt_hex.lstrip('#').replace('0x', '')
        if len(clean_t) == 6:
            tr, tg, tb = int(clean_t[0:2], 16), int(clean_t[2:4], 16), int(clean_t[4:6], 16)
        else:
            tr, tg, tb = 255, 255, 255
        # Keep text readable when the configured colour matches the detected card colour.
        luma = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if abs(luma((tr, tg, tb)) - luma((r, g, b))) < 90:
            tr, tg, tb = (255, 255, 255) if luma((r, g, b)) < 128 else (0, 0, 0)
        text_color = (tr, tg, tb, 255)

        img = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # An opaque rectangle covers every source pixel in the detected box,
        # including the corners where a rounded transparent overlay could leak text.
        d.rectangle([0, 0, bw, bh], fill=fill_color)

        # Auto-fit font size to pill dimensions
        f_size = int(getattr(seg.style, 'font_size', 0) or getattr(config, 'font_size', 0) or 46)
        text = (seg.translated_text or '').strip().replace('\\N', '\n')
        
        max_allowed_w = bw * 0.92
        max_allowed_h = bh * 0.82
        
        # The bundled Indic fonts are Regular only; a thin stroke approximates the
        # bold weight of the source captions.
        faux_bold = bool(font_file) and 'bold' not in font_file.lower()
        stroke = lambda size: max(1, size // 30) if faux_bold else 0

        def wrapped_lines(value, current_font, stroke_width):
            """Wrap translations to at most two balanced lines inside the panel."""
            explicit = [line.strip() for line in value.splitlines() if line.strip()]
            if len(explicit) > 1:
                return explicit
            words = (explicit[0] if explicit else '').split()
            if not words:
                return ['']
            result = []
            current = words[0]
            for word in words[1:]:
                candidate = f'{current} {word}'
                bounds = d.textbbox((0, 0), candidate, font=current_font, stroke_width=stroke_width)
                if bounds[2] - bounds[0] <= max_allowed_w:
                    current = candidate
                else:
                    result.append(current)
                    current = word
            result.append(current)
            return result

        font = ImageFont.truetype(font_file, f_size) if font_file else ImageFont.load_default(f_size)
        while f_size > 16:
            sw = stroke(f_size)
            lines = wrapped_lines(text, font, sw)
            max_w = max((d.textbbox((0, 0), l, font=font, stroke_width=sw)[2] - d.textbbox((0, 0), l, font=font, stroke_width=sw)[0]) for l in lines)
            spacing = 6 if len(lines) > 1 else 0
            tot_h = sum((d.textbbox((0, 0), l, font=font, stroke_width=sw)[3] - d.textbbox((0, 0), l, font=font, stroke_width=sw)[1]) for l in lines) + (len(lines) - 1) * spacing
            if len(lines) <= 2 and max_w <= max_allowed_w and tot_h <= max_allowed_h:
                break
            f_size -= 2
            font = ImageFont.truetype(font_file, f_size) if font_file else ImageFont.load_default(f_size)
        sw = stroke(f_size)
        lines = wrapped_lines(text, font, sw)
        spacing = 6 if len(lines) > 1 else 0
        total_h = sum((d.textbbox((0, 0), l, font=font, stroke_width=sw)[3] - d.textbbox((0, 0), l, font=font, stroke_width=sw)[1]) for l in lines) + (len(lines) - 1) * spacing
        cur_y = (bh - total_h) / 2 - 2
        for l in lines:
            tb = d.textbbox((0, 0), l, font=font, stroke_width=sw)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            tx = (bw - tw) / 2 - tb[0]
            d.text((tx, cur_y - tb[1]), l, font=font, fill=text_color, stroke_width=sw, stroke_fill=text_color)
            cur_y += th + spacing

        card_path = folder / f'card_{card_idx}.png'
        img.save(str(card_path))

        inputs.extend(['-i', str(card_path)])
        out_v = f'v{card_idx}'
        track_mode = getattr(seg, 'bbox_track_mode', 'linear')
        x_expr = track_expression(seg.bbox_track, 'x', bx, seg.start, seg.end, track_mode)
        y_expr = track_expression(seg.bbox_track, 'y', by, seg.start, seg.end, track_mode)
        filter_parts.append(f"[{last_v}][{card_idx}:v]overlay=x='{x_expr}':y='{y_expr}':enable='gte(t,{seg.start:.3f})*lt(t,{seg.end:.3f})'[{out_v}]")
        last_v = out_v

    if card_idx == 0:
        raise RuntimeError('No segments with valid boxes and translations to render with pill overlay')

    filter_str = ';'.join(filter_parts)
    output = folder / 'rendered.mp4'
    cmd = [
        config.ffmpeg, '-nostdin', '-y', '-v', 'warning',
        *inputs,
        '-filter_complex', filter_str,
        '-map', f'[{last_v}]', '-map', '0:a?',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-fps_mode', 'passthrough',
        '-c:a', 'copy', '-map_metadata', '0', '-movflags', '+faststart',
        str(output)
    ]
    run(cmd, config.command_timeout, cwd=folder)
    return output


def _balanced_lines(text, count):
    raw = (text or '').replace('\\N', '\n')
    explicit = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(explicit) == count:
        return explicit
    text = ' '.join(explicit)
    words = text.split()
    if count <= 1:
        return [text]
    if not words:
        return [text] * count
    if len(words) < count and not any(ch.isspace() for ch in text):
        chars = list(text)
        result = []
        offset = 0
        for slot in range(count):
            size = round(len(chars) / (count - slot))
            result.append(''.join(chars[offset:offset + size]))
            offset += size
        return result
    lines, remaining = [], words[:]
    for slot in range(count - 1):
        slots_left = count - slot
        target = max(1, round(sum(map(len, remaining)) / slots_left))
        current = []
        while remaining and (not current or len(' '.join(current)) < target) and len(remaining) >= slots_left:
            current.append(remaining.pop(0))
        lines.append(' '.join(current))
    lines.append(' '.join(remaining))
    return lines


def render_line_overlays(video, segments, config, folder):
    """Inpaint and replace detected text lines independently, preserving their gaps."""
    from PIL import Image, ImageDraw, ImageFont
    card = next((s.bbox for s in segments if s.bbox), None)
    if card is None:
        raise RuntimeError('Line rendering requires detected subtitle boxes')
    cleaned = _inpaint_source(video, segments, config, folder)
    candidates = [*config.font_dir.glob('*.ttf'), *config.font_dir.glob('*.otf')]
    lang = getattr(config, 'target_language', '').lower()
    preferred = {'hindi':'NotoSansDevanagari-Regular','marathi':'NotoSansDevanagari-Regular',
                 'tamil':'NotoSansTamil-Regular','kannada':'NotoSansKannada-Regular',
                 'telugu':'NotoSansTelugu-Regular',
                 'malayalam':'NotoSansMalayalam-Regular'}.get(lang)
    font_path = next((p for p in candidates if preferred and p.stem.casefold() == preferred.casefold()), None)
    if font_path is None and candidates:
        font_path = candidates[0]
    inputs = ['-i', str(cleaned)]
    filters, last = [], '0:v'
    n = 0

    def component_expr(track, idx, fallback):
        if len(track) < 2:
            return str(fallback)
        samples = sorted(track, key=lambda p: p[0])
        expr = str(samples[-1][idx])
        for a, b in reversed(list(zip(samples, samples[1:]))):
            if b[0] <= a[0]:
                continue
            val = f"({a[idx]}+({b[idx]}-{a[idx]})*(t-{a[0]:.6f})/{b[0]-a[0]:.6f})"
            expr = f"if(lt(t,{b[0]:.6f}),{val},{expr})"
        return expr

    for seg in segments:
        if not seg.bbox or not seg.line_regions or not seg.translated_text:
            continue
        track = sorted(seg.bbox_track or [], key=lambda p:p[0])
        base = track[0][1:5] if track else (seg.bbox.x, seg.bbox.y, seg.bbox.width, seg.bbox.height)
        lines = _balanced_lines(seg.translated_text, len(seg.line_regions))
        for (rx, ry, rw, rh), line in zip(seg.line_regions, lines):
            if not line:
                continue
            n += 1
            # The mask stays tight around each source line, while the transparent
            # text canvas can use the card's width for a natural translation fit.
            bw = max(32, base[2])
            bh = max(18, round(base[3] * rh * 1.5))
            img = Image.new('RGBA', (bw, bh), (0,0,0,0))
            draw = ImageDraw.Draw(img)
            initial = max(12, int(bh * 0.80))
            font = None
            for size in range(initial, 9, -1):
                try:
                    font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
                except OSError:
                    font = ImageFont.load_default(size=size)
                if draw.textbbox((0,0), line, font=font, stroke_width=max(1,size//16))[2] <= bw * 0.96:
                    break
            stroke = max(1, int(getattr(font, 'size', initial) / 16))
            bounds = draw.textbbox((0,0), line, font=font, stroke_width=stroke)
            tw, th = bounds[2]-bounds[0], bounds[3]-bounds[1]
            tx, ty = (bw-tw)/2-bounds[0], (bh-th)/2-bounds[1]
            draw.text((tx,ty), line, font=font, fill=(255,255,255,255),
                      stroke_width=stroke, stroke_fill=(0,0,0,230))
            image_path = folder / f'line_{n}.png'
            img.save(image_path)
            inputs.extend(['-i', str(image_path)])
            out = f'linev{n}'
            x = component_expr(track,1,base[0])
            y = component_expr(track,2,base[1])
            w = component_expr(track,3,base[2])
            h = component_expr(track,4,base[3])
            x_expr = x
            y_expr = f"({y}+({ry + rh/2:.7f})*{h}-{bh/2})"
            filters.append(f"[{last}][{n}:v]overlay=x='{x_expr}':y='{y_expr}':enable='gte(t,{seg.start:.3f})*lt(t,{seg.end:.3f})'[{out}]")
            last = out
    if not filters:
        raise RuntimeError('No independently detected subtitle lines to render')
    output = folder / 'rendered.mp4'
    run([config.ffmpeg,'-nostdin','-y','-v','warning',*inputs,'-filter_complex',';'.join(filters),
         '-map',f'[{last}]','-map','0:a?','-c:v','libx264','-crf','18','-preset','fast',
         '-fps_mode','passthrough','-c:a','copy','-map_metadata','0','-movflags','+faststart',str(output)],
         config.command_timeout,cwd=folder)
    cleaned.unlink(missing_ok=True)
    verify_rendered_masks(output, segments, config)
    return output


def verify_rendered_masks(video, segments, config):
    """OCR the rendered text bands and reject output where source captions remain."""
    import re
    from difflib import SequenceMatcher
    import cv2
    import pytesseract

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('Could not reopen rendered video for source-text masking QC')

    def at(track, t, fallback):
        if not track:
            return fallback
        if t <= track[0][0]: return tuple(track[0][1:])
        if t >= track[-1][0]: return tuple(track[-1][1:])
        for a, b in zip(track, track[1:]):
            if a[0] <= t <= b[0]:
                f = (t-a[0]) / max(1e-6, b[0]-a[0])
                return tuple(round(a[i] + f*(b[i]-a[i])) for i in range(1,5))
        return fallback

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for seg in segments:
            if not seg.bbox or not seg.line_regions or not seg.source_text or not seg.translated_text:
                continue
            source_words = {w.casefold() for w in re.findall(r'[\w]+', seg.source_text, flags=re.UNICODE) if len(w) >= 4}
            source = re.sub(r'[^\w]', '', seg.source_text.casefold(), flags=re.UNICODE)
            translated = re.sub(r'[^\w]', '', seg.translated_text.casefold(), flags=re.UNICODE)
            # Exact or near-identical source/target wording can be valid for names,
            # labels, and text already written in the requested language.
            if len(source) < 14 or source == translated or len(source_words) < 2:
                continue
            track = sorted(seg.bbox_track or [], key=lambda p:p[0])
            fallback = (seg.bbox.x, seg.bbox.y, seg.bbox.width, seg.bbox.height)
            for fraction in (0.15, 0.5, 0.85):
                t = max(seg.start, min(seg.end - 0.03, seg.start + (seg.end-seg.start)*fraction))
                capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not inspect rendered subtitle at {t:.3f}s')
                x, y, w, h = at(track, t, fallback)
                # Scan the full detected card so missed lines and characters outside
                # the tight removal bands are included in this leak check.
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = cv2.resize(frame[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                recognized = pytesseract.image_to_string(crop, lang=getattr(config,'ocr_language','eng'), config='--psm 6')
                found = re.sub(r'[^\w]', '', recognized.casefold(), flags=re.UNICODE)
                similarity = SequenceMatcher(None, source, found).ratio() if found else 0.0
                found_words = {w.casefold() for w in re.findall(r'[\w]+', recognized, flags=re.UNICODE) if len(w) >= 4}
                overlap = len(source_words & found_words) / max(1, len(source_words))
                if similarity >= 0.78 or (len(source_words) >= 3 and overlap >= 0.72):
                    raise ValueError(f'Original text may remain after masking in segment {seg.id} at {t:.2f}s')
    finally:
        capture.release()

def render(video, subtitles, segments, config, folder, burned=False, metadata=None):
    if (burned and getattr(config, 'pill_mode', False)
            and getattr(config, 'removal_mode', 'overlay') == 'inpainting'):
        line_segments = [s for s in segments if s.translated_text and s.bbox]
        if line_segments:
            missing = [s.id for s in line_segments if not s.line_regions]
            if missing:
                raise RuntimeError(f'Could not locate individual source text lines for subtitle segments: {missing}')
            return render_line_overlays(video, segments, config, folder)
    use_pill = (burned and getattr(config, 'pill_mode', False)
                and getattr(config, 'removal_mode', 'overlay') != 'inpainting'
                and any(s.bbox for s in segments) and any(s.translated_text for s in segments))
    if use_pill:
        try:
            return render_pill_overlays(video, segments, config, folder)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning('Pill overlay rendering failed, falling back to drawbox: %s', exc)

    filters=[]
    if burned and getattr(config, 'removal_mode', 'overlay') == 'inpainting':
        video = _inpaint_source(video, segments, config, folder)
    if burned:
        if config.removal_mode not in ('overlay','inpainting', 'delogo'): raise ValueError('REMOVAL_MODE must be overlay, delogo, or inpainting')
        if config.removal_mode == 'delogo':
            removal = DelogoRemoval()
        elif config.removal_mode == 'inpainting':
            removal = InpaintingRemoval()
        else:
            patch_color = getattr(config, 'overlay_patch_color', '0x000000')
            removal = OverlayRemoval(patch_color)
        patch_spans = []
        for segment in segments:
            if not config.overlay_patch and config.removal_mode not in ('inpainting', 'delogo'):
                continue
            box = segment.bbox
            if not box:
                continue
            if patch_spans:
                last_box, last_start, last_end = patch_spans[-1]
                if (last_box.x == box.x and last_box.y == box.y and
                    last_box.width == box.width and last_box.height == box.height and
                    segment.start <= last_end + 0.15):
                    patch_spans[-1] = (last_box, last_start, max(last_end, segment.end))
                    continue
            patch_spans.append((box, segment.start, segment.end))

        # Back-fill: if top-area segments exist before the first bottom-area mask,
        # add a full-width bottom mask for that uncovered gap so original text is hidden
        if metadata and patch_spans:
            frame_h = getattr(metadata, 'height', 0)
            frame_w = getattr(metadata, 'width', 0)
            if frame_h and frame_w:
                bottom_spans = [(box, s, e) for box, s, e in patch_spans if box.y > frame_h * 0.4]
                top_spans = [(box, s, e) for box, s, e in patch_spans if box.y <= frame_h * 0.4]
                if bottom_spans and top_spans:
                    first_bottom_start = min(s for _, s, _ in bottom_spans)
                    top_covers_before_bottom = any(s < first_bottom_start for _, s, _ in top_spans)
                    if top_covers_before_bottom and first_bottom_start > 0.5:
                        first_bottom_box = min(bottom_spans, key=lambda t: t[1])[0]
                        extra_box = BoundingBox(0, first_bottom_box.y, frame_w, first_bottom_box.height,
                                               patch_color=first_bottom_box.patch_color)
                        patch_spans.insert(0, (extra_box, 0.0, first_bottom_start))

        for box, start, end in patch_spans:
            removal_filter = removal.remove_original_subtitle(box, (start, end))
            if removal_filter:
                filters.append(removal_filter)
    # Fixed relative names avoid FFmpeg filtergraph escaping of user-controlled paths.
    fonts=folder/'render_fonts'
    fonts.mkdir(exist_ok=True)
    for font in config.font_dir.iterdir():
        if font.suffix.lower() in ('.ttf','.otf') and font.is_file(): shutil.copy2(font,fonts/font.name)
    filters.append(f"subtitles=filename={subtitles.name}:fontsdir=render_fonts")
    script=folder/'render.ffscript'
    script.write_text(','.join(filters),encoding='utf-8')
    output=folder/'rendered.mp4'
    run([config.ffmpeg,'-nostdin','-y','-v','warning','-i',video,
         '-map','0:V:0','-map','0:a?','-/filter:v',script.name,
         '-c:v','libx264','-crf','18','-preset','medium','-fps_mode','passthrough',
         '-c:a','copy','-map_metadata','0','-movflags','+faststart',output],config.command_timeout,cwd=folder)
    try:
        shutil.rmtree(fonts)
    except Exception:
        pass
    return output
