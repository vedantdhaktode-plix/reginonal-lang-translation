import html
import re
from dataclasses import asdict, dataclass
from .common import PipelineError

STAMP = r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})"
TIMING = re.compile(rf"^{STAMP}\s*-->\s*{STAMP}$")


@dataclass
class Cue:
    id: int
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self):
        return asdict(self)


def _parse_timestamp_ms(ts: str) -> int:
    """Parse HH:MM:SS,mmm or HH:MM:SS.mmm or MM:SS.mmm into milliseconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split(".")
        s = int(s_parts[0])
        ms = int((s_parts[1] + "000")[:3]) if len(s_parts) > 1 else 0
        return ((h * 60 + m) * 60 + s) * 1000 + ms
    elif len(parts) == 2:
        m = int(parts[0])
        s_parts = parts[1].split(".")
        s = int(s_parts[0])
        ms = int((s_parts[1] + "000")[:3]) if len(s_parts) > 1 else 0
        return (m * 60 + s) * 1000 + ms
    else:
        s_float = float(ts)
        return int(round(s_float * 1000))


def parse_srt(text, preserve_lines=False):
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise PipelineError("SRT is empty.")

    # Remove markdown code fences if present
    if "```srt" in text:
        text = text.split("```srt", 1)[1].split("```", 1)[0].strip()
    elif "```vtt" in text:
        text = text.split("```vtt", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Filter out WEBVTT headers or comments
    lines = text.splitlines()
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("WEBVTT") or stripped.startswith("NOTE"):
            continue
        filtered_lines.append(line)
    text = "\n".join(filtered_lines).strip()

    timing_pattern = re.compile(
        r"(\d{1,4}:[0-5]?\d:[0-5]?\d[,\.]\d+|\d{1,2}:[0-5]?\d[,\.]\d+)\s*-+>\s*(\d{1,4}:[0-5]?\d:[0-5]?\d[,\.]\d+|\d{1,2}:[0-5]?\d[,\.]\d+)"
    )

    blocks = re.split(r"\n\s*\n", text)
    raw_cues = []

    for block in blocks:
        blines = [l.strip() for l in block.splitlines() if l.strip()]
        if not blines:
            continue
        timing_idx = None
        match = None
        for idx, line in enumerate(blines):
            m = timing_pattern.search(line)
            if m:
                timing_idx = idx
                match = m
                break
        if not match:
            continue

        cue_id = len(raw_cues) + 1
        if timing_idx > 0 and blines[timing_idx - 1].isdigit():
            cue_id = int(blines[timing_idx - 1])

        start_str, end_str = match.group(1), match.group(2)
        try:
            start_ms = _parse_timestamp_ms(start_str)
            end_ms = _parse_timestamp_ms(end_str)
        except Exception:
            continue

        text_lines = blines[timing_idx + 1:]
        cleaned = [" ".join(html.unescape(re.sub(r"<[^>]+>", "", tl)).split()) for tl in text_lines]
        cue_text = "\n".join(cleaned) if preserve_lines else " ".join(" ".join(cleaned).split())
        cue_text = cue_text.strip()
        if not cue_text:
            continue

        raw_cues.append((cue_id, start_ms, end_ms, cue_text))

    # Fallback to line-by-line scanning if block split didn't find cues
    if not raw_cues:
        lines = [l.strip() for l in text.splitlines()]
        current_id = 1
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            match = timing_pattern.search(line)
            if match:
                try:
                    start_ms = _parse_timestamp_ms(match.group(1))
                    end_ms = _parse_timestamp_ms(match.group(2))
                except Exception:
                    idx += 1
                    continue
                if idx > 0 and lines[idx - 1].isdigit():
                    current_id = int(lines[idx - 1])
                text_lines = []
                idx += 1
                while idx < len(lines):
                    next_line = lines[idx]
                    if timing_pattern.search(next_line):
                        break
                    if next_line.isdigit() and idx + 1 < len(lines) and timing_pattern.search(lines[idx + 1]):
                        break
                    if next_line:
                        text_lines.append(next_line)
                    idx += 1
                cleaned = [" ".join(html.unescape(re.sub(r"<[^>]+>", "", tl)).split()) for tl in text_lines]
                cue_text = "\n".join(cleaned) if preserve_lines else " ".join(" ".join(cleaned).split())
                if cue_text.strip():
                    raw_cues.append((current_id, start_ms, end_ms, cue_text.strip()))
                    current_id += 1
            else:
                idx += 1

    if not raw_cues:
        raise PipelineError("Could not find any valid subtitle cues in text.")

    # Sanitize and ensure monotonic, non-overlapping timeline
    cues = []
    for idx, (cid, start, end, ctext) in enumerate(raw_cues, 1):
        if end <= start:
            end = start + 500
        if cues and start < cues[-1].end_ms:
            start = cues[-1].end_ms
            if end <= start:
                end = start + 300
        cues.append(Cue(id=idx, start_ms=start, end_ms=end, text=ctext))

    return cues


def stamp(ms):
    h, remainder = divmod(int(ms), 3600000)
    m, remainder = divmod(remainder, 60000)
    s, fraction = divmod(remainder, 1000)
    return f"{h:02}:{m:02}:{s:02},{fraction:03}"


def render_srt(segments):
    return "\n\n".join(
        f"{index}\n{stamp(s['start_ms'])} --> {stamp(s['end_ms'])}\n{s['target_text']}"
        for index, s in enumerate(segments, 1) if not s.get("skip")
    ) + "\n"
