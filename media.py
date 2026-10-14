from __future__ import annotations

import json
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Protocol

from .contracts import Artifact


class ArtifactBackend(Protocol):
    def download(self, artifact: Artifact, destination: Path) -> Path: ...

    def upload(self, source: Path, *, filename: str, content_type: str) -> Artifact: ...

    def publish(self, artifact: Artifact, *, name: str, job_id: str) -> Artifact: ...


class FileArtifactBackend:
    """Artifact backend rooted in the host's configured persistent storage.

    Production hosts can replace this with S3, GCS, Azure Blob, or their own
    artifact store without changing the pipeline or service adapters.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self, artifact: Artifact, destination: Path) -> Path:
        source = Path(artifact.metadata.get("path", ""))
        if not source.is_file():
            raise FileNotFoundError(f"Artifact is unavailable: {artifact.uri}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def upload(self, source: Path, *, filename: str, content_type: str) -> Artifact:
        destination = self.root / filename
        counter = 1
        while destination.exists():
            destination = self.root / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
            counter += 1
        shutil.copyfile(source, destination)
        return Artifact(
            uri=f"artifact://{destination.name}",
            filename=destination.name,
            content_type=content_type,
            metadata={"path": str(destination)},
        )


    def publish(self, artifact: Artifact, *, name: str, job_id: str) -> Artifact:
        source = Path(str(artifact.metadata.get("path", "")))
        if not source.is_file():
            raise FileNotFoundError(f"Cannot publish missing artifact: {artifact.uri}")
        destination = self.root / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return Artifact(
            uri=f"artifact://{name}",
            filename=name,
            content_type=artifact.content_type,
            metadata={"path": str(destination)},
        )


# Kept as a compatibility alias for callers of the first orchestration draft.
TemporaryArtifactBackend = FileArtifactBackend


class FfmpegCaptionBurner:
    """Burn audio-timed translated captions into a video using a fitted box."""

    FONT_BY_LANGUAGE = {
        "hi": "Noto Sans Devanagari",
        "mr": "Noto Sans Devanagari",
        "kn": "Noto Sans Kannada",
        "ml": "Noto Sans Malayalam",
        "ta": "Noto Sans Tamil",
        "te": "Noto Sans Telugu",
    }
    ENGLISH_TERM_REPLACEMENTS = (
        (r"प्लिक्स", "Plix"),
        (r"गुआवा\s+स्किनकेयर", "Guava Skincare"),
        (r"गुआवा", "Guava"),
        (r"स्किनकेयर", "Skincare"),
        (r"फेस\s*वॉश", "Face Wash"),
        (r"नियासिनामाइड", "Niacinamide"),
        (r"(?:विटामिन|व्हिटॅमिन)\s*सी", "Vitamin C"),
        (r"हायलूरोनिक\s*(?:अॅसिड|एसिड)", "Hyaluronic Acid"),
        (r"सॅलिसिलिक\s*(?:अॅसिड|एसिड)", "Salicylic Acid"),
        (r"ग्लायकोलिक\s*(?:अॅसिड|एसिड)", "Glycolic Acid"),
        (r"रेटिनॉल", "Retinol"),
        (r"मॉइस्चराइ[ज़ज]़र|मॉइश्चरायझर", "Moisturizer"),
        (r"सीरम", "Serum"),
        (r"टोनर", "Toner"),
        (r"एएम[-–]?पीएम", "AM-PM"),
    )

    def __init__(
        self,
        backend: ArtifactBackend,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ):
        self.backend = backend
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.font_dir = Path(__file__).resolve().parent / "vendor" / "service_a" / "fonts"

    def _probe(self, video_path: Path) -> dict:
        command = [
            self.ffprobe_binary, "-v", "error", "-show_streams", "-show_format", "-of", "json",
            str(video_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Caption video probe failed: {completed.stderr.strip()}")
        return json.loads(completed.stdout or "{}")

    def has_subtitle_stream(self, video: Artifact) -> bool:
        with tempfile.TemporaryDirectory(prefix="caption-probe-") as temp:
            video_path = self.backend.download(video, Path(temp) / video.filename)
            probe = self._probe(video_path)
            streams = probe.get("streams", [])
            return any(stream.get("codec_type") == "subtitle" for stream in streams)

    @staticmethod
    def _group_caption_words(words: list[str]) -> list[list[str]]:
        """Partition atomic words into mostly 4–5-word caption cards."""
        count = len(words)
        best: list[tuple[int, list[list[str]]] | None] = [None] * (count + 1)
        best[count] = (0, [])
        for start in range(count - 1, -1, -1):
            visible_words = 0
            for end in range(start + 1, count + 1):
                visible_words += words[end - 1].count("\u00a0") + 1
                if visible_words > 5:
                    break
                remainder = best[end]
                if remainder is None:
                    continue
                # Four or five words are ideal. Three or fewer are only used
                # when the cue's total word count makes 4–5 impossible.
                shortfall = max(0, 4 - visible_words)
                penalty = shortfall * shortfall + remainder[0]
                candidate_groups = [words[start:end], *remainder[1]]
                candidate = (penalty, candidate_groups)
                current = best[start]
                current_first_size = (
                    sum(item.count("\u00a0") + 1 for item in current[1][0])
                    if current and current[1] else 0
                )
                candidate_key = (candidate[0], len(candidate[1]), -visible_words)
                current_key = (current[0], len(current[1]), -current_first_size) if current else None
                if current is None or candidate_key < current_key:
                    best[start] = candidate
        return best[0][1] if best[0] else [words]

    def _create_ass(
        self,
        subtitle_path: Path,
        ass_path: Path,
        *,
        width: int,
        height: int,
        duration_ms: int,
        target_language: str,
    ) -> int:
        import pysubs2

        subtitles = pysubs2.load(str(subtitle_path), encoding="utf-8")
        timed_runs: list[list[tuple[str, int, int]]] = []
        current_run: list[tuple[str, int, int]] = []
        for event in subtitles.events:
            text = event.text.replace(r"\N", " ").replace("\n", " ").strip()
            for pattern, replacement in self.ENGLISH_TERM_REPLACEMENTS:
                # Keep a multi-word entity in the same caption chunk. NBSP is
                # converted back to a normal display space after chunking.
                protected = replacement.replace(" ", "\u00a0")
                text = re.sub(pattern, protected, text, flags=re.IGNORECASE)
            text = re.sub(r"[ \t\r\n]+", " ", text).strip()
            if not text:
                continue
            start = max(0, int(event.start))
            end = min(duration_ms, int(event.end))
            if end - start < 100:
                continue

            words = text.split(" ")
            weights = [max(1, len(re.sub(r"[^\w]", "", word.replace("\u00a0", "")))) for word in words]
            total_weight = max(1, sum(weights))
            elapsed_weight = 0
            for index, (word, weight) in enumerate(zip(words, weights)):
                word_start = start + round((end - start) * elapsed_weight / total_weight)
                elapsed_weight += weight
                word_end = end if index == len(words) - 1 else start + round((end - start) * elapsed_weight / total_weight)
                if current_run and word_start - current_run[-1][2] > 350:
                    timed_runs.append(current_run)
                    current_run = []
                current_run.append((word, word_start, word_end))
        if current_run:
            timed_runs.append(current_run)

        valid_events = []
        for run in timed_runs:
            grouped_words = self._group_caption_words([word for word, _start, _end in run])
            offset = 0
            for group in grouped_words:
                group_units = run[offset:offset + len(group)]
                offset += len(group)
                caption_event = pysubs2.SSAEvent(
                    start=group_units[0][1],
                    end=max(group_units[0][1] + 100, group_units[-1][2]),
                    text=" ".join(group).replace("\u00a0", " "),
                    style="AutoCaption",
                )
                valid_events.append(caption_event)
        if not valid_events:
            raise RuntimeError("The translated caption file contains no timed text.")

        subtitles.events = valid_events
        subtitles.info["PlayResX"] = str(width)
        subtitles.info["PlayResY"] = str(height)
        subtitles.info["WrapStyle"] = "0"
        language = target_language.lower().split("-")[0]
        font_name = self.FONT_BY_LANGUAGE.get(language, "DejaVu Sans")
        font_size = max(28, min(64, round(height * 0.036)))
        horizontal_margin = max(32, round(width * 0.07))
        vertical_margin = max(42, round(height * 0.065))
        subtitles.styles["AutoCaption"] = pysubs2.SSAStyle(
            fontname=font_name,
            fontsize=font_size,
            primarycolor=pysubs2.Color(255, 255, 255, 0),
            secondarycolor=pysubs2.Color(255, 255, 255, 0),
            outlinecolor=pysubs2.Color(0, 0, 0, 0),
            backcolor=pysubs2.Color(0, 0, 0, 0),
            bold=True,
            borderstyle=3,
            outline=max(5, round(font_size * 0.16)),
            shadow=0,
            alignment=pysubs2.Alignment.BOTTOM_CENTER,
            marginl=horizontal_margin,
            marginr=horizontal_margin,
            marginv=vertical_margin,
        )
        subtitles.save(str(ass_path), format_="ass", encoding="utf-8")
        return len(valid_events)

    def burn(
        self,
        video: Artifact,
        subtitles: Artifact,
        *,
        target_language: str,
        job_id: str,
    ) -> Artifact:
        with tempfile.TemporaryDirectory(prefix=f"caption-burn-{job_id}-") as temp:
            root = Path(temp)
            video_path = self.backend.download(video, root / "input.mp4")
            subtitle_path = self.backend.download(subtitles, root / "captions.srt")
            probe = self._probe(video_path)
            streams = probe.get("streams", [])
            video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
            if not video_stream:
                raise RuntimeError("Caption input has no video stream.")
            width = int(video_stream.get("width") or 0)
            height = int(video_stream.get("height") or 0)
            if width <= 0 or height <= 0:
                raise RuntimeError("Caption input has invalid video dimensions.")
            try:
                duration_ms = round(float(probe.get("format", {}).get("duration")) * 1000)
            except (TypeError, ValueError):
                duration_ms = max((event.end for event in __import__("pysubs2").load(str(subtitle_path)).events), default=0)
            if duration_ms <= 0:
                raise RuntimeError("Caption input has invalid duration.")

            ass_path = root / "captions.ass"
            event_count = self._create_ass(
                subtitle_path,
                ass_path,
                width=width,
                height=height,
                duration_ms=duration_ms,
                target_language=target_language,
            )
            fonts_path = root / "fonts"
            if self.font_dir.is_dir():
                shutil.copytree(self.font_dir, fonts_path)
            else:
                fonts_path.mkdir()
            output = root / "captioned-video.mp4"
            command = [
                self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(video_path),
                "-vf", "subtitles=filename=captions.ass:fontsdir=fonts",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-map_metadata", "0",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy", "-movflags", "+faststart",
                str(output),
            ]
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"Caption burn failed: {completed.stderr.strip()}")
            print(f"[Captions] Burned {event_count} translated cues with fitted opaque backgrounds.")
            return self.backend.upload(output, filename="captioned-video.mp4", content_type="video/mp4")


class FfmpegFinalAssembler:
    """Keep Service B's dubbed video/audio and attach Service A's visual SRT."""

    def __init__(self, backend: ArtifactBackend, ffmpeg_binary: str = "ffmpeg"):
        self.backend = backend
        self.ffmpeg_binary = ffmpeg_binary

    def assemble(
        self,
        video: Artifact,
        audio: Artifact | None,
        visual_subtitles: Artifact,
        *,
        job_id: str,
    ) -> Artifact:
        with tempfile.TemporaryDirectory(prefix=f"dir-language-{job_id}-") as temp:
            root = Path(temp)
            video_path = self.backend.download(video, root / "service-a-video.mp4")
            subtitle_path = self.backend.download(visual_subtitles, root / "visual.srt")
            audio_path = self.backend.download(audio, root / audio.filename) if audio else None
            output = root / "final_video.mp4"

            command = [self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video_path)]
            if audio_path:
                command += ["-i", str(audio_path)]
            command += ["-i", str(subtitle_path), "-map", "0:v:0"]
            if audio_path:
                command += ["-map", "1:a:0"]
            else:
                command += ["-map", "0:a:0?"]
            subtitle_input = 2 if audio_path else 1
            command += [
                "-map", f"{subtitle_input}:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=und",
                "-metadata:s:s:0", "title=Visual subtitles (Service A)",
                "-movflags", "+faststart",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"Final video mux failed: {completed.stderr.strip()}")
            return self.backend.upload(output, filename="final_video.mp4", content_type="video/mp4")
