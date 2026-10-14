from __future__ import annotations

from pathlib import Path

from .contracts import Artifact, PipelineResult
from .media import FfmpegCaptionBurner, FfmpegFinalAssembler
from .runner import DirectoryLanguagePipeline, PipelineServices
from .service_adapters import LocalSubtitleTranslatorAdapter, TextToAudioAdapter


def run_artifact(
    input_artifact: Artifact,
    *,
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
    source_transcript: Path | None = None,
    reviewed_translation: Path | None = None,
) -> PipelineResult:
    """Run both embedded services from an artifact provided by the host."""
    services = PipelineServices(
        visual_subtitles=LocalSubtitleTranslatorAdapter(backend),
        audio_dubbing=TextToAudioAdapter(backend),
        assembler=FfmpegFinalAssembler(backend),
        publisher=backend,
    )
    return DirectoryLanguagePipeline(services).run(
        input_artifact,
        target_language=target_language,
        source_language=source_language,
        job_id=job_id,
        on_stage=on_stage,
        source_transcript=source_transcript,
        reviewed_translation=reviewed_translation,
    )


def run_operation(
    input_artifact: Artifact,
    *,
    operation: str,
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
    source_transcript: Path | None = None,
    reviewed_translation: Path | None = None,
    add_auto_captions: bool = False,
) -> dict[str, Artifact]:
    """Run masking, dubbing, or both and publish only that operation's outputs."""
    job_id = job_id or "job"
    if operation == "combine":
        result = run_artifact(
            input_artifact, target_language=target_language, backend=backend,
            source_language=source_language, job_id=job_id, on_stage=on_stage,
            source_transcript=source_transcript, reviewed_translation=reviewed_translation,
        )
        final_video = result.final_video
        captions_added = False
        if add_auto_captions and result.audio_subtitles:
            burner = FfmpegCaptionBurner(backend)
            if burner.has_subtitle_stream(input_artifact):
                print("[Captions] Source already contains a subtitle stream; skipping optional auto captions.")
            else:
                if on_stage:
                    on_stage("adding_auto_captions")
                captioned = burner.burn(
                    final_video,
                    result.audio_subtitles,
                    target_language=target_language,
                    job_id=job_id,
                )
                final_video = backend.publish(captioned, name="final_video.mp4", job_id=job_id)
                captions_added = True
        artifacts = {
            "final_video": final_video,
            "visual_subtitles": result.visual_subtitles,
            "service_a_video": result.service_a_video,
        }
        if result.audio_subtitles:
            artifacts["audio_subtitles"] = result.audio_subtitles
        if result.service_b_video:
            artifacts["service_b_video"] = result.service_b_video
        if result.service_b_audio:
            artifacts["service_b_audio"] = result.service_b_audio
        if captions_added:
            artifacts["captioned_video"] = final_video
        return {name: artifact for name, artifact in artifacts.items() if artifact is not None}

    if operation == "masking":
        if on_stage:
            on_stage("masking_running")
        output = LocalSubtitleTranslatorAdapter(backend).run(
            input_artifact, target_language=target_language,
            source_language=source_language, job_id=job_id,
        )
        video = backend.publish(output["video"], name="masking_video.mp4", job_id=job_id)
        subtitles = backend.publish(output["visual_subtitles"], name="visual_subtitles.srt", job_id=job_id)
        if on_stage:
            on_stage("masking_completed")
        return {"final_video": video, "masking_video": video, "visual_subtitles": subtitles}

    if operation == "dubbing":
        if on_stage:
            on_stage("dubbing_running")
        output = TextToAudioAdapter(backend).run(
            input_artifact, target_language=target_language,
            source_language=source_language, job_id=job_id,
            source_transcript=source_transcript, reviewed_translation=reviewed_translation,
        )
        output_video = output["video"]
        captions_added = False
        if add_auto_captions and output.get("audio_subtitles"):
            burner = FfmpegCaptionBurner(backend)
            if burner.has_subtitle_stream(input_artifact):
                print("[Captions] Source already contains a subtitle stream; skipping optional auto captions.")
            else:
                if on_stage:
                    on_stage("adding_auto_captions")
                output_video = burner.burn(
                    output_video,
                    output["audio_subtitles"],
                    target_language=target_language,
                    job_id=job_id,
                )
                captions_added = True
        video = backend.publish(output_video, name="dubbed_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("dubbing_completed")
        artifacts = {"final_video": video, "dubbed_video": video}
        if output.get("audio_subtitles"):
            artifacts["audio_subtitles"] = backend.publish(
                output["audio_subtitles"], name="audio_subtitles.srt", job_id=job_id
            )
        if output.get("audio"):
            artifacts["service_b_audio"] = backend.publish(
                output["audio"], name="dubbed_audio.mp3", job_id=job_id
            )
        if captions_added:
            artifacts["captioned_video"] = video
        return artifacts

    raise ValueError(f"Unsupported operation: {operation}")
