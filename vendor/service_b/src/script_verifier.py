"""AI quality checks for transcript and translation review gates."""
import json
import os

import requests

from .common import PipelineError
from .srt_parser import parse_srt


def verify_srt(srt_text: str, *, kind: str, source_text: str = "", target_language: str = "", source_language: str = "auto") -> dict:
    """Return a concise Groq review report without changing user-approved text."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise PipelineError("GROQ_API_KEY is required for the transcript and translation AI review gates.")
    cues = parse_srt(srt_text)
    model = os.getenv("SCRIPT_REVIEW_MODEL", os.getenv("TRANSLATION_MODEL", "openai/gpt-oss-120b"))
    if kind == "source":
        instructions = (
            "Review an automatic speech transcript for likely transcription defects, unsupported hallucinations, "
            "broken phrases, suspicious names, and cue/timing problems. You do not have the audio, so do not claim "
            "to verify words against it. Only flag evidence visible in the transcript. Return JSON: {summary, issues:[{cue_id,severity,reason,suggestion}]}."
        )
        payload = {"source_language": source_language, "transcript": [{"id": c.id, "start_ms": c.start_ms, "end_ms": c.end_ms, "text": c.text} for c in cues]}
    else:
        spoken_register = ""
        if target_language.lower().split("-")[0] == "hi":
            spoken_register = (
                " For Hindi dubbing, also flag formal, literary, Sanskritized, or translated-sounding wording that ordinary Indian speakers would replace with simple conversational Hindi/Hinglish. "
                "Preserve natural English loanwords and code-switching from the source (for example issue/problem, shampoo, use, try, product, dryness, link, comment, and DM). "
                "Flag unnecessary words such as समस्या, उपयोग, अनेक प्रकार, उत्पाद, सूखापन, सहायता, हेतु, पर्याप्त, or प्राप्त when a familiar spoken alternative fits. "
                "Also flag added promotional adjectives or claims. Suggestions must sound natural aloud, not merely be grammatically correct."
            )
        instructions = (
            f"Review the target-language translation for semantic fidelity, omissions, additions, awkward or incorrect wording, "
            f"and names. Target language is {target_language}. Compare each translated cue only with its corresponding source cue. "
            "Flag any brand name, product/range name, model name, or ingredient name that was transliterated into the target script instead of being restored to its standard English Latin spelling. "
            "Examples include Plix, Guava, Niacinamide, Vitamin C, Hyaluronic Acid, Salicylic Acid, Retinol, and SPF 50. "
            f"{spoken_register} "
            "Return JSON: {summary, issues:[{cue_id,severity,reason,suggestion}]}. Do not rewrite the transcript."
        )
        source_cues = parse_srt(source_text)
        if len(source_cues) != len(cues):
            raise PipelineError("Translation verification requires the same number of source and target cues.")
        payload = {"source_language": source_language, "target_language": target_language,
                   "cues": [{"id": a.id, "source": a.text, "translation": b.text} for a, b in zip(source_cues, cues)]}
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": instructions},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
            timeout=90,
        )
        response.raise_for_status()
        report = json.loads(response.json()["choices"][0]["message"]["content"])
        if not isinstance(report.get("issues"), list) or not isinstance(report.get("summary"), str):
            raise ValueError("missing summary or issues")
        return report
    except Exception as exc:
        raise PipelineError("Groq AI script review failed; the approval gate remains closed.") from exc
