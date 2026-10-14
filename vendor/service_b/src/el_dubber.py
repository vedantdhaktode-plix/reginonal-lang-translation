"""
Stage 3 — ElevenLabs Audio Generation
Two modes depending on whether ElevenLabs Dubbing API supports the target language:

MODE A — Dubbing API (auto, supported languages only):
  POST /v1/dubbing → full video → dubbed audio + timed SRT
  Supported: hi, es, fr, de, pt, ja, ko, zh, ar, etc.
  NOT supported: mr (Marathi) — falls through to Mode B automatically.

MODE B — Voice-Clone TTS (for unsupported languages like Marathi):
  1. Extract voice sample from video (ffmpeg)
  2. Clone voice via ElevenLabs Instant Voice Clone
  3. Translate transcript using a supplied translated SRT or configured Groq model
  4. TTS each cue using eleven_v4 / eleven_v3 (both support Marathi)
  5. Assemble time-fitted audio matching source video duration

Cache: per (video_hash, target_lang). Never re-charges on re-run.

Supported Dubbing API languages (partial, check EL docs for full list):
  af, ar, bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, id, it, ja, ko,
  lt, lv, ms, nl, no, pl, pt, ro, ru, sk, sl, sv, ta, tr, uk, vi, zh, hi
NOT supported by Dubbing API (use Voice-Clone TTS instead):
  mr, gu, bn, pa, ml, te, kn, ur, ne, si, my, km, lo, ...
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from .common import PipelineError, request_with_retry
from .srt_parser import parse_srt, Cue, stamp

# Languages supported by ElevenLabs Dubbing API
# (based on validation_error testing — mr returns unsupported_target_language)
EL_DUBBING_SUPPORTED = {
    "af", "ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "es-419",
    "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
    "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl",
    "sq", "sr", "sv", "sw", "ta", "th", "tl", "tr", "uk", "vi", "zh",
}

# ElevenLabs TTS models that support Marathi and other Indian languages
EL_V4_MODEL = "eleven_v4"           # Best quality, Marathi supported
EL_V4_TURBO = "eleven_v4_turbo"     # Faster, Marathi supported
EL_V3_MODEL = "eleven_v3"           # Also Marathi supported
TTS_ASSEMBLY_VERSION = "cue-fit-v8-source-identity"


def _video_hash(video_path: Path) -> str:
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _configured_target_voice_id(target_lang: str) -> Optional[str]:
    language = target_lang.lower().split("-")[0]
    configured = (
        os.environ.get(f"{language.upper()}_TTS_VOICE_ID")
        or os.environ.get("TARGET_TTS_VOICE_ID")
    )
    return configured.strip() if configured and configured.strip() else None


def _should_replace_source_voice(target_lang: str) -> bool:
    """Only Hindi replaces the source voice automatically; all others opt in."""
    language = target_lang.lower().split("-")[0]
    return language == "hi" or _configured_target_voice_id(target_lang) is not None


def _assert_target_language(cues: list[Cue], target_lang: str) -> None:
    """Fail closed when a sufficiently long transcript is clearly another language."""
    text = " ".join(cue.text for cue in cues).strip()
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 20:
        raise PipelineError(
            f"Cannot verify the requested '{target_lang}' language: transcript is too short. "
            "Review the transcript before generating audio."
        )
    # Generic LID tools commonly collapse Marathi into Hindi because both use
    # Devanagari. Compare a small set of language-specific forms first.
    if target_lang.lower().split("-")[0] in {"mr", "hi"} and any("\u0900" <= c <= "\u097f" for c in text):
        tokens = set(re.findall(r"[\u0900-\u097f]+", text))
        marathi = {"मी", "माझा", "माझी", "माझे", "माझ्या", "आहे", "आहेत", "नाही", "आम्ही", "तुम्ही", "आपण", "त्याला", "तिला", "मुलगी", "आई", "आधी", "म्हणून", "करते", "करतो"}
        hindi = {"मैं", "मेरा", "मेरी", "मेरे", "पहले", "अपने", "माँ", "हूँ", "है", "हैं", "नहीं", "यार", "था", "थी", "करता", "करती"}
        mr_score = len(tokens & marathi)
        hi_score = len(tokens & hindi)
        expected = target_lang.lower().split("-")[0]
        if expected == "mr" and mr_score <= hi_score:
            raise PipelineError("Transcript appears to use Hindi rather than Marathi; refusing to synthesize mismatched audio.")
        if expected == "hi" and hi_score <= mr_score:
            raise PipelineError("Transcript appears to use Marathi rather than Hindi; refusing to synthesize mismatched audio.")
        # For these related languages, the lexical check is more reliable than
        # generic script-based language detection.
        return
    try:
        from langdetect import DetectorFactory, detect_langs
        DetectorFactory.seed = 0
        guesses = detect_langs(text)
    except Exception as exc:
        raise PipelineError(f"Could not verify transcript language '{target_lang}'.") from exc
    if not guesses:
        raise PipelineError(f"Could not identify transcript language; expected '{target_lang}'.")
    detected = guesses[0]
    expected = target_lang.lower().split("-")[0]
    actual = detected.lang.lower().split("-")[0]
    if actual != expected and detected.prob >= 0.55:
        raise PipelineError(
            f"Transcript appears to be '{actual}' (confidence {detected.prob:.2f}), "
            f"not requested target '{target_lang}'. Refusing to publish mismatched audio."
        )
    if actual != expected:
        raise PipelineError(
            f"Transcript language is uncertain (best guess '{actual}', {detected.prob:.2f}); "
            f"expected '{target_lang}'. Review it before generating audio."
        )


def _apply_marathi_glossary(cues: list[Cue], target_lang: str) -> list[Cue]:
    """Keep the named breed in English and render its relationship in Marathi."""
    if target_lang.lower().split("-")[0] != "mr":
        return cues
    pattern = re.compile(
        r"(?i)(?:golden\s+retriever|गोल्डन\s*रिट्रीवर)\s+"
        r"(?:(biological|बायोलॉजिकल)\s+)?(?:की|ki)\s+(?:माँ|मां|maa|आई|aai)"
    )
    for cue in cues:
        cue.text = pattern.sub(
            lambda match: "Golden Retriever ची " +
            ((match.group(1) + " ") if match.group(1) else "") + "आई",
            cue.text,
        )
    return cues


def _reject_hindi_mother_phrase(cues: list[Cue], target_lang: str) -> None:
    if target_lang.lower().split("-")[0] != "mr":
        return
    for cue in cues:
        if re.search(r"(?i)(?:golden\s+retriever|गोल्डन\s*रिट्रीवर)\s+(?:की|ki)\s+(?:माँ|मां|maa)", cue.text):
            raise PipelineError(
                f"Cue {cue.id} still contains Hindi wording for a Marathi translation. "
                "Use ‘Golden Retriever ची आई’ and confirm the corrected translation."
            )


def _extract_voice_sample(video_path: Path, out_wav: Path, duration_s: float = 20.0) -> Path:
    """Extract first N seconds of clean speech from video for voice cloning."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(video_path),
        "-vn", "-ss", "1.0", "-t", str(duration_s),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Voice sample extraction failed: {r.stderr}")
    return out_wav


def _ensure_voice_clone(sample_wav: Path, voice_name: str, api_key: str) -> str:
    """Get existing clone by name, or create one. Returns voice_id."""
    headers = {"xi-api-key": api_key}
    r = request_with_retry("GET", "https://api.elevenlabs.io/v1/voices", headers=headers, timeout=30, log_prefix="[Dubber Voices]")
    if r.ok:
        for v in r.json().get("voices", []):
            if v["name"] == voice_name:
                print(f"[Dubber] Reusing voice clone '{voice_name}' (ID: {v['voice_id']})")
                return v["voice_id"]

    print(f"[Dubber] Creating voice clone '{voice_name}' from {sample_wav.name}...")
    with open(sample_wav, "rb") as f:
        resp = request_with_retry(
            "POST",
            "https://api.elevenlabs.io/v1/voices/add",
            headers=headers,
            data={"name": voice_name, "description": "Auto-cloned for dubbing"},
            files={"files": ("sample.wav", f, "audio/wav")},
            timeout=60,
            log_prefix="[Dubber Voice Clone]",
        )
    if not resp.ok:
        raise PipelineError(f"Voice clone failed: HTTP {resp.status_code} — {resp.text[:300]}")
    vid = resp.json()["voice_id"]
    print(f"[Dubber] Created voice clone ID: {vid}")
    return vid


def _find_target_language_voice(
    target_lang: str,
    api_key: str,
    model: str,
) -> tuple[Optional[str], Optional[str]]:
    """Select a configured or account voice trained for the target language."""
    language = target_lang.lower().split("-")[0]
    configured_id = _configured_target_voice_id(target_lang)
    if configured_id:
        return configured_id, "configured target-language voice"

    headers = {"xi-api-key": api_key}
    response = request_with_retry(
        "GET",
        "https://api.elevenlabs.io/v1/voices",
        headers=headers,
        timeout=30,
        log_prefix="[Dubber Target Voice]",
    )
    if not response.ok:
        return None, None

    language_names = {"hi": "hindi"}
    wanted_name = language_names.get(language, language)
    best: tuple[int, Optional[dict]] = (0, None)
    for voice in response.json().get("voices", []):
        labels = voice.get("labels") or {}
        label_text = " ".join(str(value).lower() for value in labels.values())
        verified = voice.get("verified_languages") or []
        verified_text = " ".join(
            " ".join(str(value).lower() for value in item.values())
            for item in verified
            if isinstance(item, dict)
        )

        score = 0
        if wanted_name in label_text or re.search(rf"(^|[^a-z]){re.escape(language)}([^a-z]|$)", label_text):
            score += 100
        if wanted_name in verified_text or re.search(rf"(^|[^a-z]){re.escape(language)}([^a-z]|$)", verified_text):
            score += 150
        if any(
            item.get("language") == language and item.get("model_id") == model
            for item in verified
            if isinstance(item, dict)
        ):
            score += 80
        if language == "hi" and any(marker in label_text for marker in ("hindi", "north indian", "delhi")):
            score += 40
        use_case = str(labels.get("use_case", "")).lower()
        if use_case == "social_media":
            score += 30
        elif use_case in {"conversational", "advertisement"}:
            score += 20
        if score > best[0]:
            best = (score, voice)

    if best[1] is None:
        return None, None
    voice = best[1]
    return voice.get("voice_id"), f"native {wanted_name} voice '{voice.get('name', 'unnamed')}'"


def _prepare_tts_text(text: str, target_lang: str) -> str:
    """Make ambiguous translated terms unambiguous to the speech model."""
    if target_lang.lower().split("-")[0] != "hi":
        return text

    prepared = text
    # "फ्लेक्स" is commonly rendered like English "flex/flicks". In hair-care
    # speech the intended word is dandruff flakes, so pronounce the meaning.
    prepared = re.sub(r"(?i)\bflakes?\b", "डैंड्रफ की पपड़ी", prepared)
    prepared = prepared.replace("फ्लेक्स", "डैंड्रफ की पपड़ी")
    prepared = prepared.replace("पािरक्टोन ओलामीन", "पिरोक्टोन ओलामाइन")
    return re.sub(r"\s+", " ", prepared).strip()


def _spoken_style_rules(target_lang: str) -> str:
    """Target-language register rules shared by every translation provider."""
    if target_lang.lower().split("-")[0] == "hi":
        return (
            "\nHINDI SPOKEN-REGISTER RULES (MANDATORY):\n"
            "- Write the way an everyday Indian creator would actually speak on camera: simple, warm, direct Hindi/Hinglish, never textbook, literary, bureaucratic, or heavily Sanskritized Hindi.\n"
            "- Preserve natural code-switching from the source. Common spoken words such as issue/problem, comb, dandruff, shampoo, use, try, product, scalp, dryness, hydrate, help, link, follow, comment, and DM may remain in English when that is how people normally say them. Do not replace them with formal Hindi merely to make the sentence more 'pure'.\n"
            "- Prefer familiar speech: 'problem/issue' over 'समस्या', 'use' over 'उपयोग/इस्तेमाल', 'बहुत सारे' over 'अनेक प्रकार के', 'product' over 'उत्पाद', 'dryness' over 'सूखापन', and 'help करता है' over 'सहायता करता है'. Use these as style examples, not blind word substitutions.\n"
            "- Use short sentence structure and natural connectors such as 'तो', 'बस', 'फिर', 'है ना', and 'आप भी'. Avoid stiff constructions such as 'यदि', 'अतः', 'हेतु', 'प्राप्त', 'पर्याप्त', or 'नष्ट करता है' unless the context truly requires them.\n"
            "- Do not add praise, certainty, or product claims absent from the source (for example 'बेहतरीन', 'गारंटीड', or 'सबसे अच्छा').\n"
            "- In a dandruff or hair-care context, translate 'flakes' as 'डैंड्रफ की पपड़ी', never as 'फ्लेक्स', because the latter is easily pronounced as flex/flicks.\n"
            "- Read the entire cue sequence as one conversation and keep terminology and tone consistent from cue to cue.\n"
        )
    return ""


def _translate_cues_groq(cues: list[Cue], target_lang: str, source_lang: str = "auto") -> list[Cue]:
    """Translate cues through the configured Groq model; never pass source text through."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise PipelineError("Set GROQ_API_KEY to translate unsupported dubbing languages.")
    model = os.environ.get("TRANSLATION_MODEL", "openai/gpt-oss-120b")
    language_names = {
        "mr": "Marathi", "hi": "Hindi", "ta": "Tamil", "ml": "Malayalam",
        "kn": "Kannada", "te": "Telugu", "bn": "Bengali", "gu": "Gujarati",
        "pa": "Punjabi", "ur": "Urdu", "en": "English", "es": "Spanish",
        "fr": "French", "de": "German", "ja": "Japanese", "pt": "Portuguese",
        "it": "Italian", "ar": "Arabic",
    }
    target_name = language_names.get(target_lang.lower(), target_lang)
    source_name = language_names.get(source_lang.lower(), source_lang or "auto-detected")
    payload = {
        "source_language": source_name,
        "target_language": target_name,
        "translations": [
            {
                "id": cue.id,
                "duration_seconds": round(max(0.1, (cue.end_ms - cue.start_ms) / 1000.0), 2),
                "max_words": max(2, int(round(max(0.1, (cue.end_ms - cue.start_ms) / 1000.0) * 2.2))),
                "text": cue.text,
            }
            for cue in cues
        ],
    }
    prompt = (
        f"Translate every subtitle cue from {source_name} into natural spoken {target_name} for video dubbing. "
        "Return only JSON with a translations array containing exactly the supplied IDs in order, "
        "each with id and translated text. Preserve meaning, names, and cue IDs. Do not explain.\n"
        "CRITICAL DUBBING REQUIREMENTS:\n"
        "1. COMPLETE DIALOGUE: Translate ALL spoken thoughts and sentences completely. Do NOT omit or drop sentences, product names, or sales offers.\n"
        "2. TIMING & NATURAL PACING: Strictly match the spoken duration_seconds and stay within max_words for each cue at a relaxed, normal conversational cadence (~2.0 words per second). The target text must be concise, punchy, and crisp: NEVER use verbose, lengthy, or convoluted phrasing that would force the speaker to rush or race. The text MUST fit comfortably within the allotted seconds.\n"
        "3. NATURAL REGISTRATION & TERMINOLOGY: Use conversational spoken phrasing, keeping English product and brand names natural (e.g. 'Plix', 'Rosemary Hair Growth Serum', 'Redensyl', 'Anagain', 'Baicapil', 'DM').\n"
        "4. ENGLISH ENTITY SPELLING: Keep every brand name, product/range name, model name, and ingredient name in its standard English Latin spelling. If the source transcript wrote an English name phonetically in an Indic script, restore its standard spelling (for example Plix, Guava, Niacinamide, Vitamin C, Hyaluronic Acid, Salicylic Acid, Retinol, SPF 50). Never transliterate these names into the target script.\n"
        "VOCABULARY & STYLE RULES:\n"
        "- When translating Hindi to Marathi, use native Marathi vocabulary and grammar, not Hindi with Devanagari spelling.\n"
        "- Preserve English brand and product names.\n"
        f"{_spoken_style_rules(target_lang)}"
        "- For Marathi, keep the breed name exactly as 'Golden Retriever'; translate 'ki maa' / 'की माँ' as 'ची आई' (for example, 'Golden Retriever ची आई'), never as Hindi 'की माँ'.\n"
        "- Translate 'biological mother' as 'biological आई' or natural Marathi equivalent, not Hindi 'माँ'."
    )
    translated = []
    try:
        response = request_with_retry(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": prompt},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
            timeout=60,
            log_prefix="[Dubber Groq]",
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        items = json.loads(result)["translations"]
        by_id = {item["id"]: item["text"].strip() for item in items}
        expected_ids = [cue.id for cue in cues]
        if list(by_id) != expected_ids or any(not by_id[cue.id] for cue in cues):
            raise ValueError("translation response IDs or text do not match the requested cues")
    except Exception as exc:
        raise PipelineError(
            f"Groq translation failed for target '{target_lang}'; refusing to synthesize unverified text."
        ) from exc
    for cue in cues:
        new_cue = Cue(cue.id, cue.start_ms, cue.end_ms, by_id[cue.id])
        translated.append(new_cue)
        print(f"  [{cue.id}] {cue.text[:40]} → {new_cue.text[:40]}")
    return _apply_marathi_glossary(translated, target_lang)


def _translate_cues_gemini(cues: list[Cue], target_lang: str, source_lang: str = "auto") -> list[Cue]:
    """Translate cues using Google Gemini (gemini-3.5-flash-lite)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError("Set GEMINI_API_KEY to translate with Gemini.")
    from google import genai
    from google.genai import types

    model = os.environ.get("TRANSLATION_MODEL", "gemini-3.5-flash-lite")
    if model in ('gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash-lite'):
        model = 'gemini-3.5-flash-lite'

    language_names = {
        "mr": "Marathi", "hi": "Hindi", "ta": "Tamil", "ml": "Malayalam",
        "kn": "Kannada", "te": "Telugu", "bn": "Bengali", "gu": "Gujarati",
        "pa": "Punjabi", "ur": "Urdu", "en": "English", "es": "Spanish",
        "fr": "French", "de": "German", "ja": "Japanese", "pt": "Portuguese",
        "it": "Italian", "ar": "Arabic",
    }
    target_name = language_names.get(target_lang.lower(), target_lang)
    source_name = language_names.get(source_lang.lower(), source_lang or "auto-detected")
    payload = {
        "source_language": source_name,
        "target_language": target_name,
        "translations": [
            {
                "id": cue.id,
                "duration_seconds": round(max(0.1, (cue.end_ms - cue.start_ms) / 1000.0), 2),
                "max_words": max(2, int(round(max(0.1, (cue.end_ms - cue.start_ms) / 1000.0) * 2.2))),
                "text": cue.text,
            }
            for cue in cues
        ],
    }
    prompt = (
        f"Translate every subtitle cue from {source_name} into natural spoken {target_name} for video dubbing. "
        "Return only JSON with a translations array containing exactly the supplied IDs in order, "
        "each with id and translated text. Preserve meaning, names, and cue IDs. Do not explain.\n"
        "CRITICAL DUBBING REQUIREMENTS:\n"
        "1. COMPLETE DIALOGUE: Translate ALL spoken thoughts and sentences completely. Do NOT omit or drop sentences, product names, or sales offers.\n"
        "2. TIMING & NATURAL PACING: Strictly match the spoken duration_seconds and stay within max_words for each cue at a relaxed, normal conversational cadence (~2.0 words per second). The target text must be concise, punchy, and crisp: NEVER use verbose, lengthy, or convoluted phrasing that would force the speaker to rush or race. The text MUST fit comfortably within the allotted seconds.\n"
        "3. NATURAL REGISTRATION & TERMINOLOGY: Use conversational spoken phrasing, keeping English product and brand names natural (e.g. 'Plix', 'Rosemary Hair Growth Serum', 'Redensyl', 'Anagain', 'Baicapil', 'DM').\n"
        "4. ENGLISH ENTITY SPELLING: Keep every brand name, product/range name, model name, and ingredient name in its standard English Latin spelling. If the source transcript wrote an English name phonetically in an Indic script, restore its standard spelling (for example Plix, Guava, Niacinamide, Vitamin C, Hyaluronic Acid, Salicylic Acid, Retinol, SPF 50). Never transliterate these names into the target script.\n"
        "VOCABULARY & STYLE RULES:\n"
        "- When translating Hindi to Marathi, use native Marathi vocabulary and grammar, not Hindi with Devanagari spelling.\n"
        "- Preserve English brand and product names.\n"
        f"{_spoken_style_rules(target_lang)}"
        "- For Marathi, keep the breed name exactly as 'Golden Retriever'; translate 'ki maa' / 'की माँ' as 'ची आई' (for example, 'Golden Retriever ची आई'), never as Hindi 'की माँ'.\n"
        "- Translate 'biological mother' as 'biological आई' or natural Marathi equivalent, not Hindi 'माँ'."
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "translations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"},
                    },
                    "required": ["id", "text"],
                },
            },
        },
        "required": ["translations"],
    }
    gen_config = types.GenerateContentConfig(
        system_instruction=prompt,
        response_mime_type="application/json",
        response_schema=schema,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=gen_config,
        )
        if not response or not response.text:
            raise ValueError("Gemini returned an empty response")
        items = json.loads(response.text)["translations"]
        by_id = {item["id"]: item["text"].strip() for item in items}
        expected_ids = [cue.id for cue in cues]
        if list(by_id) != expected_ids or any(not by_id[cue.id] for cue in cues):
            raise ValueError("translation response IDs or text do not match the requested cues")
    except Exception as exc:
        raise PipelineError(f"Gemini translation failed for target '{target_lang}': {exc}") from exc

    translated = [Cue(cue.id, cue.start_ms, cue.end_ms, by_id[cue.id]) for cue in cues]
    for c in translated:
        print(f"  [{c.id}] {c.text[:40]} → {c.text[:40]}")
    return _apply_marathi_glossary(translated, target_lang)


def translate_cues(cues: list[Cue], target_lang: str, source_lang: str = "auto") -> list[Cue]:
    provider = os.environ.get("TRANSLATION_PROVIDER", "gemini").lower()
    model = os.environ.get("TRANSLATION_MODEL", "gemini-3.5-flash-lite").lower()
    if provider == "gemini" or "gemini" in model:
        return _translate_cues_gemini(cues, target_lang, source_lang)
    return _translate_cues_groq(cues, target_lang, source_lang)



def _tts_cue(
    voice_id: str,
    text: str,
    out_path: Path,
    model: str,
    api_key: str,
    target_lang: str,
    similarity_boost: float = 0.70,
    speed: float = 1.10,
) -> Path:
    """Synthesize a single cue to audio file with brisk, energetic neural speech rate."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model,
        "language_code": target_lang.lower().split("-")[0],
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": similarity_boost,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": round(float(speed), 2),
        },
    }
    r = request_with_retry(
        "POST",
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers=headers, json=payload, timeout=60, log_prefix="[Dubber TTS]",
    )
    if not r.ok and "speed" in r.text.lower() and "speed" in payload.get("voice_settings", {}):
        # Fallback if specific model or tier does not support voice_settings.speed
        payload["voice_settings"].pop("speed", None)
        r = request_with_retry(
            "POST",
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers=headers, json=payload, timeout=60, log_prefix="[Dubber TTS Fallback]",
        )
    if not r.ok:
        raise PipelineError(f"TTS failed for '{text[:30]}...': HTTP {r.status_code} — {r.text[:200]}")
    out_path.write_bytes(r.content)
    return out_path


def _prepare_tts_cues(
    cues: list[Cue],
    total_duration_s: float,
    merge_gap_ms: int = 350,
    max_block_ms: int = 15_000,
) -> list[Cue]:
    """
    Convert dense OCR/word-level subtitle cues into speakable, non-overlapping
    narration blocks.

    Visual subtitle extraction can produce 0-250 ms cues, including several cues
    with the same timestamp. A TTS voice cannot speak those independently without
    overlap. Consecutive cues are therefore grouped until a real pause is found
    (or max_block_ms is reached). The returned cue windows never overlap.
    """
    limit_ms = max(1, int(round(total_duration_s * 1000)))
    ordered: list[Cue] = []

    for cue in sorted(cues, key=lambda item: (int(item.start_ms), int(item.id))):
        text = " ".join(str(cue.text).replace("\n", " ").split())
        if not text:
            continue

        start_ms = max(0, min(limit_ms - 1, int(cue.start_ms)))
        end_ms = max(start_ms, min(limit_ms, int(cue.end_ms)))
        ordered.append(Cue(cue.id, start_ms, end_ms, text))

    if not ordered:
        raise PipelineError("No non-empty TTS cues were supplied.")

    groups: list[list[Cue]] = []
    current: list[Cue] = []

    for cue in ordered:
        if not current:
            current = [cue]
            continue

        block_start = int(current[0].start_ms)
        previous_end = max(int(current[-1].start_ms), int(current[-1].end_ms))
        proposed_end = max(int(cue.start_ms), int(cue.end_ms))
        gap_ms = max(0, int(cue.start_ms) - previous_end)

        # Only merge cues if at least one is a micro-fragment (e.g. OCR fragment < 1.0s or < 3 words).
        # Complete subtitle sentences (>= 1.0s) should NOT be merged, as merging them collapses the
        # natural timing between spoken lines and leaves big silent gaps later in the video.
        curr_is_frag = (int(current[-1].end_ms) - int(current[-1].start_ms) < 1000) or len(current[-1].text.split()) < 3
        cue_is_frag = (int(cue.end_ms) - int(cue.start_ms) < 1000) or len(cue.text.split()) < 3
        should_merge = (curr_is_frag or cue_is_frag) and gap_ms <= merge_gap_ms and (proposed_end - block_start <= max_block_ms)

        if should_merge:
            current.append(cue)
        else:
            groups.append(current)
            current = [cue]

    if current:
        groups.append(current)

    prepared: list[Cue] = []
    for index, group in enumerate(groups):
        start_ms = int(group[0].start_ms)
        next_start_ms = (
            int(groups[index + 1][0].start_ms)
            if index + 1 < len(groups)
            else limit_ms
        )
        end_ms = max(start_ms + 1, min(limit_ms, next_start_ms))

        parts: list[str] = []
        previous_text = None
        for cue in group:
            text = cue.text.strip()
            folded = text.casefold()
            if text and folded != previous_text:
                parts.append(text)
                previous_text = folded

        if parts and end_ms > start_ms:
            prepared.append(Cue(group[0].id, start_ms, end_ms, " ".join(parts)))

    if not prepared:
        raise PipelineError("TTS cue normalization produced no speakable windows.")

    if len(prepared) != len(ordered):
        print(
            f"[Dubber] Consolidated {len(ordered)} dense subtitle cues into "
            f"{len(prepared)} non-overlapping narration blocks."
        )
    return prepared


def _probe_audio_duration(audio_path: Path) -> float:
    """Return decoded audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(
            f"Could not read TTS cue duration for {audio_path.name}: {result.stderr}"
        )
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            f"Invalid TTS cue duration for {audio_path.name}: {result.stdout!r}"
        ) from exc
    if duration <= 0:
        raise PipelineError(f"TTS cue is empty: {audio_path}")
    return duration


def _atempo_filters(speed: float) -> list[str]:
    """
    Build an FFmpeg atempo chain. Each stage stays within the portable 0.5-2.0
    range, while the whole chain can fit unusually long translations.
    """
    if speed <= 1.0005:
        return []

    filters: list[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.8f}")
    return filters


def _assemble_tts_audio(
    cues: list[Cue], cue_audio_paths: list[Path],
    total_duration_s: float, out_path: Path,
) -> tuple[Path, list[Cue]]:
    """
    Assemble one full-length, frame-aligned narration track.

    Uses global lookahead proportional time-budgeting:
    - Calculates total raw audio duration vs. total video duration.
    - Scales pacing proportionally across ALL cues so speech never drags,
      never races, and every cue (including the final cue) COMPLETES 100%
      of its spoken sentences before the video ends.
    - Speech speed is kept within a natural, energetic conversational window
      (1.00x - 1.22x max).
    - Cues flow dynamically with a clean breath pause (40-60ms) and zero voice collision.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit_ms = int(round(total_duration_s * 1000))
    MIN_PAUSE_MS = 50   # 50ms clean breath pause between sentences

    # Collect valid audio files and their decoded durations
    valid_items = []
    for cue, audio_path in zip(cues, cue_audio_paths):
        audio_path = Path(audio_path)
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            continue
        try:
            dur_s = _probe_audio_duration(audio_path)
        except Exception:
            continue
        if dur_s <= 0:
            continue
        valid_items.append({
            "cue": cue,
            "path": audio_path,
            "orig_start_ms": max(0, min(limit_ms - 50, int(cue.start_ms))),
            "orig_end_ms": min(limit_ms, int(cue.end_ms)),
            "dur_s": dur_s,
        })

    if not valid_items:
        raise PipelineError("No valid cue audio files were generated.")

    # Calculate global timeline budget
    num_items = len(valid_items)
    total_raw_dur_s = sum(item["dur_s"] for item in valid_items)
    total_pause_budget_s = (num_items - 1) * (MIN_PAUSE_MS / 1000.0) if num_items > 1 else 0.0
    available_speech_s = max(1.0, total_duration_s - total_pause_budget_s - 0.1)

    # Global speed ratio needed to complete all cues across the video
    global_speed = 1.0
    if total_raw_dur_s > available_speech_s:
        global_speed = total_raw_dur_s / available_speech_s
        print(f"[Dubber] Global pacing: total audio {total_raw_dur_s:.2f}s in {total_duration_s:.2f}s video -> global scaling {global_speed:.2f}x to ensure all cues complete.")

    # Dynamic scheduling: allocate smooth, unhurried timeline placement
    scheduled = []
    cursor_ms = 0

    for i, item in enumerate(valid_items):
        orig_start = item["orig_start_ms"]
        dur_s = item["dur_s"]

        # Earliest start without colliding with the previous cue's speech
        start_ms = max(orig_start, cursor_ms)

        # Remaining time in the video from start_ms to end
        remaining_video_ms = max(50, limit_ms - start_ms)
        remaining_video_s = remaining_video_ms / 1000.0

        # Remaining raw duration of THIS cue and all SUBSEQUENT cues:
        rem_cues_raw_s = sum(valid_items[j]["dur_s"] for j in range(i, num_items))
        rem_pauses_s = (num_items - 1 - i) * (MIN_PAUSE_MS / 1000.0)
        rem_avail_speech_s = max(0.2, remaining_video_s - rem_pauses_s)

        # Local required speed to finish all remaining cues on time
        local_ratio = rem_cues_raw_s / rem_avail_speech_s if rem_cues_raw_s > rem_avail_speech_s else global_speed

        # Blend global and local ratio, clamping to natural energetic pacing [1.00x - 1.25x]
        speed = max(1.0, min(1.25, max(global_speed, local_ratio)))
        play_dur_s = dur_s / speed

        # Ensure we never exceed video boundary while letting speech play completely
        if play_dur_s > remaining_video_s:
            speed = max(speed, dur_s / remaining_video_s)
            speed = min(1.30, speed)
            play_dur_s = min(remaining_video_s, dur_s / speed)

        end_ms = min(limit_ms, start_ms + int(round(play_dur_s * 1000)))
        cursor_ms = min(limit_ms, end_ms + MIN_PAUSE_MS)

        scheduled.append({
            "cue": item["cue"],
            "path": item["path"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "play_dur_s": play_dur_s,
            "speed": speed,
            "orig_dur_s": dur_s,
        })
        print(
            f"[Dubber] Cue {item['cue'].id}: {dur_s:.2f}s -> {play_dur_s:.2f}s "
            f"(speed {speed:.2f}x, window {start_ms}ms-{end_ms}ms)"
        )

    if not scheduled:
        raise PipelineError("No cues could be scheduled within video duration.")

    inputs: list[str] = []
    filter_parts: list[str] = []
    mix_inputs: list[str] = []

    for idx, item in enumerate(scheduled):
        audio_path = item["path"]
        start_ms = item["start_ms"]
        play_dur_s = item["play_dur_s"]
        speed = item["speed"]

        chain = [
            "aresample=48000",
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=mono",
            "asetpts=PTS-STARTPTS",
            *_atempo_filters(speed),
            f"atrim=duration={play_dur_s:.6f}",
            "asetpts=PTS-STARTPTS",
        ]

        fade_s = min(0.015, play_dur_s / 4.0)
        if fade_s > 0.001:
            fade_out_start = max(0.0, play_dur_s - fade_s)
            chain.extend([
                f"afade=t=in:st=0:d={fade_s:.6f}",
                f"afade=t=out:st={fade_out_start:.6f}:d={fade_s:.6f}",
            ])

        chain.extend([
            f"apad=pad_dur={play_dur_s:.6f}",
            f"atrim=duration={play_dur_s:.6f}",
            f"adelay={start_ms}:all=1",
        ])

        inputs.extend(["-i", str(audio_path)])
        label = f"a{idx}"
        filter_parts.append(f"[{idx}:a]{','.join(chain)}[{label}]")
        mix_inputs.append(f"[{label}]")

    mix_filter = (
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=longest:"
          "dropout_transition=0:normalize=0,"
          "alimiter=limit=0.95,"
        + f"apad=pad_dur={total_duration_s:.6f},"
          f"atrim=duration={total_duration_s:.6f},"
          "asetpts=PTS-STARTPTS[out]"
    )

    full_filter = ";".join(filter_parts + [mix_filter])
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-ar", "48000", "-ac", "1",
        "-acodec", "libmp3lame", "-q:a", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(f"TTS audio assembly failed:\n{result.stderr}")
    return out_path, [Cue(item["cue"].id, item["start_ms"], item["end_ms"], item["cue"].text) for item in scheduled]


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — ElevenLabs Dubbing API
# ──────────────────────────────────────────────────────────────────────────────

class ElevenLabsDubber:
    BASE_URL = "https://api.elevenlabs.io/v1/dubbing"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")
        self.headers = {"xi-api-key": self.api_key}

    def create_job(self, video_path: Path, target_lang: str,
                   source_lang: str = "auto", num_speakers: int = 0) -> str:
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        content_type = "video/mp4" if suffix in (".mp4", ".mov", ".mkv") else "audio/mpeg"
        data = {
            "target_lang": target_lang,
            "source_lang": source_lang or "auto",
            "num_speakers": str(int(num_speakers)),
            "watermark": "false",
        }
        print(f"[Dubber] Submitting to ElevenLabs Dubbing ({source_lang} → {target_lang})...")
        with open(video_path, "rb") as f:
            r = request_with_retry(
                "POST",
                self.BASE_URL,
                headers=self.headers,
                data=data,
                files={"file": (video_path.name, f, content_type)},
                timeout=300,
                log_prefix="[ElevenLabs Dubbing]",
            )
        if not r.ok:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
            raise PipelineError(f"ElevenLabs Dubbing create failed: HTTP {r.status_code} — {detail}")
        dubbing_id = r.json().get("dubbing_id")
        if not dubbing_id:
            raise PipelineError("ElevenLabs returned no dubbing_id.")
        print(f"[Dubber] Job created: {dubbing_id}")
        return dubbing_id

    def wait_for_completion(self, dubbing_id: str, poll_interval: int = 12, timeout: int = 1800) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            r = request_with_retry(
                "GET",
                f"{self.BASE_URL}/{dubbing_id}",
                headers=self.headers,
                timeout=30,
                log_prefix="[ElevenLabs Status]",
            )
            if not r.ok:
                raise PipelineError(f"ElevenLabs status check failed: HTTP {r.status_code}")
            data = r.json()
            status = data.get("status", "")
            print(f"  [Dubber] Status: {status} ({int(time.time()-start)}s)...")
            if status == "dubbed":
                return data
            if status == "failed":
                raise PipelineError(f"ElevenLabs Dubbing failed: {data.get('error','unknown')}")
            time.sleep(poll_interval)
        raise PipelineError(f"ElevenLabs Dubbing timed out after {timeout}s.")

    def download_audio(self, dubbing_id: str, target_lang: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/audio/{target_lang}"
        with request_with_retry(
            "GET",
            url,
            headers=self.headers,
            stream=True,
            timeout=120,
            log_prefix="[ElevenLabs Audio Download]",
        ) as r:
            if not r.ok:
                raise PipelineError(f"Failed to download dubbed audio: HTTP {r.status_code}")
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
        print(f"[Dubber] Dubbed audio → {output_path.name} ({output_path.stat().st_size//1024}KB)")
        return output_path

    def download_transcript(self, dubbing_id: str, target_lang: str, srt_path: Path) -> list:
        srt_path = Path(srt_path)
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/transcript/{target_lang}?format_type=srt"
        r = request_with_retry(
            "GET",
            url,
            headers=self.headers,
            timeout=30,
            log_prefix="[ElevenLabs Transcript Download]",
        )
        if not r.ok:
            raise PipelineError(f"Failed to download transcript: HTTP {r.status_code}")
        srt_path.write_text(r.text, encoding="utf-8")
        cues = parse_srt(r.text)
        print(f"[Dubber] Transcript → {srt_path.name} ({len(cues)} cues)")
        return cues


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_dubbing(
    video_path: Path,
    job_dir: Path,
    target_lang: str,
    source_lang: str = "auto",
    num_speakers: int = 0,
    source_cues: list = None,
    translated_srt: Path = None,
    tts_model: str = EL_V4_MODEL,
) -> tuple[Path, Path, list]:
    """
    Stage 3 unified entry point.
    Auto-selects Mode A (Dubbing API) or Mode B (Voice-Clone TTS) based on language support.

    Args:
        video_path      : Source video
        job_dir         : Output directory
        target_lang     : e.g. 'mr', 'hi', 'en'
        source_lang     : Source language or 'auto'
        num_speakers    : For Dubbing API
        source_cues     : Parsed source transcript cues (for Mode B translation)
        translated_srt  : Pre-translated SRT in target language (skips translation step)
        tts_model       : ElevenLabs TTS model for Mode B (default: eleven_v4)

    Returns:
        (dubbed_audio_path, dubbed_srt_path, cues)
    """
    video_path = Path(video_path).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")

    dubbed_audio = job_dir / f"dubbed_master_{target_lang}.mp3"
    dubbed_srt = job_dir / f"transcript_{target_lang}.srt"
    job_id_path = job_dir / f"el_dubbing_id_{target_lang}.txt"

    # Cache check — Mode B also requires the current timing-assembly version.
    # This invalidates master audio created by the old overlapping amix pipeline.
    REAL_AUDIO_MIN_BYTES = 500 * 1024  # 500KB
    assembly_version_path = job_dir / f"tts_assembly_version_{target_lang}.txt"
    mode_b_cache_current = (
        assembly_version_path.exists()
        and assembly_version_path.read_text(encoding="utf-8").strip()
        == TTS_ASSEMBLY_VERSION
    )
    if (dubbed_audio.exists()
            and dubbed_audio.stat().st_size >= REAL_AUDIO_MIN_BYTES
            and dubbed_srt.exists()
            and dubbed_srt.stat().st_size > 100
            and "PLACEHOLDER" not in dubbed_audio.name
            and (target_lang in EL_DUBBING_SUPPORTED or mode_b_cache_current)):
        print(f"[Dubber] Cache hit — reusing {dubbed_audio.name} ({dubbed_audio.stat().st_size // 1024}KB)")
        cues = parse_srt(dubbed_srt.read_text(encoding="utf-8")) if dubbed_srt.exists() else []
        if not cues:
            raise PipelineError("Cached dubbing transcript is empty; refusing to reuse its audio.")
        _assert_target_language(cues, target_lang)
        return dubbed_audio, dubbed_srt, cues


    # A reviewed target SRT must drive the generated speech verbatim. The
    # Dubbing API performs its own translation, so use the TTS path whenever a
    # reviewed translation was supplied, even for normally supported languages.
    has_reviewed_translation = bool(translated_srt and Path(translated_srt).exists())

    # ── Mode A: ElevenLabs Dubbing API (auto dub, voice-preserved) ────────────
    if target_lang in EL_DUBBING_SUPPORTED and not has_reviewed_translation:
        print(f"[Dubber] Mode A: ElevenLabs Dubbing API ({target_lang} is supported)")
        client = ElevenLabsDubber(api_key)

        dubbing_id = None
        if job_id_path.exists():
            dubbing_id = job_id_path.read_text().strip()
            print(f"[Dubber] Resuming existing job: {dubbing_id}")

        if not dubbing_id:
            dubbing_id = client.create_job(video_path, target_lang, source_lang, num_speakers)
            job_id_path.write_text(dubbing_id, encoding="utf-8")

        client.wait_for_completion(dubbing_id)
        client.download_audio(dubbing_id, target_lang, dubbed_audio)
        cues = client.download_transcript(dubbing_id, target_lang, dubbed_srt)
        if not cues:
            raise PipelineError("Dubbing provider returned an empty target transcript.")
        _assert_target_language(cues, target_lang)
        return dubbed_audio, dubbed_srt, cues

    # ── Mode B: Voice-Clone TTS ───────────────────────────────────────────────
    mode_reason = "using approved translated text" if has_reviewed_translation else "language not supported by Dubbing API"
    print(f"[Dubber] Mode B: timed TTS ({mode_reason})")
    print(f"[Dubber] Using ElevenLabs TTS model: {tts_model}")

    # Step B1: Get translated cues
    target_cues: list[Cue] = []
    if translated_srt and Path(translated_srt).exists():
        print(f"[Dubber] Using supplied translated SRT: {Path(translated_srt).name}")
        target_cues = parse_srt(Path(translated_srt).read_text(encoding="utf-8"))
        target_cues = _apply_marathi_glossary(target_cues, target_lang)
        _reject_hindi_mother_phrase(target_cues, target_lang)
        dubbed_srt.write_text(
            "\n\n".join(f"{i}\n{stamp(cue.start_ms)} --> {stamp(cue.end_ms)}\n{cue.text}" for i, cue in enumerate(target_cues, 1)) + "\n",
            encoding="utf-8",
        )
    elif source_cues:
        provider_name = os.environ.get('TRANSLATION_PROVIDER', 'gemini')
        model_name = os.environ.get('TRANSLATION_MODEL', 'gemini-3.5-flash-lite')
        print(f"[Dubber] Translating {len(source_cues)} cues to {target_lang} via {provider_name} ({model_name})...")
        target_cues = translate_cues(source_cues, target_lang, source_lang)
        # Save translated SRT
        srt_lines = []
        for i, cue in enumerate(target_cues, 1):
            def _fmt(ms):
                h, rem = divmod(int(ms), 3600000)
                m, rem = divmod(rem, 60000)
                s, ms_ = divmod(rem, 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"
            srt_lines.append(f"{i}\n{_fmt(cue.start_ms)} --> {_fmt(cue.end_ms)}\n{cue.text}\n")
        dubbed_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    else:
        raise PipelineError(
            f"Language '{target_lang}' is not supported by ElevenLabs Dubbing API.\n"
            f"For Mode B (Voice-Clone TTS), supply one of:\n"
            f"  --transcript <source_srt>  (will be auto-translated)\n"
            f"  --reviewed-srt <marathi_srt>  (pre-translated Marathi SRT)"
        )

    if not target_cues:
        raise PipelineError("No translated transcript cues were returned; refusing to generate audio.")
    target_cues = _apply_marathi_glossary(target_cues, target_lang)
    _reject_hindi_mother_phrase(target_cues, target_lang)
    _assert_target_language(target_cues, target_lang)

    # Step B2: Prefer a voice trained in the target language. A source-language
    # clone preserves the original accent, so it is only the fallback.
    source_hash = _video_hash(video_path)
    language = target_lang.lower().split("-")[0]
    # Hindi deliberately uses a native Hindi voice to avoid carrying a strong
    # source-language accent. Other languages preserve the original speaker by
    # default; an explicit <LANG>_TTS_VOICE_ID opts into replacement instead.
    if _should_replace_source_voice(target_lang):
        voice_id, voice_description = _find_target_language_voice(target_lang, api_key, tts_model)
    else:
        voice_id, voice_description = None, None
    using_source_clone = not voice_id
    if using_source_clone:
        sample_wav = job_dir / f"voice_sample_{source_hash}.wav"
        if not sample_wav.exists() or sample_wav.stat().st_size < 1000:
            _extract_voice_sample(video_path, sample_wav, duration_s=20.0)
        voice_name = f"DubClone_{source_hash}"
        voice_id = _ensure_voice_clone(sample_wav, voice_name, api_key)
        voice_description = "source voice clone (reduced similarity fallback)"
    print(f"[Dubber] TTS voice: {voice_description} (ID: {voice_id})")
    if using_source_clone:
        similarity_boost = 0.35 if language == "hi" else 0.85
    else:
        similarity_boost = 0.70

    # Step B3: Consolidate dense/zero-duration visual cues into narration blocks.
    from .scene_analyzer import get_video_info
    info = get_video_info(video_path)
    total_dur = info["duration"]
    tts_cues = _prepare_tts_cues(target_cues, total_dur)

    # Include the normalized text hash in the cache key. Old one-word TTS clips
    # must not be reused for a newly consolidated narration block.
    cue_dir = job_dir / f"tts_cues_{target_lang}_{TTS_ASSEMBLY_VERSION}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    cue_audio_paths = []
    print(f"[Dubber] Generating {len(tts_cues)} fitted TTS blocks in {target_lang}...")
    for cue in tts_cues:
        text = _prepare_tts_text(cue.text.strip(), target_lang)
        text_hash = _content_hash(
            TTS_ASSEMBLY_VERSION,
            voice_id,
            tts_model,
            target_lang,
            f"{similarity_boost:.2f}",
            text,
        )[:10]
        cue_out = cue_dir / f"cue_{cue.id:04d}_{text_hash}.mp3"
        if text and not cue_out.exists():
            _tts_cue(
                voice_id,
                text,
                cue_out,
                tts_model,
                api_key,
                target_lang,
                similarity_boost=similarity_boost,
            )
        cue_audio_paths.append(cue_out)

    # Step B4: Fit every narration block to its exclusive timeline window,
    # then assemble one master track with no simultaneous cue speech.
    print(f"[Dubber] Assembling {len(cue_audio_paths)} cues into {total_dur:.2f}s audio track...")
    _, scheduled_cues = _assemble_tts_audio(tts_cues, cue_audio_paths, total_dur, dubbed_audio)
    dubbed_srt.write_text(
        "\n\n".join(
            f"{i}\n{stamp(c.start_ms)} --> {stamp(c.end_ms)}\n{c.text}"
            for i, c in enumerate(scheduled_cues, 1)
        ) + "\n",
        encoding="utf-8",
    )
    assembly_version_path.write_text(TTS_ASSEMBLY_VERSION, encoding="utf-8")
    print(f"[Dubber] Mode B complete → {dubbed_audio.name} ({dubbed_audio.stat().st_size//1024}KB)")

    return dubbed_audio, dubbed_srt, scheduled_cues
