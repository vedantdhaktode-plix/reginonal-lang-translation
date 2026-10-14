"""Gemini model selection with fallbacks.

Google retires models (404 NOT_FOUND) and overloads others (503), so every Gemini call
goes through an ordered model chain: the configured model first, then
GEMINI_FALLBACK_MODELS. Models that return 404 are skipped for the rest of the process.
"""
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_FALLBACKS = 'gemini-3.8-flash,gemini-3.5-flash,gemini-3.5-flash-lite'
_retired: set[str] = set()


def model_chain(preferred: str | None) -> list[str]:
    fallbacks = os.getenv('GEMINI_FALLBACK_MODELS', DEFAULT_FALLBACKS)
    chain = []
    for name in [preferred or '', *fallbacks.split(',')]:
        name = name.strip()
        if name and name not in chain and name not in _retired:
            chain.append(name)
    return chain


def _is_not_found(exc) -> bool:
    return getattr(exc, 'code', None) == 404 or getattr(exc, 'status_code', None) == 404 or 'NOT_FOUND' in str(exc)


def note_failure(name: str, exc: Exception) -> None:
    """Records a retired model so later calls skip it."""
    if _is_not_found(exc):
        _retired.add(name)


def generate_content(client, preferred, purpose, **kwargs):
    """Calls client.models.generate_content on each model in the chain until one returns text."""
    last_exc = None
    for name in model_chain(preferred):
        try:
            response = client.models.generate_content(model=name, **kwargs)
            if response is not None and response.text:
                return response
            logger.warning('Gemini model %s returned an empty response for %s', name, purpose)
        except Exception as exc:
            last_exc = exc
            note_failure(name, exc)
            if _is_not_found(exc):
                logger.warning('Gemini model %s is unavailable (404); skipping it from now on', name)
            else:
                logger.warning('Gemini model %s failed for %s: %s', name, purpose, exc)
    if last_exc is not None:
        raise last_exc
    raise ValueError(f'No Gemini model returned a response for {purpose}')
