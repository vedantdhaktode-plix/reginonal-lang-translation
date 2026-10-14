import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class PipelineError(Exception):
    """An actionable, safe-to-display job error."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows previewers/scanners can briefly hold the destination open.
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_env(path):
    # Existing process variables take priority; deliberately no shell evaluation.
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def request_json(url, payload=None, headers=None, attempts=3):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", **(headers or {})})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # Do not echo provider bodies, which may contain credentials or source text.
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise PipelineError(f"Provider returned HTTP {error.code}; check account, settings and limits.") from None
            time.sleep(min(2 ** attempt, 8))
        except (urllib.error.URLError, TimeoutError):
            # POST timeout may already have incurred a charge; avoid blind paid retries.
            raise PipelineError("Provider connection failed. Resume after checking connectivity; a timed-out request may have been processed.") from None
        except (ValueError, UnicodeError):
            raise PipelineError("Provider returned invalid JSON.") from None


def request_with_retry(
    method: str,
    url: str,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
    retry_statuses: tuple = (429, 500, 502, 503, 504),
    log_prefix: str = "[HTTP]",
    **kwargs
):
    """
    Execute an HTTP request with exponential backoff on transient errors:
    - ConnectionError (RemoteDisconnected, ConnectionResetError, BrokenPipe, etc.)
    - Timeout (ReadTimeout, ConnectTimeout)
    - ChunkedEncodingError
    - HTTP 429, 500, 502, 503, 504
    Automatically rewinds file handles in `files` and `data` if retried.
    """
    import requests
    from requests.exceptions import ConnectionError as ReqConnectionError, Timeout as ReqTimeout, ChunkedEncodingError

    last_exc = None
    last_resp = None
    timeout = kwargs.pop("timeout", 45)

    for attempt in range(max_retries):
        # Rewind any file pointers before each attempt so retries don't send 0 bytes
        if "files" in kwargs and kwargs["files"]:
            files_dict = kwargs["files"]
            if isinstance(files_dict, dict):
                for item in files_dict.values():
                    if isinstance(item, tuple) and len(item) >= 2 and hasattr(item[1], "seek"):
                        try:
                            item[1].seek(0)
                        except Exception:
                            pass
                    elif hasattr(item, "seek"):
                        try:
                            item.seek(0)
                        except Exception:
                            pass

        if "data" in kwargs and hasattr(kwargs["data"], "seek"):
            try:
                kwargs["data"].seek(0)
            except Exception:
                pass

        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in retry_statuses:
                last_resp = resp
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    print(f"{log_prefix} {method} {url} returned HTTP {resp.status_code}. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
            return resp
        except (ReqConnectionError, ReqTimeout, ChunkedEncodingError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"{log_prefix} {method} {url} connection dropped ({type(exc).__name__}: {exc}). Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
                continue
            raise PipelineError(f"HTTP request failed after {max_retries} attempts: {type(exc).__name__}: {exc}") from exc

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise PipelineError(f"HTTP request failed after {max_retries} attempts: {last_exc}") from last_exc

