import os
import time
from pathlib import Path
import requests
from .common import PipelineError, request_with_retry
from .srt_parser import parse_srt


class ElevenLabsDubbingClient:
    """
    Client for ElevenLabs Dubbing API (distinct from TTS).
    Handles creating dubbing jobs, polling status, and downloading dubbed audio + transcript.
    """

    def __init__(self, api_key=None, base_url="https://api.elevenlabs.io/v1/dubbing"):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise PipelineError("Set ELEVENLABS_API_KEY in .env or your environment.")
        self.base_url = base_url.rstrip("/")
        self.headers = {"xi-api-key": self.api_key}

    def create_dub_job(
        self,
        media_path,
        target_lang,
        source_lang="auto",
        num_speakers=0,
        watermark=False
    ):
        """
        Submits a video or audio file to the ElevenLabs Dubbing API.
        Returns the dubbing_id.
        """
        media_path = Path(media_path).resolve()
        if not media_path.exists():
            raise PipelineError(f"Media file not found: {media_path}")

        url = self.base_url
        data = {
            "target_lang": target_lang,
            "source_lang": source_lang or "auto",
            "num_speakers": int(num_speakers),
            "watermark": "true" if watermark else "false"
        }

        content_type = "video/mp4" if media_path.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm"] else "audio/wav"
        try:
            with open(media_path, "rb") as f:
                files = {"file": (media_path.name, f, content_type)}
                response = request_with_retry("POST", url, headers=self.headers, data=data, files=files, timeout=120, log_prefix="[Dubbing Client]")
        except Exception as e:
            raise PipelineError(f"Network error while submitting to ElevenLabs Dubbing: {e}") from None

        if response.status_code != 200:
            msg = f"ElevenLabs Dubbing API returned HTTP {response.status_code}"
            try:
                err_detail = response.json().get("detail", "")
                if err_detail:
                    msg += f": {err_detail}"
            except Exception:
                pass
            raise PipelineError(msg)

        try:
            res_json = response.json()
            dubbing_id = res_json.get("dubbing_id")
            if not dubbing_id:
                raise ValueError("Missing dubbing_id in response")
            return dubbing_id
        except (ValueError, KeyError):
            raise PipelineError("Invalid response JSON from ElevenLabs Dubbing API.") from None

    def get_status(self, dubbing_id):
        """
        Queries status of the dubbing job.
        Returns dict with keys: 'status' ('dubbing', 'dubbed', 'failed', 'not_found'), 'error', etc.
        """
        url = f"{self.base_url}/{dubbing_id}"
        try:
            response = request_with_retry("GET", url, headers=self.headers, timeout=30, log_prefix="[Dubbing Client Status]")
        except Exception as e:
            raise PipelineError(f"Failed to check ElevenLabs Dubbing status: {e}") from None

        if response.status_code == 404:
            return {"status": "not_found", "error": "Job ID not found on ElevenLabs."}

        if response.status_code != 200:
            raise PipelineError(f"ElevenLabs Dubbing status check failed with HTTP {response.status_code}.")

        try:
            return response.json()
        except ValueError:
            raise PipelineError("Invalid JSON received when checking ElevenLabs Dubbing status.")

    def wait_for_completion(self, dubbing_id, poll_interval=10, timeout=1200, callback=None):
        """
        Polls until dubbing status is 'dubbed' or fails.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(dubbing_id)
            status = status_data.get("status")
            elapsed = int(time.time() - start_time)
            print(f"  [ElevenLabs] Status: {status} ({elapsed}s elapsed)...")
            if callback:
                callback(status, status_data)

            if status == "dubbed":
                return status_data
            elif status == "failed":
                err = status_data.get("error") or "Unknown error"
                raise PipelineError(f"ElevenLabs Dubbing job failed: {err}")

            time.sleep(poll_interval)

        raise PipelineError(f"ElevenLabs Dubbing timed out after {timeout} seconds.")

    def download_dubbed_audio(self, dubbing_id, target_lang, output_path):
        """
        Downloads the complete dubbed audio track from ElevenLabs.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{dubbing_id}/audio/{target_lang}"

        try:
            with request_with_retry("GET", url, headers=self.headers, stream=True, timeout=60, log_prefix="[Dubbing Client Download]") as resp:
                if resp.status_code != 200:
                    raise PipelineError(f"Failed to download dubbed audio: HTTP {resp.status_code}")
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise PipelineError(f"Network error downloading dubbed audio: {e}") from None

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise PipelineError("Downloaded dubbed audio file is empty.")
        return output_path

    def download_transcript(self, dubbing_id, target_lang, output_path, format_type="srt"):
        """
        Downloads transcript cues for the dubbed audio.
        Returns parsed Cue objects if format is srt.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/{dubbing_id}/transcript/{target_lang}?format_type={format_type}"

        try:
            resp = request_with_retry("GET", url, headers=self.headers, timeout=30, log_prefix="[Dubbing Client Transcript]")
            if resp.status_code != 200:
                raise PipelineError(f"Failed to download dubbed transcript: HTTP {resp.status_code}")
            output_path.write_text(resp.text, encoding="utf-8")
        except Exception as e:
            raise PipelineError(f"Network error downloading dubbed transcript: {e}") from None

        if format_type == "srt":
            cues = parse_srt(resp.text)
            return cues
        return resp.text
