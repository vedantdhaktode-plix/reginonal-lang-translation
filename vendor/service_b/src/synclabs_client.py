import mimetypes
import os
import time
from pathlib import Path
import requests
from .common import PipelineError, request_with_retry


class SyncLabsClient:
    """
    Client for Sync Labs API (https://api.sync.so).
    Supports asset uploading via presigned URLs, lipsync-2 generation, polling, and downloading.
    """

    def __init__(self, api_key=None, base_url="https://api.sync.so", default_model=None):
        self.api_key = api_key or os.environ.get("SYNC_API_KEY")
        if not self.api_key:
            raise PipelineError("Set SYNC_API_KEY in .env or your environment.")
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-api-key": self.api_key}
        self.default_model = default_model or os.environ.get("SYNC_MODEL", "lipsync-2")

    def upload_file(self, file_path, content_type=None):
        """
        Uploads a local file to Sync Labs via the 3-step presigned URL flow:
        1. POST /v2/assets/upload to get uploadUrl and canonical url
        2. Direct PUT to uploadUrl with raw binary bytes
        3. POST /v2/assets to register the uploaded asset
        Returns canonical media URL.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise PipelineError(f"File not found for upload: {file_path}")

        if not content_type:
            suffix = file_path.suffix.lower()
            if suffix in (".mp4", ".m4v"):
                content_type = "video/mp4"
            elif suffix == ".wav":
                content_type = "audio/wav"
            elif suffix == ".mp3":
                content_type = "audio/mpeg"
            else:
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise PipelineError(f"Cannot upload empty file: {file_path}")

        # Step 1: Request presigned URL
        req_url = f"{self.base_url}/v2/assets/upload"
        req_body = {
            "fileName": file_path.name,
            "contentType": content_type,
            "size": file_size
        }
        try:
            r1 = request_with_retry(
                "POST",
                req_url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=req_body,
                timeout=30,
                log_prefix="[SyncLabs Upload Init]",
            )
        except Exception as e:
            raise PipelineError(f"Failed to request presigned upload URL from Sync Labs: {e}") from None

        if r1.status_code not in (200, 201):
            raise PipelineError(f"Sync Labs asset upload init failed: HTTP {r1.status_code} - {r1.text}")

        upload_data = r1.json()
        upload_url = upload_data.get("uploadUrl")
        canonical_url = upload_data.get("url")

        if not upload_url or not canonical_url:
            raise PipelineError("Sync Labs did not return valid uploadUrl and canonical url.")

        # Step 2: PUT file bytes to presigned URL (without API key)
        try:
            with open(file_path, "rb") as f:
                r2 = request_with_retry(
                    "PUT",
                    upload_url,
                    headers={"Content-Type": content_type},
                    data=f,
                    timeout=300,
                    log_prefix="[SyncLabs S3 PUT]",
                )
        except Exception as e:
            raise PipelineError(f"Failed to upload file bytes to Sync Labs storage: {e}") from None

        if r2.status_code not in (200, 201, 204):
            raise PipelineError(f"Failed to PUT file to Sync Labs storage: HTTP {r2.status_code}")

        # Step 3: Register asset
        reg_url = f"{self.base_url}/v2/assets"
        asset_type = "VIDEO" if content_type.startswith("video") else ("IMAGE" if content_type.startswith("image") else "AUDIO")
        try:
            r3 = request_with_retry(
                "POST",
                reg_url,
                headers={**self.headers, "Content-Type": "application/json"},
                json={"url": canonical_url, "type": asset_type},
                timeout=30,
                log_prefix="[SyncLabs Asset Register]",
            )
        except Exception as e:
            raise PipelineError(f"Failed to register asset with Sync Labs: {e}") from None

        if r3.status_code not in (200, 201):
            raise PipelineError(f"Sync Labs asset registration failed: HTTP {r3.status_code} - {r3.text}")

        return canonical_url

    def generate(self, video_url, audio_url, model=None, options=None):
        """
        Creates a lip-sync generation using the specified model (defaults to self.default_model, e.g. 'lipsync-2').
        Returns the generation record containing 'id'.
        """
        model_name = model or self.default_model
        url = f"{self.base_url}/v2/generate"
        payload = {
            "model": model_name,
            "input": [
                {"type": "video", "url": video_url},
                {"type": "audio", "url": audio_url}
            ]
        }
        default_options = {
            "sync_mode": "cut_off",
            "active_speaker_detection": {"auto_detect": True}
        }
        if options:
            default_options.update(options)
        payload["options"] = default_options

        try:
            resp = request_with_retry(
                "POST",
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
                timeout=30,
                log_prefix="[SyncLabs Generate]",
            )
        except Exception as e:
            raise PipelineError(f"Failed to submit generation to Sync Labs: {e}") from None

        if resp.status_code not in (200, 201):
            msg = f"Sync Labs generate returned HTTP {resp.status_code}"
            try:
                err_detail = resp.json().get("error") or resp.json().get("message")
                if err_detail:
                    msg += f": {err_detail}"
            except Exception:
                pass
            raise PipelineError(msg)

        try:
            data = resp.json()
            gen_id = data.get("id")
            if not gen_id:
                raise ValueError("Missing 'id' in Sync Labs generate response.")
            return data
        except (ValueError, KeyError):
            raise PipelineError("Invalid JSON from Sync Labs generate.")

    def get_status(self, generation_id):
        """
        Polls the status of an ongoing generation.
        Returns dict with status ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'), outputUrl, error.
        """
        url = f"{self.base_url}/v2/generate/{generation_id}"
        try:
            resp = request_with_retry("GET", url, headers=self.headers, timeout=30, log_prefix="[SyncLabs Status]")
        except Exception as e:
            raise PipelineError(f"Failed to get Sync Labs generation status: {e}") from None

        if resp.status_code != 200:
            raise PipelineError(f"Sync Labs status check failed: HTTP {resp.status_code}")

        try:
            return resp.json()
        except ValueError:
            raise PipelineError("Invalid JSON from Sync Labs status endpoint.")

    def wait_for_completion(self, generation_id, poll_interval=10, timeout=1200, callback=None):
        """
        Polls until the generation is COMPLETED or FAILED.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_data = self.get_status(generation_id)
            status = status_data.get("status", "").upper()
            elapsed = int(time.time() - start_time)
            print(f"    [Sync Labs] Job {generation_id}: {status} ({elapsed}s elapsed)...")
            if callback:
                callback(status, status_data)

            if status == "COMPLETED":
                output_url = status_data.get("outputUrl")
                if not output_url:
                    raise PipelineError("Sync Labs generation completed but outputUrl is missing.")
                return status_data
            elif status in ("FAILED", "REJECTED"):
                err = status_data.get("error") or status_data.get("message") or "Unknown error"
                raise PipelineError(f"Sync Labs generation {generation_id} failed: {err}")

            time.sleep(poll_interval)

        raise PipelineError(f"Sync Labs generation timed out after {timeout} seconds.")

    def download_result(self, output_url, output_path):
        """
        Downloads the generated lip-synced video to local disk.
        """
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with request_with_retry("GET", output_url, stream=True, timeout=120, log_prefix="[SyncLabs Download]") as resp:
                if resp.status_code != 200:
                    raise PipelineError(f"Failed to download Sync Labs video: HTTP {resp.status_code}")
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise PipelineError(f"Network error downloading Sync Labs video: {e}") from None

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise PipelineError("Downloaded Sync Labs video file is empty.")
        return output_path
