"""
Stage 2 — Scene Classification
Detects camera cuts, then classifies each shot as:
  A_ROLL   — target speaker confirmed visible & mouth moving  → send to lipsync
  B_ROLL   — product/background shot, no target speaker      → dubbed audio + original video
  SILENT   — no speech in this interval                      → copy untouched

Uses:
  - PySceneDetect / ffmpeg for cut detection
  - OpenCV YuNet + SFace for deep face identity (replaces legacy skin-color tracker)
  - Mouth motion score to separate speaking vs posed shots
"""
import json
import os
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional

from .common import PipelineError
from .scene_analyzer import detect_scenes, get_video_info

SERVICE_ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = SERVICE_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = SERVICE_ROOT / "models" / "face_recognition_sface_2021dec.onnx"

AROLL_THRESHOLD = 0.70   # SFace cosine similarity ≥ this → target speaker present
REJECT_THRESHOLD = 0.40  # Below this → definitely not the target speaker
MOTION_THRESHOLD = 4.0   # Mouth motion score above this → speaker is talking
MAX_HEAD_YAW_DEG = float(os.getenv('LIPSYNC_MAX_HEAD_YAW_DEG', '16'))
MAX_HEAD_PITCH_DEG = float(os.getenv('LIPSYNC_MAX_HEAD_PITCH_DEG', '20'))
MAX_HEAD_PITCH_DELTA_DEG = float(os.getenv('LIPSYNC_MAX_HEAD_PITCH_DELTA_DEG', '15'))
MAX_AWAY_FACE_RATIO = float(os.getenv('LIPSYNC_MAX_AWAY_FACE_RATIO', '0.4'))
CLASSIFY_WINDOW_SEC = float(os.getenv('LIPSYNC_CLASSIFY_WINDOW_SEC', '0.5'))


class SceneClassifier:
    """
    Classifies video shots against a locked identity anchor.
    Call set_anchor() first, then classify_scenes() for all shots.
    """

    def __init__(self, yunet_path: Path = None, sface_path: Path = None,
                 conf_threshold: float = 0.7):
        y = str(yunet_path or YUNET_PATH)
        s = str(sface_path or SFACE_PATH)
        self._has_models = False
        self.detector = None
        self.recognizer = None
        if Path(y).exists() and Path(s).exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(y, "", (640, 640), conf_threshold, 0.3, 5000)
                self.recognizer = cv2.FaceRecognizerSF.create(s, "")
                self._has_models = True
            except Exception as exc:
                print(f"[Classifier] Model load warning ({exc}) — falling back to motion-only classification.")
        self.anchor_embedding = None
        self.anchor_angles = None

    # ------------------------------------------------------------------
    # Anchor setup
    # ------------------------------------------------------------------
    def set_anchor_from_video(self, video_path: Path, target_time: float = None,
                               target_point: tuple = None):
        """
        Extract identity anchor from the video itself.
        target_time: timestamp (seconds) showing the speaker — uses midpoint if None.
        target_point: (x, y) pixel point inside her face when multiple faces visible.
        """
        if not self._has_models:
            print("[Classifier] YuNet/SFace models not found — face identity disabled.")
            return None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Probe multiple candidate frames and pick the one with the clearest face
        probe_times = []
        if target_time is not None:
            probe_times = [target_time, target_time + 0.5, target_time - 0.5]
        else:
            dur = total / fps
            probe_times = [dur * fraction for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]

        candidates = []
        for t in probe_times:
            if t < 0:
                continue
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue

            selected = None
            if target_point:
                px, py = target_point
                for f in faces:
                    fx, fy, fw, fh = f[:4]
                    if fx <= px <= fx + fw and fy <= py <= fy + fh:
                        selected = f
                        break
                if selected is None:
                    continue
            else:
                selected = max(faces, key=lambda f: f[2] * f[3])

            aligned = self.recognizer.alignCrop(frame, selected)
            embedding = self.recognizer.feature(aligned)
            box = [int(v) for v in selected[:4]]
            angles = self._head_angles(frame, selected)
            pose_cost = abs(angles[0]) + abs(angles[1]) if angles else 180.0
            area = float(selected[2] * selected[3])
            candidates.append((pose_cost, -area, t, embedding, box, angles))

        cap.release()
        if candidates:
            _, _, t, embedding, box, angles = min(candidates, key=lambda item: (item[0], item[1]))
            self.anchor_embedding = embedding
            self.anchor_angles = angles
            pose_text = f", head pose={angles[0]:.0f}/{angles[1]:.0f} deg" if angles else ""
            print(f"[Classifier] Camera-facing anchor set from t={t:.2f}s{pose_text}, face box={box}")
            return box
        print("[Classifier] Warning: could not set identity anchor — will classify by motion only.")
        return None

    def set_anchor_from_image(self, image_path: Path):
        """Set anchor from a reference photo of the speaker."""
        if not self._has_models:
            return None
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise PipelineError(f"Cannot load reference image: {image_path}")
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            raise PipelineError("No face detected in reference image.")
        selected = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(frame, selected)
        self.anchor_embedding = self.recognizer.feature(aligned)
        self.anchor_angles = self._head_angles(frame, selected)
        print(f"[Classifier] Anchor set from reference image: {image_path.name}")

    # ------------------------------------------------------------------
    # Per-frame matching
    # ------------------------------------------------------------------
    def _similarity(self, embedding) -> float:
        if self.anchor_embedding is None:
            return -1.0
        return float(self.recognizer.match(
            self.anchor_embedding, embedding, cv2.FaceRecognizerSF_FR_COSINE
        ))

    @staticmethod
    def _head_angles(frame: np.ndarray, face: np.ndarray) -> tuple[float, float] | None:
        """Estimate head yaw/pitch from YuNet's five facial landmarks."""
        try:
            h, w = frame.shape[:2]
            # YuNet order: subject's right eye, left eye, nose, right mouth, left mouth.
            image_points = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
            model_points = np.asarray([
                [-30.0,  30.0, -30.0], [30.0,  30.0, -30.0],
                [  0.0,   0.0,   0.0], [-25.0, -30.0, -20.0],
                [ 25.0, -30.0, -20.0],
            ], dtype=np.float64)
            focal = float(max(w, h))
            camera = np.asarray([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
            ok, rotation, _ = cv2.solvePnP(
                model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_SQPNP
            )
            if not ok:
                return None
            matrix, _ = cv2.Rodrigues(rotation)
            forward = matrix @ np.asarray([0.0, 0.0, 1.0])
            # A front-facing model can point along either +/- Z depending on the
            # landmark convention; absolute Z keeps frontal pose near zero yaw.
            depth = abs(forward[2])
            yaw = np.degrees(np.arctan2(forward[0], depth))
            pitch = np.degrees(np.arctan2(forward[1], depth))
            return float(yaw), float(pitch)
        except (cv2.error, ValueError, TypeError):
            return None

    def _sample_frames(self, video_path: Path, start_s: float, end_s: float,
                        n_samples: int = 8) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        first = int(start_s * fps)
        last = int(end_s * fps)
        step = max(1, (last - first) // n_samples)
        frames = []
        for idx in range(first, last, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        return frames

    def _mouth_motion_score(self, frames: list[np.ndarray]) -> float:
        """Estimate mouth region motion relative to detected face bounding boxes."""
        if len(frames) < 2:
            return 0.0

        # If face detector is available, crop the actual mouth area from the face
        if self._has_models and self.detector is not None:
            mouth_crops = []
            for frame in frames:
                h, w = frame.shape[:2]
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = face[:4]
                    # Mouth region inside the face bounding box
                    mx1 = max(0, int(fx + 0.25 * fw))
                    mx2 = min(w, int(fx + 0.75 * fw))
                    my1 = max(0, int(fy + 0.65 * fh))
                    my2 = min(h, int(fy + 0.95 * fh))
                    m = frame[my1:my2, mx1:mx2]
                    if m.size > 0:
                        m_gray = cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (64, 48))
                        mouth_crops.append(m_gray)
            if len(mouth_crops) >= 2:
                scores = []
                for i in range(1, len(mouth_crops)):
                    diff = cv2.absdiff(mouth_crops[i - 1].astype(np.float32), mouth_crops[i].astype(np.float32))
                    scores.append(float(diff.mean()))
                return float(np.mean(scores)) if scores else 0.0

        # Fallback: lower third of frame if face detector unavailable
        scores = []
        for i in range(1, len(frames)):
            h, w = frames[i].shape[:2]
            r1 = cv2.cvtColor(frames[i - 1][h * 2 // 3:, w // 4: w * 3 // 4], cv2.COLOR_BGR2GRAY)
            r2 = cv2.cvtColor(frames[i][h * 2 // 3:, w // 4: w * 3 // 4], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(r1.astype(np.float32), r2.astype(np.float32))
            scores.append(float(diff.mean()))
        return float(np.mean(scores)) if scores else 0.0

    def _jaw_articulation_score(self, frames: list[np.ndarray]) -> float:
        """Measure variation in jaw opening (mouth-to-nose vertical distance / face height).
        Real dialogue shows active articulation variance; static smiles or posed looks have near-zero variation."""
        if not self._has_models or self.detector is None or len(frames) < 3:
            return 0.05
        dists = []
        for frame in frames:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is not None and len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                fh = float(face[3])
                if fh > 10:
                    ny = float(face[9])
                    my = float((face[11] + face[13]) / 2.0)
                    dists.append((my - ny) / fh)
        if len(dists) < 3:
            return 0.05
        return float(np.std(dists))

    def classify_shot(self, video_path: Path, start_s: float, end_s: float,
                       speech_intervals: list[tuple] = None) -> dict:
        """
        Classify a single shot.
        speech_intervals: list of (start_s, end_s) pairs from transcript where speech occurs.
        Returns dict: {class: 'A_ROLL'|'B_ROLL'|'SILENT', similarity, motion, face_coords}
        """
        # Check if any speech falls in this window
        has_speech = True
        if speech_intervals is not None:
            has_speech = any(
                not (end_s <= si[0] or start_s >= si[1])
                for si in speech_intervals
            )

        if not has_speech:
            return {
                "class": "SILENT",
                "similarity": -1.0,
                "motion": 0.0,
                "face_coords": None,
                "reason": "No speech in this interval",
            }

        frames = self._sample_frames(video_path, start_s, end_s)
        motion = self._mouth_motion_score(frames)

        if not self._has_models or self.anchor_embedding is None:
            return {
                "class": "B_ROLL",
                "similarity": -1.0,
                "motion": round(motion, 2),
                "face_coords": None,
                "camera_facing": None,
                "reason": "Face identity/head-pose models unavailable; skipping lip-sync",
            }

        # Deep face identity check
        best_sim = -1.0
        best_coords = None
        pose_count = 0
        away_count = 0
        for frame in frames:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None:
                continue
            for face in faces:
                aligned = self.recognizer.alignCrop(frame, face)
                emb = self.recognizer.feature(aligned)
                sim = self._similarity(emb)
                # Pose is useful even when an angled/downward face weakens the
                # identity embedding, so collect it at the lower reject boundary.
                if sim >= REJECT_THRESHOLD:
                    angles = self._head_angles(frame, face)
                    if angles is not None:
                        pose_count += 1
                        yaw, pitch = angles
                        yaw_off = abs(yaw) > MAX_HEAD_YAW_DEG
                        if self.anchor_angles is not None:
                            yaw_off = yaw_off or abs(yaw - self.anchor_angles[0]) > MAX_HEAD_YAW_DEG
                            pitch_off = abs(pitch - self.anchor_angles[1]) > MAX_HEAD_PITCH_DELTA_DEG
                        else:
                            pitch_off = abs(pitch) > MAX_HEAD_PITCH_DEG
                        if yaw_off or pitch_off:
                            away_count += 1
                if sim > best_sim:
                    best_sim = sim
                    best_coords = [int(v) for v in face[:4]]

        jaw_std = self._jaw_articulation_score(frames)

        away_ratio = away_count / pose_count if pose_count else 0.0
        if best_sim >= AROLL_THRESHOLD and pose_count and away_ratio > MAX_AWAY_FACE_RATIO:
            label = "B_ROLL"
            reason = (f"Target head is off-axis in {away_ratio:.0%} of samples; "
                      "skipping lip-sync while the speaker looks away")
        elif best_sim >= AROLL_THRESHOLD and motion >= MOTION_THRESHOLD and jaw_std < 0.014:
            # Motion was detected (hands/towel moving), but jaw opening is static (smile/posing)
            label = "B_ROLL"
            reason = f"Target face holding static expression/smile (jaw articulation std={jaw_std:.4f})"
        elif best_sim >= AROLL_THRESHOLD and motion >= MOTION_THRESHOLD:
            label = "A_ROLL"
            reason = f"Target speaker confirmed (sim={best_sim:.2f}, motion={motion:.2f}, jaw_std={jaw_std:.4f})"
        elif best_sim >= AROLL_THRESHOLD:
            # Face present but not talking (posed/listening shot)
            label = "B_ROLL"
            reason = f"Target face visible but not speaking (motion={motion:.2f})"
        elif best_sim >= REJECT_THRESHOLD:
            label = "B_ROLL"
            reason = f"Uncertain face (sim={best_sim:.2f}) — treating as B-roll"
        else:
            label = "B_ROLL"
            reason = f"Non-target face or no face (sim={best_sim:.2f})"

        return {
            "class": label,
            "similarity": round(best_sim, 3),
            "motion": round(motion, 2),
            "face_coords": best_coords,
            "camera_facing": (away_ratio <= MAX_AWAY_FACE_RATIO) if pose_count else None,
            "head_away_ratio": round(away_ratio, 3),
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Full scene classification
    # ------------------------------------------------------------------
    def classify_scenes(self, video_path: Path, scenes: list[dict],
                         words: list[dict] = None) -> list[dict]:
        """
        Classify all scenes. Returns enriched scene list with 'class', 'reason', etc.
        words: transcript words with {start, end} for speech interval detection.
        """
        speech_intervals = None
        if words:
            speech_intervals = [(w["start"], w["end"]) for w in words]

        results = []
        for i, scene in enumerate(scenes):
            start_s = scene["start"]
            end_s = scene["end"]
            window = max(0.25, CLASSIFY_WINDOW_SEC)
            t = start_s
            while t < end_s - 1e-4:
                window_end = min(end_s, t + window)
                result = self.classify_shot(video_path, t, window_end, speech_intervals)
                enriched = {**scene, "start": t, "end": window_end, **result}
                if results and results[-1]["class"] == enriched["class"] and abs(results[-1]["end"] - t) < 0.02:
                    results[-1]["end"] = window_end
                    old_ratio = results[-1].get("head_away_ratio")
                    new_ratio = enriched.get("head_away_ratio")
                    if old_ratio is not None and new_ratio is not None:
                        results[-1]["head_away_ratio"] = round((old_ratio + new_ratio) / 2, 3)
                else:
                    results.append(enriched)
                t = window_end
            print(f"[Classifier] Shot {i:02d} [{start_s:.2f}s-{end_s:.2f}s] → "
                  f"{result['class']} | {result['reason']}")
        return results


def classify_video(
    video_path: Path,
    job_dir: Path,
    words: list[dict],
    scene_threshold: float = 32.0,
    target_time: float = None,
    target_point: tuple = None,
    reference_image: Path = None,
) -> list[dict]:
    """
    Top-level Stage 2 entry point.
    Detects scenes, classifies each shot, saves scenes.json.
    Returns enriched scene list.
    """
    job_dir = Path(job_dir)
    scenes_path = job_dir / "scenes.json"

    # Detect scenes (cached by scene_analyzer)
    print("[Classifier] Detecting camera cuts...")
    scenes = detect_scenes(video_path, threshold=scene_threshold)
    info = get_video_info(video_path)
    print(f"[Classifier] Found {len(scenes)} shots in {info['duration']:.2f}s video.")

    clf = SceneClassifier()

    # Set identity anchor
    if reference_image and Path(reference_image).exists():
        clf.set_anchor_from_image(Path(reference_image))
    elif clf._has_models:
        clf.set_anchor_from_video(video_path, target_time=target_time,
                                   target_point=target_point)

    # Classify
    classified = clf.classify_scenes(video_path, scenes, words=words)

    # Save
    scenes_path.write_text(
        json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aroll = sum(1 for s in classified if s["class"] == "A_ROLL")
    broll = sum(1 for s in classified if s["class"] == "B_ROLL")
    silent = sum(1 for s in classified if s["class"] == "SILENT")
    print(f"[Classifier] Classified: {aroll} A-Roll | {broll} B-Roll | {silent} Silent")
    print(f"[Classifier] Saved → {scenes_path.name}")
    return classified
