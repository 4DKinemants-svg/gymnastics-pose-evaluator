"""service.py - Integration facade for the 4D Kinematics pose evaluation pipeline.

PoseService is the single entry point for application code. It owns the
MediaPipe landmarker, drives the per-frame adapter conversion, applies
normalization, computes joint angles, and returns scored results.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from .contracts import LandmarkIndex, PoseFrame, SkillScore, SkillType
from .adapters.mediapipe_tasks import result_to_pose_frame
from .normalize import filter_landmarks, joint_angle, landmarks_visible

_UPPER_BODY = [
    LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER,
    LandmarkIndex.LEFT_ELBOW, LandmarkIndex.RIGHT_ELBOW,
    LandmarkIndex.LEFT_WRIST, LandmarkIndex.RIGHT_WRIST,
]


class PoseService:
    """Facade that wraps MediaPipe and exposes a clean per-frame API."""

    def __init__(
        self,
        model_path: str,
        visibility_threshold: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._visibility_threshold = visibility_threshold
        self._frames: List[PoseFrame] = []

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=num_poses,
            min_pose_detection_confidence=visibility_threshold,
            min_pose_presence_confidence=visibility_threshold,
            min_tracking_confidence=visibility_threshold,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame, frame_index: int, timestamp_ms: float) -> Optional[PoseFrame]:
        """Run pose detection on a single BGR OpenCV frame.

        Returns a normalised PoseFrame or None if no pose is detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        frame = result_to_pose_frame(result, frame_index, timestamp_ms)
        if frame is None:
            return None

        frame = filter_landmarks(frame, self._visibility_threshold)
        self._frames.append(frame)
        return frame

    def joint_angle(self, frame: PoseFrame, a: LandmarkIndex, b: LandmarkIndex, c: LandmarkIndex) -> Optional[float]:
        """Convenience wrapper around normalize.joint_angle."""
        return joint_angle(frame, a, b, c)

    def score_frame(self, frame: PoseFrame) -> List[SkillScore]:
        """Simple heuristic scorer - returns detected skill scores for one frame."""
        scores: List[SkillScore] = []

        if landmarks_visible(frame, _UPPER_BODY + [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]):
            lw = frame.get(LandmarkIndex.LEFT_WRIST)
            ls = frame.get(LandmarkIndex.LEFT_SHOULDER)
            lh = frame.get(LandmarkIndex.LEFT_HIP)
            if lw and ls and lh and lw.y < ls.y and lh.y < ls.y:
                scores.append(SkillScore(
                    skill=SkillType.HANDSTAND,
                    score=75.0,
                    start_frame=frame.frame_index,
                    end_frame=frame.frame_index,
                    notes=["Wrists above shoulders, hips inverted"],
                ))

        return scores

    def all_frames(self) -> List[PoseFrame]:
        """Return all processed PoseFrames in order."""
        return list(self._frames)

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        self._landmarker.close()

    def __enter__(self) -> PoseService:
        return self

    def __exit__(self, *_) -> None:
        self.close()
