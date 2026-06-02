"""mediapipe_tasks.py - MediaPipe Tasks API adapter.

Converts raw mediapipe.tasks.python.vision.PoseLandmarkerResult objects
into canonical PoseFrame instances used by the rest of the integration layer.
"""

from __future__ import annotations
from typing import List, Optional

from ..contracts import Landmark, PoseFrame


def _convert_landmark(lm) -> Landmark:
    """Map a single mediapipe NormalizedLandmark to a canonical Landmark."""
    return Landmark(
        x=float(lm.x),
        y=float(lm.y),
        z=float(getattr(lm, 'z', 0.0)),
        visibility=float(getattr(lm, 'visibility', 1.0)),
        presence=float(getattr(lm, 'presence', 1.0)),
    )


def result_to_pose_frame(
    result,
    frame_index: int,
    timestamp_ms: float,
) -> Optional[PoseFrame]:
    """Convert a PoseLandmarkerResult to a PoseFrame.

    Returns None if the result contains no detected poses.
    """
    if not result.pose_landmarks:
        return None

    # Use the first detected person only
    raw_landmarks = result.pose_landmarks[0]
    landmarks = [_convert_landmark(lm) for lm in raw_landmarks]

    world_landmarks: List[Landmark] = []
    if result.pose_world_landmarks:
        world_landmarks = [
            _convert_landmark(lm)
            for lm in result.pose_world_landmarks[0]
        ]

    return PoseFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
    )
