"""normalize.py - Confidence filtering and landmark normalization.

This module filters out low-confidence landmarks and normalises coordinates
so downstream scoring code never has to deal with raw provider values.
"""

from __future__ import annotations
from typing import List, Optional
import math

from .contracts import Landmark, LandmarkIndex, PoseFrame

# Default visibility threshold - landmarks below this are treated as missing
DEFAULT_VISIBILITY_THRESHOLD: float = 0.5
DEFAULT_PRESENCE_THRESHOLD: float = 0.5


def filter_landmarks(
    frame: PoseFrame,
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
    presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD,
) -> PoseFrame:
    """Return a new PoseFrame with low-confidence landmarks zeroed out.

    Landmarks that fall below either threshold have their visibility set to 0.0
    so callers can test `lm.visibility > 0` as a reliable confidence gate.
    """
    def _filter(lm: Landmark) -> Landmark:
        if lm.visibility < visibility_threshold or lm.presence < presence_threshold:
            return Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=0.0, presence=0.0)
        return lm

    return PoseFrame(
        frame_index=frame.frame_index,
        timestamp_ms=frame.timestamp_ms,
        landmarks=[_filter(lm) for lm in frame.landmarks],
        world_landmarks=[_filter(lm) for lm in frame.world_landmarks],
    )


def joint_angle(
    frame: PoseFrame,
    a: LandmarkIndex,
    b: LandmarkIndex,
    c: LandmarkIndex,
    use_world: bool = True,
) -> Optional[float]:
    """Compute the angle (degrees) at joint b formed by segments a-b and b-c.

    Uses world landmarks when available and use_world=True.
    Returns None if any of the three landmarks has zero visibility.
    """
    pool = frame.world_landmarks if (use_world and frame.world_landmarks) else frame.landmarks

    def get(idx: LandmarkIndex) -> Optional[Landmark]:
        try:
            lm = pool[int(idx)]
            return lm if lm.visibility > 0 else None
        except IndexError:
            return None

    pa, pb, pc = get(a), get(b), get(c)
    if pa is None or pb is None or pc is None:
        return None

    # Vectors from b
    v1 = (pa.x - pb.x, pa.y - pb.y, pa.z - pb.z)
    v2 = (pc.x - pb.x, pc.y - pb.y, pc.z - pb.z)

    dot = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x ** 2 for x in v1))
    mag2 = math.sqrt(sum(x ** 2 for x in v2))

    if mag1 == 0 or mag2 == 0:
        return None

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def landmarks_visible(frame: PoseFrame, indices: List[LandmarkIndex]) -> bool:
    """Return True only if every requested landmark has non-zero visibility."""
    for idx in indices:
        lm = frame.get(idx)
        if lm is None or lm.visibility <= 0:
            return False
    return True
