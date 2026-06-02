"""contracts.py - Canonical pose types for the 4D Kinematics integration layer.

All provider adapters must convert their output into these dataclasses before
passing results to the normalization layer or the service facade.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class LandmarkIndex(int, Enum):
    NOSE              = 0
    LEFT_SHOULDER    = 11
    RIGHT_SHOULDER   = 12
    LEFT_ELBOW       = 13
    RIGHT_ELBOW      = 14
    LEFT_WRIST       = 15
    RIGHT_WRIST      = 16
    LEFT_HIP         = 23
    RIGHT_HIP        = 24
    LEFT_KNEE        = 25
    RIGHT_KNEE       = 26
    LEFT_ANKLE       = 27
    RIGHT_ANKLE      = 28
    LEFT_HEEL        = 29
    RIGHT_HEEL       = 30
    LEFT_FOOT_INDEX  = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class Landmark:
    """A single 3-D pose landmark with visibility confidence."""
    x: float
    y: float
    z: float
    visibility: float
    presence: float = 1.0


@dataclass
class PoseFrame:
    """All landmarks for a single video frame."""
    frame_index: int
    timestamp_ms: float
    landmarks: List[Landmark] = field(default_factory=list)
    world_landmarks: List[Landmark] = field(default_factory=list)

    def get(self, idx: LandmarkIndex) -> Optional[Landmark]:
        try:
            return self.landmarks[int(idx)]
        except IndexError:
            return None


class SkillType(Enum):
    UNKNOWN         = auto()
    BACKHANDSPRING  = auto()
    BACKWALKOVER    = auto()
    FRONTWALKOVER   = auto()
    ROUNDOFF        = auto()
    CARTWHEEL       = auto()
    HANDSTAND       = auto()
    AERIAL          = auto()


@dataclass
class SkillScore:
    """Output of the scoring layer for a single detected skill attempt."""
    skill: SkillType
    score: float
    start_frame: int
    end_frame: int
    notes: List[str] = field(default_factory=list)
