"""pose_targets.py - Per-skill joint angle targets and in-position detection.

Each skill defines:
  - required_joints: landmark indices that must be visible
  - angle_targets: dict of (a, b, c) triplet -> (min_deg, max_deg) range
  - label: human-readable display name
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pose_integration.contracts import LandmarkIndex, PoseFrame
from pose_integration.normalize import joint_angle, landmarks_visible


# Joint triplet type: (vertex_a, vertex_b_pivot, vertex_c)
AngleTriplet = Tuple[LandmarkIndex, LandmarkIndex, LandmarkIndex]


@dataclass
class SkillTarget:
    label: str
    required_joints: List[LandmarkIndex]
    # maps angle triplet -> acceptable (min, max) range in degrees
    angle_targets: Dict[AngleTriplet, Tuple[float, float]] = field(default_factory=dict)
    tolerance: float = 15.0  # degrees of grace on each side

    def check_frame(self, frame: PoseFrame) -> Tuple[bool, Dict[str, bool]]:
        """Return (all_in_position, per_joint_status_dict)."""
        if not landmarks_visible(frame, self.required_joints):
            return False, {}
        statuses: Dict[str, bool] = {}
        for (a, b, c), (lo, hi) in self.angle_targets.items():
            angle = joint_angle(frame, a, b, c)
            key = f"{b.name}"
            if angle is None:
                statuses[key] = False
            else:
                statuses[key] = lo - self.tolerance <= angle <= hi + self.tolerance
        all_ok = all(statuses.values()) if statuses else False
        return all_ok, statuses


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

HANDSTAND = SkillTarget(
    label="Handstand",
    required_joints=[
        LandmarkIndex.LEFT_WRIST, LandmarkIndex.RIGHT_WRIST,
        LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER,
        LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP,
        LandmarkIndex.LEFT_KNEE, LandmarkIndex.RIGHT_KNEE,
        LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE,
    ],
    angle_targets={
        # Straight body line: hip angle ~180deg (shoulder-hip-knee)
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE): (160.0, 180.0),
        # Straight knee: hip-knee-ankle ~180deg
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE): (160.0, 180.0),
        # Shoulder stack: elbow-shoulder-hip ~170-180 (arms overhead / inline)
        (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP): (155.0, 180.0),
    },
    tolerance=12.0,
)

FRONT_WALKOVER = SkillTarget(
    label="Front Walkover",
    required_joints=[
        LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP,
        LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
        LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE,
    ],
    angle_targets={
        # Lead leg: hip-knee-ankle extended ~170
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE): (155.0, 180.0),
        # Hip split: right hip flexion (wide split angle)
        (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_HIP, LandmarkIndex.LEFT_HIP): (140.0, 180.0),
    },
    tolerance=15.0,
)

BACK_WALKOVER = SkillTarget(
    label="Back Walkover",
    required_joints=[
        LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP,
        LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
        LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE,
    ],
    angle_targets={
        # Same split mechanics as front, mirrored entry
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE): (155.0, 180.0),
        (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_HIP, LandmarkIndex.LEFT_HIP): (140.0, 180.0),
    },
    tolerance=15.0,
)

BACK_HANDSPRING = SkillTarget(
    label="Back Handspring",
    required_joints=[
        LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP,
        LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
        LandmarkIndex.LEFT_WRIST, LandmarkIndex.LEFT_ELBOW,
    ],
    angle_targets={
        # Preflight: upright body (shoulder-hip-knee)
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE): (160.0, 180.0),
        # Arms raised overhead before takeoff
        (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP): (150.0, 180.0),
    },
    tolerance=15.0,
)

CARTWHEEL = SkillTarget(
    label="Cartwheel",
    required_joints=[
        LandmarkIndex.LEFT_WRIST, LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_SHOULDER,
        LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
    ],
    angle_targets={
        # Arm straight: wrist-elbow-shoulder ~170
        (LandmarkIndex.LEFT_WRIST, LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_SHOULDER): (155.0, 180.0),
        # Leg extended: hip-knee-ankle ~170
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE): (155.0, 180.0),
    },
    tolerance=15.0,
)

ROUNDOFF = SkillTarget(
    label="Roundoff",
    required_joints=[
        LandmarkIndex.LEFT_WRIST, LandmarkIndex.LEFT_SHOULDER,
        LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE,
    ],
    angle_targets={
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE): (155.0, 180.0),
        (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP): (150.0, 180.0),
    },
    tolerance=15.0,
)


# Registry - ordered for display menu
SKILL_REGISTRY: Dict[str, SkillTarget] = {
    "1": HANDSTAND,
    "2": FRONT_WALKOVER,
    "3": BACK_WALKOVER,
    "4": BACK_HANDSPRING,
    "5": CARTWHEEL,
    "6": ROUNDOFF,
}


def prompt_skill_selection() -> SkillTarget:
    """Interactive CLI skill picker. Returns the chosen SkillTarget."""
    print("\n=== 4D Kinematics - Select Skill ===")
    for key, skill in SKILL_REGISTRY.items():
        print(f"  [{key}] {skill.label}")
    while True:
        choice = input("Enter skill number: ").strip()
        if choice in SKILL_REGISTRY:
            selected = SKILL_REGISTRY[choice]
            print(f"[skill] Selected: {selected.label}")
            return selected
        print(f"  Invalid choice '{choice}'. Try again.")
