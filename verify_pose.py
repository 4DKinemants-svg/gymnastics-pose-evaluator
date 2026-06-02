"""verify_pose.py - Minimal verification harness for the pose integration layer.

Run without any model or video to verify that all modules import correctly
and the core data contracts work as expected.

Usage:
    python verify_pose.py
"""

from __future__ import annotations
import math
import sys
import traceback
from typing import List


PASS = "[PASS]"
FAIL = "[FAIL]"
_results: List[bool] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append(condition)
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------
def test_imports():
    print("\n-- Import checks --")
    try:
        from pose_integration.contracts import (
            Landmark, LandmarkIndex, PoseFrame, SkillScore, SkillType
        )
        check("contracts imports", True)
    except Exception as e:
        check("contracts imports", False, str(e))
        return

    try:
        from pose_integration.adapters.mediapipe_tasks import result_to_pose_frame
        check("adapter imports", True)
    except Exception as e:
        check("adapter imports", False, str(e))

    try:
        from pose_integration.normalize import filter_landmarks, joint_angle, landmarks_visible
        check("normalize imports", True)
    except Exception as e:
        check("normalize imports", False, str(e))

    try:
        from pose_integration.service import PoseService
        check("service imports", True)
    except Exception as e:
        check("service imports", False, str(e))


# ---------------------------------------------------------------------------
# 2. Contract unit checks
# ---------------------------------------------------------------------------
def test_contracts():
    print("\n-- Contract unit checks --")
    from pose_integration.contracts import Landmark, LandmarkIndex, PoseFrame, SkillScore, SkillType

    lm = Landmark(x=0.5, y=0.4, z=-0.1, visibility=0.9)
    check("Landmark creation", lm.x == 0.5 and lm.visibility == 0.9)

    frame = PoseFrame(
        frame_index=0,
        timestamp_ms=0.0,
        landmarks=[Landmark(x=i/33, y=i/33, z=0.0, visibility=0.8) for i in range(33)],
    )
    check("PoseFrame creation", len(frame.landmarks) == 33)
    lm_got = frame.get(LandmarkIndex.LEFT_SHOULDER)
    check("PoseFrame.get()", lm_got is not None)

    score = SkillScore(skill=SkillType.HANDSTAND, score=80.0, start_frame=0, end_frame=5)
    check("SkillScore creation", score.score == 80.0 and score.skill == SkillType.HANDSTAND)


# ---------------------------------------------------------------------------
# 3. Normalization checks
# ---------------------------------------------------------------------------
def test_normalization():
    print("\n-- Normalization checks --")
    from pose_integration.contracts import Landmark, LandmarkIndex, PoseFrame
    from pose_integration.normalize import filter_landmarks, joint_angle, landmarks_visible

    lms = [Landmark(x=i/33, y=i/33, z=0.0, visibility=0.8) for i in range(33)]
    # Set one landmark below threshold
    lms[LandmarkIndex.LEFT_KNEE] = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.1)
    frame = PoseFrame(frame_index=0, timestamp_ms=0.0, landmarks=lms)

    filtered = filter_landmarks(frame, visibility_threshold=0.5)
    knee = filtered.landmarks[LandmarkIndex.LEFT_KNEE]
    check("Low-visibility landmark filtered", knee.visibility == 0.0)

    # Create a simple right-angle triangle for angle check
    world = [
        Landmark(x=0.0, y=1.0, z=0.0, visibility=0.9),  # point A
        Landmark(x=0.0, y=0.0, z=0.0, visibility=0.9),  # point B (joint)
        Landmark(x=1.0, y=0.0, z=0.0, visibility=0.9),  # point C
    ] + [Landmark(x=0, y=0, z=0, visibility=0.9)] * 30
    frame2 = PoseFrame(frame_index=1, timestamp_ms=33.0, landmarks=world, world_landmarks=world)
    angle = joint_angle(frame2, LandmarkIndex(0), LandmarkIndex(1), LandmarkIndex(2))
    check("joint_angle right-angle", angle is not None and abs(angle - 90.0) < 0.01,
          f"got {angle}")

    check("landmarks_visible True", landmarks_visible(frame2, [LandmarkIndex(0), LandmarkIndex(1)]))
    check("landmarks_visible False (filtered knee)",
          not landmarks_visible(filtered, [LandmarkIndex.LEFT_KNEE]))


# ---------------------------------------------------------------------------
# 4. Adapter smoke check (no real model needed)
# ---------------------------------------------------------------------------
def test_adapter():
    print("\n-- Adapter smoke check --")
    from pose_integration.adapters.mediapipe_tasks import result_to_pose_frame

    class FakeLM:
        x = 0.5; y = 0.5; z = 0.0; visibility = 0.9; presence = 0.9

    class FakeResult:
        pose_landmarks = [[FakeLM()] * 33]
        pose_world_landmarks = [[FakeLM()] * 33]

    pf = result_to_pose_frame(FakeResult(), frame_index=0, timestamp_ms=0.0)
    check("Adapter returns PoseFrame", pf is not None)
    check("Adapter landmark count", pf is not None and len(pf.landmarks) == 33)

    class EmptyResult:
        pose_landmarks = []
        pose_world_landmarks = []

    pf_none = result_to_pose_frame(EmptyResult(), frame_index=1, timestamp_ms=33.0)
    check("Adapter returns None for empty result", pf_none is None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print(" 4D Kinematics Pose Integration Verification")
    print("=" * 50)

    for test_fn in [test_imports, test_contracts, test_normalization, test_adapter]:
        try:
            test_fn()
        except Exception:
            print(f"[ERROR] Unhandled exception in {test_fn.__name__}:")
            traceback.print_exc()
            _results.append(False)

    passed = sum(_results)
    total = len(_results)
    print(f"\n{'=' * 50}")
    print(f" Result: {passed}/{total} checks passed")
    print("=" * 50)

    sys.exit(0 if all(_results) else 1)


if __name__ == "__main__":
    main()
