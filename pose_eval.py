from __future__ import annotations
import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""pose_eval.py - 4D Kinematics Gymnastics Pose Evaluator CLI entry point.

Usage:
    python pose_eval.py --model pose_landmarker.task --source 0
    python pose_eval.py --model pose_landmarker.task --source video.mp4
    python pose_eval.py --model pose_landmarker.task --source 0 --skill 1

Skill IDs (pass via --skill or select interactively):
    1 = Handstand  2 = Front Walkover  3 = Back Walkover
    4 = Back Handspring  5 = Cartwheel  6 = Roundoff

Outputs:
    - Live annotated window with skeleton, HUD, and timer
    - CSV and JSON metric exports
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import cv2

from pose_integration.contracts import LandmarkIndex, SkillScore
from pose_integration.draw import (
    draw_controls_hint,
    draw_feedback,
    draw_hud,
    draw_skeleton,
)
from pose_integration.normalize import joint_angle
from pose_integration.service import PoseService
from pose_integration.timer import SkillTimer
from pose_targets import SKILL_REGISTRY, SkillTarget, prompt_skill_selection


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics(svc: PoseService) -> List[dict]:
    rows = []
    for pf in svc.all_frames():
        knee_angle = joint_angle(
            pf,
            LandmarkIndex.LEFT_HIP,
            LandmarkIndex.LEFT_KNEE,
            LandmarkIndex.LEFT_ANKLE,
        )
        hip_angle = joint_angle(
            pf,
            LandmarkIndex.LEFT_SHOULDER,
            LandmarkIndex.LEFT_HIP,
            LandmarkIndex.LEFT_KNEE,
        )
        rows.append({
            "frame": pf.frame_index,
            "timestamp_ms": pf.timestamp_ms,
            "left_knee_angle": round(knee_angle, 2) if knee_angle is not None else None,
            "left_hip_angle": round(hip_angle, 2) if hip_angle is not None else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_csv(rows: List[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] CSV -> {path}")


def _export_json(rows: List[dict], path: Path):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[export] JSON -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4D Kinematics Pose Evaluator")
    parser.add_argument("--model", required=True, help="Path to pose_landmarker.task model file")
    parser.add_argument("--source", default="0", help="Video file path or webcam index (default: 0)")
    parser.add_argument("--out-dir", default="output", help="Directory for exported metrics")
    parser.add_argument("--no-display", action="store_true", help="Disable video window (headless mode)")
    parser.add_argument("--visibility", type=float, default=0.5, help="Visibility confidence threshold")
    parser.add_argument("--skill", default=None,
                        help="Skill ID to evaluate (1-6). Skips interactive prompt.")
    args = parser.parse_args()

    # --- Skill selection ---
    if args.skill and args.skill in SKILL_REGISTRY:
        skill_target: SkillTarget = SKILL_REGISTRY[args.skill]
        print(f"[skill] Using: {skill_target.label}")
    else:
        skill_target = prompt_skill_selection()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] Cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    timer = SkillTimer()

    print(f"[info] Starting | skill={skill_target.label} | model={args.model} | source={args.source}")
    print("[info] Keys: Q = quit   SPACE = reset hold timer")

    with PoseService(model_path=args.model, visibility_threshold=args.visibility) as svc:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            timestamp_ms = frame_idx * (1000.0 / fps)
            pose_frame = svc.process_frame(bgr, frame_idx, timestamp_ms)

            joint_statuses = {}
            in_position = False

            if pose_frame:
                in_position, joint_statuses = skill_target.check_frame(pose_frame)
                timer.update(in_position)

                if not args.no_display:
                    draw_skeleton(bgr, pose_frame, joint_statuses)
                    draw_feedback(bgr, joint_statuses)

            if not args.no_display:
                draw_hud(bgr, skill_target.label, timer.elapsed, in_position)
                draw_controls_hint(bgr)
                cv2.imshow("4D Kinematics", bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    timer.reset()
                    print("[timer] Reset")
            else:
                # headless: still update timer
                pass

            frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    # Print session summary
    summary = timer.summary()
    print(f"[done] Processed {frame_idx} frames.")
    print(f"[timer] Total hold: {summary['total_hold_secs']}s | "
          f"Best hold: {summary['best_hold_secs']}s | "
          f"Hold count: {summary['hold_count']}")

    metrics = _extract_metrics(svc)
    _export_csv(metrics, out_dir / "metrics.csv")
    _export_json(metrics, out_dir / "metrics.json")

    # Append timer summary to JSON
    summary_path = out_dir / "session_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "skill": skill_target.label,
            "frames_processed": frame_idx,
            **summary,
        }, f, indent=2)
    print(f"[export] Session summary -> {summary_path}")


if __name__ == "__main__":
    main()
