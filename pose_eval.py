import os
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

"""pose_eval.py - 4D Kinematics Gymnastics Pose Evaluator CLI entry point.

Usage:
    python pose_eval.py --model pose_landmarker.task --source video.mp4
    python pose_eval.py --model pose_landmarker.task --source 0  # webcam

Outputs:
    - Annotated video (optional, --no-display to skip window)
    - CSV and JSON metric exports
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List

import cv2

from pose_integration.contracts import LandmarkIndex, SkillScore
from pose_integration.normalize import joint_angle
from pose_integration.service import PoseService


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_landmarks(frame, pose_frame):
    """Overlay green dots for each visible landmark."""
    h, w = frame.shape[:2]
    for lm in pose_frame.landmarks:
        if lm.visibility > 0:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def _draw_scores(frame, scores: List[SkillScore]):
    """Print skill scores in the top-left corner."""
    for i, s in enumerate(scores):
        label = f"{s.skill.name}: {s.score:.1f}"
        cv2.putText(frame, label, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[error] Cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    print(f"[info] Starting pose evaluation | model={args.model} source={args.source}")

    with PoseService(model_path=args.model, visibility_threshold=args.visibility) as svc:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            timestamp_ms = frame_idx * (1000.0 / fps)
            pose_frame = svc.process_frame(bgr, frame_idx, timestamp_ms)

            if pose_frame and not args.no_display:
                _draw_landmarks(bgr, pose_frame)
                scores = svc.score_frame(pose_frame)
                _draw_scores(bgr, scores)

            if not args.no_display:
                cv2.imshow("4D Kinematics", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    metrics = _extract_metrics(svc)
    _export_csv(metrics, out_dir / "metrics.csv")
    _export_json(metrics, out_dir / "metrics.json")
    print(f"[done] Processed {frame_idx} frames.")


if __name__ == "__main__":
    main()
