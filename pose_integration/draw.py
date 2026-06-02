"""draw.py - OpenCV drawing helpers for the 4D Kinematics pose evaluator.

Provides:
  draw_skeleton(frame, pose_frame, joint_statuses) - colored skeleton lines + landmark dots
  draw_hud(frame, skill_label, timer_secs, in_position) - HUD overlay
  draw_feedback(frame, joint_statuses) - per-joint feedback text
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .contracts import LandmarkIndex, PoseFrame

# ---------------------------------------------------------------------------
# Skeleton connection map (MediaPipe Pose 33-point topology)
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    # Face / head
    (LandmarkIndex.NOSE, LandmarkIndex.LEFT_SHOULDER),
    (LandmarkIndex.NOSE, LandmarkIndex.RIGHT_SHOULDER),
    # Torso
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER),
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP),
    (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP),
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
    # Left arm
    (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
    (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_WRIST),
    # Right arm
    (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
    (LandmarkIndex.RIGHT_ELBOW, LandmarkIndex.RIGHT_WRIST),
    # Left leg
    (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
    (LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
    (LandmarkIndex.LEFT_ANKLE, LandmarkIndex.LEFT_HEEL),
    (LandmarkIndex.LEFT_HEEL, LandmarkIndex.LEFT_FOOT_INDEX),
    # Right leg
    (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
    (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
    (LandmarkIndex.RIGHT_ANKLE, LandmarkIndex.RIGHT_HEEL),
    (LandmarkIndex.RIGHT_HEEL, LandmarkIndex.RIGHT_FOOT_INDEX),
]

# BGR color constants
COLOR_CORRECT   = (0, 220, 0)       # green  - joint in target range
COLOR_WRONG     = (0, 0, 220)       # red    - joint out of range
COLOR_DOT       = (255, 200, 0)     # cyan-ish landmark dot
COLOR_HUD_BG    = (30, 30, 30)      # dark panel
COLOR_YELLOW    = (0, 215, 255)
COLOR_TIMER_OK   = (0, 220, 100)    # green timer when in position
COLOR_TIMER_IDLE = (180, 180, 180)  # grey timer when idle


def _get_pt(frame: PoseFrame, idx: LandmarkIndex, w: int, h: int) -> Optional[Tuple[int, int]]:
    """Return pixel coords for a landmark, or None if not visible."""
    lm = frame.get(idx)
    if lm is None or lm.visibility <= 0:
        return None
    x = max(0, min(w - 1, int(lm.x * w)))
    y = max(0, min(h - 1, int(lm.y * h)))
    return (x, y)


def draw_skeleton(
    bgr_frame: np.ndarray,
    pose_frame: PoseFrame,
    joint_statuses: Dict[str, bool],
) -> None:
    """Draw skeleton lines and landmark dots on bgr_frame in-place.

    Lines are RED by default; GREEN when the pivot joint is in correct range.
    Dots are drawn in cyan on top of lines.
    """
    h, w = bgr_frame.shape[:2]
    correct_set = {name for name, ok in joint_statuses.items() if ok}

    # Draw connection lines
    for (a_idx, b_idx) in SKELETON_CONNECTIONS:
        pt_a = _get_pt(pose_frame, a_idx, w, h)
        pt_b = _get_pt(pose_frame, b_idx, w, h)
        if pt_a is None or pt_b is None:
            continue
        color = COLOR_CORRECT if b_idx.name in correct_set else COLOR_WRONG
        cv2.line(bgr_frame, pt_a, pt_b, color, 2, cv2.LINE_AA)

    # Draw landmark dots on top
    for lm in pose_frame.landmarks:
        if lm.visibility > 0:
            cx = max(0, min(w - 1, int(lm.x * w)))
            cy = max(0, min(h - 1, int(lm.y * h)))
            cv2.circle(bgr_frame, (cx, cy), 5, COLOR_DOT, -1, cv2.LINE_AA)
            cv2.circle(bgr_frame, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(
    bgr_frame: np.ndarray,
    skill_label: str,
    timer_secs: float,
    in_position: bool,
) -> None:
    """Draw a semi-transparent HUD panel in the top-left corner."""
    h, w = bgr_frame.shape[:2]
    panel_w, panel_h = 330, 95

    # Safe overlay: copy first, draw on copy, blend back
    overlay = bgr_frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), COLOR_HUD_BG, -1)
    # Blend: 55% overlay (dark rect) + 45% original
    roi = bgr_frame[8:8 + panel_h, 8:8 + panel_w]
    ov_roi = overlay[8:8 + panel_h, 8:8 + panel_w]
    blended = cv2.addWeighted(ov_roi, 0.55, roi, 0.45, 0)
    bgr_frame[8:8 + panel_h, 8:8 + panel_w] = blended

    # Skill name
    cv2.putText(bgr_frame, f"Skill: {skill_label}", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2, cv2.LINE_AA)

    # Timer
    mins   = int(timer_secs) // 60
    secs   = int(timer_secs) % 60
    centis = int((timer_secs - int(timer_secs)) * 100)
    timer_str   = f"{mins:02d}:{secs:02d}.{centis:02d}"
    timer_color = COLOR_TIMER_OK if in_position else COLOR_TIMER_IDLE
    cv2.putText(bgr_frame, timer_str, (18, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, timer_color, 2, cv2.LINE_AA)

    # Status badge
    status_txt   = "IN POSITION" if in_position else "MONITORING"
    status_color = COLOR_CORRECT if in_position else (120, 120, 120)
    cv2.putText(bgr_frame, status_txt, (190, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)


def draw_feedback(
    bgr_frame: np.ndarray,
    joint_statuses: Dict[str, bool],
) -> None:
    """Draw per-joint correctness feedback in the bottom-left corner."""
    h, w = bgr_frame.shape[:2]
    y_base = h - 15
    for name, ok in reversed(list(joint_statuses.items())):
        color  = COLOR_CORRECT if ok else COLOR_WRONG
        symbol = "OK" if ok else "!!"
        label  = f"{symbol}  {name.replace('_', ' ').title()}"
        cv2.putText(bgr_frame, label, (12, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
        y_base -= 22


def draw_no_pose_warning(bgr_frame: np.ndarray) -> None:
    """Show a warning when no pose is detected in the frame."""
    h, w = bgr_frame.shape[:2]
    msg = "No pose detected - step into view"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = (w - tw) // 2
    y = h // 2
    # Dark background behind text for readability
    cv2.rectangle(bgr_frame, (x - 8, y - 22), (x + tw + 8, y + 8), (30, 30, 30), -1)
    cv2.putText(bgr_frame, msg, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)


def draw_controls_hint(bgr_frame: np.ndarray) -> None:
    """Draw key hint at bottom-right: Q=quit, SPACE=reset timer."""
    h, w = bgr_frame.shape[:2]
    hint = "[Q] Quit   [SPACE] Reset Timer"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(bgr_frame, hint, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
