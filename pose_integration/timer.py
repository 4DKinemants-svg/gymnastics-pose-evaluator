"""timer.py - Pose hold timer for the 4D Kinematics evaluator.

SkillTimer tracks how long a gymnast stays continuously in position.
It auto-starts when in_position=True and pauses when they break form.
Pressing SPACE (handled externally) calls reset() to zero the timer.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class HoldRecord:
    """Records a single continuous in-position hold."""
    start_wall: float
    end_wall: float

    @property
    def duration(self) -> float:
        return self.end_wall - self.start_wall


class SkillTimer:
    """Tracks cumulative hold time and individual hold streaks.

    Usage:
        timer = SkillTimer()
        # each frame:
        timer.update(in_position)
        elapsed = timer.elapsed      # current hold seconds
        total   = timer.total        # cumulative hold across all holds
    """

    def __init__(self) -> None:
        self._in_position: bool = False
        self._hold_start: float = 0.0
        self._elapsed_hold: float = 0.0   # current streak duration
        self._total_hold: float = 0.0     # all completed holds
        self._records: List[HoldRecord] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, in_position: bool) -> None:
        """Call once per processed frame with the current in-position bool."""
        now = time.monotonic()

        if in_position and not self._in_position:
            # Transitioned INTO position - start a new streak
            self._in_position = True
            self._hold_start = now

        elif not in_position and self._in_position:
            # Transitioned OUT of position - record the hold
            self._in_position = False
            duration = now - self._hold_start
            self._total_hold += duration
            self._records.append(HoldRecord(self._hold_start, now))
            self._elapsed_hold = 0.0

        if self._in_position:
            self._elapsed_hold = now - self._hold_start

    def reset(self) -> None:
        """Reset current streak (keep historical total)."""
        if self._in_position:
            # Finalize any running hold before reset
            now = time.monotonic()
            self._total_hold += now - self._hold_start
            self._records.append(HoldRecord(self._hold_start, now))
        self._in_position = False
        self._hold_start = 0.0
        self._elapsed_hold = 0.0

    def full_reset(self) -> None:
        """Reset everything including totals (new session)."""
        self._in_position = False
        self._hold_start = 0.0
        self._elapsed_hold = 0.0
        self._total_hold = 0.0
        self._records.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def in_position(self) -> bool:
        return self._in_position

    @property
    def elapsed(self) -> float:
        """Seconds of the CURRENT continuous hold (0 if not in position)."""
        return self._elapsed_hold

    @property
    def total(self) -> float:
        """Cumulative seconds held across all holds this session."""
        return self._total_hold + self._elapsed_hold

    @property
    def best_hold(self) -> float:
        """Longest single continuous hold recorded."""
        if not self._records:
            return self._elapsed_hold
        return max(r.duration for r in self._records)

    @property
    def hold_count(self) -> int:
        """Number of completed holds."""
        return len(self._records)

    def summary(self) -> dict:
        return {
            "total_hold_secs": round(self.total, 2),
            "best_hold_secs": round(self.best_hold, 2),
            "hold_count": self.hold_count,
        }
