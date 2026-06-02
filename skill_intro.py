"""skill_intro.py - On-screen skill selection, breakdown, and safety quiz.

Provides three sequential OpenCV screens before live pose evaluation:
  1. Skill selection menu (press 1-6)
  2. Skill breakdown + instructions
  3. Safety quiz (2-3 questions - must answer correctly to proceed)
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional, List, Tuple
from pose_targets import SKILL_REGISTRY, SkillTarget

# Colors
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (240, 240, 240)
HIGHLIGHT = (0, 215, 255)  # yellow
CORRECT = (0, 220, 0)      # green
WRONG = (0, 0, 220)        # red

# ---------------------------------------------------------------------------
# Skill knowledge base
# ---------------------------------------------------------------------------

SKILL_INFO = {
    "1": {
        "breakdown": [
            "HANDSTAND - Inverted balance on hands",
            "",
            "Key Points:",
            "  - Start in a lunge position, hands flat on ground",
            "  - Kick lead leg up while pushing through shoulders",
            "  - Stack shoulders over wrists, hips over shoulders",
            "  - Engage core, point toes, keep body in straight line",
            "  - Look slightly forward between hands",
            "",
            "Safety:",
            "  - Use a wall or spotter when learning",
            "  - Keep fingers spread wide for balance",
            "  - Roll forward if falling - tuck chin to chest",
        ],
        "quiz": [
            {
                "q": "Where should your shoulders be positioned?",
                "options": ["A) Behind wrists", "B) Directly over wrists", "C) In front of wrists"],
                "correct": "B",
            },
            {
                "q": "What should you do if you start falling forward?",
                "options": ["A) Lock elbows and fight it", "B) Tuck chin and roll", "C) Jump off hands"],
                "correct": "B",
            },
            {
                "q": "Is it safe to attempt handstands without prior training?",
                "options": ["A) Yes, just go for it", "B) No, start with wall support", "C) Only on soft surfaces"],
                "correct": "B",
            },
        ],
    },
    "2": {
        "breakdown": [
            "FRONT WALKOVER - Forward split entry into bridge",
            "",
            "Key Points:",
            "  - Start standing, kick one leg high in front",
            "  - Lean forward, place hands on ground",
            "  - Pass through handstand split",
            "  - Lower back leg to bridge position",
            "  - Push through shoulders to stand",
            "",
            "Safety:",
            "  - Requires flexible back and hamstrings",
            "  - Practice bridges and handstands separately first",
            "  - Use spotter for first attempts",
        ],
        "quiz": [
            {
                "q": "Should you attempt this without flexibility training?",
                "options": ["A) Yes, I'll stretch as I go", "B) No, build flexibility first", "C) Only if I hurry"],
                "correct": "B",
            },
            {
                "q": "What position do you pass through mid-skill?",
                "options": ["A) Seated split", "B) Handstand split", "C) Forward roll"],
                "correct": "B",
            },
        ],
    },
    "3": {
        "breakdown": [
            "BACK WALKOVER - Backward bridge into split",
            "",
            "Key Points:",
            "  - Start standing, arms overhead",
            "  - Lean back into bridge position",
            "  - Shift weight onto hands",
            "  - Kick one leg over into split handstand",
            "  - Lower lead leg to stand",
            "",
            "Safety:",
            "  - Requires strong bridge and back flexibility",
            "  - NEVER force your back - warm up thoroughly",
            "  - Use a spotter to support your back",
        ],
        "quiz": [
            {
                "q": "What is the first position you enter?",
                "options": ["A) Handstand", "B) Bridge", "C) Cartwheel"],
                "correct": "B",
            },
            {
                "q": "Should you force your back if you're not flexible?",
                "options": ["A) Yes, that's how you improve", "B) No, risk of injury", "C) Only a little"],
                "correct": "B",
            },
        ],
    },
    "4": {
        "breakdown": [
            "BACK HANDSPRING - Backward jump to hands then feet",
            "",
            "Key Points:",
            "  - Start standing, arms overhead",
            "  - Sit back slightly, swing arms down then up",
            "  - Jump backward, arch body",
            "  - Land on hands with arms extended",
            "  - Snap legs over to land on feet",
            "",
            "Safety:",
            "  - HIGH INJURY RISK - requires spotter",
            "  - Build strength with drills first (rebounds, back rolls)",
            "  - NEVER attempt on hard floor without mats",
            "  - Common injuries: wrist, ankle, head if under-rotated",
        ],
        "quiz": [
            {
                "q": "Is it safe to try this alone as a beginner?",
                "options": ["A) Yes, just be confident", "B) No, requires spotter and mats", "C) Only on grass"],
                "correct": "B",
            },
            {
                "q": "What's the biggest danger if under-rotated?",
                "options": ["A) Twisted ankle", "B) Landing on head/neck", "C) Sore wrists"],
                "correct": "B",
            },
        ],
    },
    "5": {
        "breakdown": [
            "CARTWHEEL - Sideways rotation through hands",
            "",
            "Key Points:",
            "  - Start in lunge, arms extended sideways",
            "  - Reach lead hand to ground",
            "  - Kick legs over in a wheel motion",
            "  - Land one foot at a time",
            "  - Keep arms and legs straight throughout",
            "",
            "Safety:",
            "  - Beginner-friendly skill",
            "  - Practice on soft surface (grass, mat)",
            "  - Keep head neutral - don't look at hands too long",
        ],
        "quiz": [
            {
                "q": "Should your arms and legs be straight?",
                "options": ["A) No, bent is fine", "B) Yes, for proper form", "C) Only arms"],
                "correct": "B",
            },
            {
                "q": "Is cartwheel considered beginner-safe?",
                "options": ["A) No, very advanced", "B) Yes, with proper form", "C) Only for experts"],
                "correct": "B",
            },
        ],
    },
    "6": {
        "breakdown": [
            "ROUNDOFF - Cartwheel with 180° twist to face backward",
            "",
            "Key Points:",
            "  - Start like cartwheel but turn shoulders early",
            "  - Bring legs together mid-air",
            "  - Snap down with both feet simultaneously",
            "  - Land facing opposite direction from start",
            "  - Used to generate power for back tumbling",
            "",
            "Safety:",
            "  - Master cartwheels first",
            "  - Land with knees slightly bent",
            "  - Practice on mats or grass",
        ],
        "quiz": [
            {
                "q": "What skill should you master first?",
                "options": ["A) Handstand", "B) Cartwheel", "C) Backflip"],
                "correct": "B",
            },
            {
                "q": "How do you land a roundoff?",
                "options": ["A) One foot at a time", "B) Both feet together", "C) On hands"],
                "correct": "B",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Screen rendering helpers
# ---------------------------------------------------------------------------

def _create_canvas(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create blank BGR canvas."""
    return np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

def _put_text_lines(canvas: np.ndarray, lines: List[str], start_y: int, color=TEXT_COLOR, scale=0.6, weight=1):
    """Draw multiple lines of text."""
    y = start_y
    for line in lines:
        cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
        y += 30

def _center_text(canvas: np.ndarray, text: str, y: int, color=HIGHLIGHT, scale=0.9, weight=2):
    """Draw centered text."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    x = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# Stage 1: Skill Selection
# ---------------------------------------------------------------------------

def _show_skill_menu() -> Optional[str]:
    """Display skill menu and return selected key (1-6) or None if quit."""
    canvas = _create_canvas()
    _center_text(canvas, "4D KINEMATICS - SELECT SKILL", 70)
    
    menu_lines = [
        "",
        "Press the number key to select:",
        "",
        "  [1] Handstand",
        "  [2] Front Walkover",
        "  [3] Back Walkover",
        "  [4] Back Handspring",
        "  [5] Cartwheel",
        "  [6] Roundoff",
        "",
        "",
        "Press [Q] to quit",
    ]
    _put_text_lines(canvas, menu_lines, 130, scale=0.7, weight=2)
    
    cv2.imshow("4D Kinematics", canvas)
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            return None
        if chr(key) in SKILL_REGISTRY:
            return chr(key)

# ---------------------------------------------------------------------------
# Stage 2: Skill Breakdown
# ---------------------------------------------------------------------------

def _show_breakdown(skill_key: str) -> bool:
    """Show skill breakdown. Returns True to proceed, False to quit."""
    info = SKILL_INFO.get(skill_key, {})
    breakdown = info.get("breakdown", ["No info available"])
    skill_label = SKILL_REGISTRY[skill_key].label
    
    canvas = _create_canvas()
    _center_text(canvas, skill_label.upper(), 50)
    _put_text_lines(canvas, breakdown, 100, scale=0.55)
    
    footer = [
        "",
        "Read carefully. Press [SPACE] to continue to safety quiz",
        "Press [B] to go back to menu, [Q] to quit",
    ]
    _put_text_lines(canvas, footer, 600, color=HIGHLIGHT, scale=0.55)
    
    cv2.imshow("4D Kinematics", canvas)
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('b'):
            return None  # signal to go back
        if key == ord(' '):
            return True

# ---------------------------------------------------------------------------
# Stage 3: Safety Quiz
# ---------------------------------------------------------------------------

def _show_quiz(skill_key: str) -> bool:
    """Run safety quiz. Returns True if passed, False if quit."""
    info = SKILL_INFO.get(skill_key, {})
    quiz_qs = info.get("quiz", [])
    skill_label = SKILL_REGISTRY[skill_key].label
    
    for i, q_data in enumerate(quiz_qs, 1):
        result = _ask_question(skill_label, i, len(quiz_qs), q_data)
        if result is None:  # quit
            return False
        if result is False:  # wrong answer
            _show_fail_screen(skill_label)
            return None  # signal to restart
    
    _show_pass_screen(skill_label)
    return True

def _ask_question(skill_label: str, q_num: int, total: int, q_data: dict) -> Optional[bool]:
    """Show one question. Returns True if correct, False if wrong, None if quit."""
    canvas = _create_canvas()
    _center_text(canvas, f"{skill_label} - Safety Quiz ({q_num}/{total})", 50)
    
    lines = [
        "",
        q_data["q"],
        "",
    ] + q_data["options"] + [
        "",
        "",
        "Press A, B, or C    |    [Q] to quit",
    ]
    _put_text_lines(canvas, lines, 130, scale=0.7, weight=2)
    
    cv2.imshow("4D Kinematics", canvas)
    
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            return None
        choice = chr(key).upper()
        if choice in ['A', 'B', 'C']:
            return choice == q_data["correct"]

def _show_fail_screen(skill_label: str):
    """Show 'incorrect answer' screen."""
    canvas = _create_canvas()
    _center_text(canvas, "INCORRECT ANSWER", 250, color=WRONG, scale=1.2)
    _center_text(canvas, f"Please review {skill_label} safety info and try again.", 320, color=TEXT_COLOR, scale=0.7, weight=1)
    _center_text(canvas, "Press any key to return to menu", 400, color=HIGHLIGHT, scale=0.6, weight=1)
    cv2.imshow("4D Kinematics", canvas)
    cv2.waitKey(0)

def _show_pass_screen(skill_label: str):
    """Show 'quiz passed' screen."""
    canvas = _create_canvas()
    _center_text(canvas, "SAFETY QUIZ PASSED!", 250, color=CORRECT, scale=1.2)
    _center_text(canvas, f"You may now proceed with {skill_label} evaluation.", 320, color=TEXT_COLOR, scale=0.7, weight=1)
    _center_text(canvas, "Press any key to start camera...", 400, color=HIGHLIGHT, scale=0.6, weight=1)
    cv2.imshow("4D Kinematics", canvas)
    cv2.waitKey(0)

# ---------------------------------------------------------------------------
# Main flow coordinator
# ---------------------------------------------------------------------------

def run_intro_flow() -> Optional[SkillTarget]:
    """Run full intro: menu -> breakdown -> quiz. Returns SkillTarget or None if quit."""
    while True:
        # Stage 1: Skill selection
        skill_key = _show_skill_menu()
        if skill_key is None:
            cv2.destroyAllWindows()
            return None
        
        # Stage 2: Breakdown
        proceed = _show_breakdown(skill_key)
        if proceed is False:
            cv2.destroyAllWindows()
            return None
        if proceed is None:  # go back
            continue
        
        # Stage 3: Quiz
        passed = _show_quiz(skill_key)
        if passed is False:
            cv2.destroyAllWindows()
            return None
        if passed is None:  # failed quiz - restart
            continue
        
        # Success!
        cv2.destroyAllWindows()
        return SKILL_REGISTRY[skill_key]
