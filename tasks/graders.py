"""
env/graders.py
────────────────────────────────────────────────────────────────────────────────
Task graders — one per difficulty level.

Each grader exposes a single method:
    score(action: dict, label: dict) -> float  (0.0 – 1.0)

Scoring philosophy
──────────────────
• Category, priority, and route are weighted independently.
• Easy:   no partial credit — exact match only.
• Medium: partial credit for adjacent priority levels.
• Hard:   partial credit + multi-issue penalty when the agent ignores
          secondary issues that affect routing confidence.

Priority adjacency (for partial credit):
    P1 <-> P2 <-> P3 <-> P4
    One step away → 0.5 partial credit for that field.
    Two+ steps away → 0.

Route–category consistency bonus (Hard only):
    If category and route are both correct → +0.05 consistency bonus.
"""

from __future__ import annotations
from typing import Dict


# ─── Helpers ──────────────────────────────────────────────────────────────────

_PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

_CATEGORY_TO_ROUTE = {
    "hardware": "hardware_team",
    "software": "dev_team",
    "network": "network_team",
    "access": "auth_team",
    "security": "security_team",
}


def _priority_score(predicted: str, actual: str, partial: bool = False) -> float:
    """Return 1.0 / 0.5 / 0.0 for priority match quality."""
    if predicted == actual:
        return 1.0
    if not partial:
        return 0.0
    p_idx = _PRIORITY_ORDER.get(predicted, -1)
    a_idx = _PRIORITY_ORDER.get(actual, -1)
    if p_idx < 0 or a_idx < 0:
        return 0.0
    gap = abs(p_idx - a_idx)
    return 0.5 if gap == 1 else 0.0


def _field_score(predicted: str, actual: str) -> float:
    return 1.0 if str(predicted).strip().lower() == str(actual).strip().lower() else 0.0


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0, 1) as required by Phase 2 validation."""
    return round(max(0.01, min(score, 0.99)), 4)


# ─── Easy Grader ──────────────────────────────────────────────────────────────

class EasyGrader:
    """
    Strict grader for unambiguous tickets.
    No partial credit — either the agent nailed it or did not.

    Weights (from openenv.yaml):
        category : 0.40
        priority : 0.35
        route    : 0.25
    """

    WEIGHTS = {"category": 0.40, "priority": 0.35, "route": 0.25}

    def score(self, action: Dict, label: Dict) -> float:
        cat   = _field_score(action.get("category", ""), label["category"])
        pri   = _priority_score(action.get("priority", ""), label["priority"], partial=False)
        route = _field_score(action.get("route", ""), label["route"])

        total = (
            self.WEIGHTS["category"] * cat
            + self.WEIGHTS["priority"] * pri
            + self.WEIGHTS["route"]   * route
        )
        return _clamp(total)


# ─── Medium Grader ────────────────────────────────────────────────────────────

class MediumGrader:
    """
    Grader for ambiguous tickets.
    Allows partial credit for priority (one step off → 50 % of priority weight).

    Weights (from openenv.yaml):
        category : 0.35
        priority : 0.35
        route    : 0.30
    """

    WEIGHTS = {"category": 0.35, "priority": 0.35, "route": 0.30}

    def score(self, action: Dict, label: Dict) -> float:
        cat   = _field_score(action.get("category", ""), label["category"])
        pri   = _priority_score(action.get("priority", ""), label["priority"], partial=True)
        route = _field_score(action.get("route", ""), label["route"])

        total = (
            self.WEIGHTS["category"] * cat
            + self.WEIGHTS["priority"] * pri
            + self.WEIGHTS["route"]   * route
        )
        return _clamp(total)


# ─── Hard Grader ──────────────────────────────────────────────────────────────

class HardGrader:
    """
    Grader for multi-issue tickets.
    Partial credit + consistency bonus + multi-issue awareness.

    Weights (from openenv.yaml):
        category : 0.30
        priority : 0.40
        route    : 0.30

    Bonuses / Penalties:
        +0.05  if both category and route are correct (consistency bonus)
        -0.10  multi_issue_penalty applied when category is wrong AND
               the expected route would have been derivable from the label
               (agent ignored a secondary critical issue)
    """

    WEIGHTS = {"category": 0.30, "priority": 0.40, "route": 0.30}
    CONSISTENCY_BONUS = 0.05
    MULTI_ISSUE_PENALTY = 0.10

    def score(self, action: Dict, label: Dict) -> float:
        cat_ok   = _field_score(action.get("category", ""), label["category"]) == 1.0
        pri_score = _priority_score(action.get("priority", ""), label["priority"], partial=True)
        route_ok  = _field_score(action.get("route", ""), label["route"]) == 1.0

        base = (
            self.WEIGHTS["category"] * (1.0 if cat_ok else 0.0)
            + self.WEIGHTS["priority"] * pri_score
            + self.WEIGHTS["route"]   * (1.0 if route_ok else 0.0)
        )

        # Consistency bonus
        if cat_ok and route_ok:
            base += self.CONSISTENCY_BONUS

        # Multi-issue penalty: wrong category AND route is derivable from correct category
        if not cat_ok:
            expected_route = _CATEGORY_TO_ROUTE.get(label["category"], "")
            if action.get("route", "") != expected_route:
                base -= self.MULTI_ISSUE_PENALTY

        return _clamp(base)


# ─── Registry ─────────────────────────────────────────────────────────────────

_EASY   = EasyGrader()
_MEDIUM = MediumGrader()
_HARD   = HardGrader()


def _extract_data(trajectory_or_action: Any, task_or_label: Any = None) -> tuple[Dict, Dict]:
    """
    Resiliently extract the last action and the ground-truth label.
    
    Supports:
        (action_dict, label_dict)
        (trajectory_list, task_dict)
    """
    action = {}
    label = {}

    # Case 1: Trajectory list (Standard OpenEnv validator)
    if isinstance(trajectory_or_action, list) and len(trajectory_or_action) > 0:
        last_step = trajectory_or_action[-1]
        
        # Action is usually in the step itself or a sub-dict
        action = last_step.get("action", {})
        
        # Label is often in 'info' or 'ground_truth' or the 'task' arg
        info = last_step.get("info", {})
        label = info.get("ground_truth") or info.get("label") or {}
        
        # If label still empty, check the second argument (task)
        if not label and isinstance(task_or_label, dict):
            label = task_or_label.get("label") or task_or_label

    # Case 2: Direct call (Our internal environment)
    elif isinstance(trajectory_or_action, dict):
        action = trajectory_or_action
        label = task_or_label or {}

    return action, label


def grade_easy(trajectory_or_action: Any, task_or_label: Any = None) -> float:
    """Resilient entry point for OpenEnv easy task validation."""
    action, label = _extract_data(trajectory_or_action, task_or_label)
    if not label:
        return 0.01 # Baseline floor
    return _EASY.score(action, label)


def grade_medium(trajectory_or_action: Any, task_or_label: Any = None) -> float:
    """Resilient entry point for OpenEnv medium task validation."""
    action, label = _extract_data(trajectory_or_action, task_or_label)
    if not label:
        return 0.01
    return _MEDIUM.score(action, label)


def grade_hard(trajectory_or_action: Any, task_or_label: Any = None) -> float:
    """Resilient entry point for OpenEnv hard task validation."""
    action, label = _extract_data(trajectory_or_action, task_or_label)
    if not label:
        return 0.01
    return _HARD.score(action, label)


def get_grader(difficulty: str):
    """Return a callable for internal environment use."""
    if difficulty == "easy":
        return _EASY
    if difficulty == "medium":
        return _MEDIUM
    if difficulty == "hard":
        return _HARD
    raise ValueError(f"Unknown difficulty: {difficulty!r}")
