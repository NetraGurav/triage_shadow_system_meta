"""
env/environment.py
────────────────────────────────────────────────────────────────────────────────
OpenEnv-compliant environment for IT Ticket Triage.

Implements the three mandatory methods:
    reset(task, ticket_id=None) -> TriageObservation
    step(action)               -> (TriageObservation, float, bool, dict)
    state()                    -> EnvState

Shadow RL (core innovation)
───────────────────────────
On every step() call the environment:
  1. Scores the agent's actual action.
  2. Exhaustively evaluates ALL possible actions (5 × 4 × 5 = 100 combos).
  3. Finds the best possible score achievable.
  4. Computes regret-based reward = agent_score - best_possible_score

This gives a dense, informative training signal even when the agent is wrong.
Reward is always ≤ 0.0:
    0.0   → optimal decision
    -0.3  → agent scored 0.7 while optimal was 1.0
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple

from data.tickets import get_tickets_by_difficulty, get_all_tickets, get_ticket_by_id
from env.features import extract_features, fit_vectorizer
from env.graders import get_grader
from env.models import (
    TriageObservation,
    TriageAction,
    TriageReward,
    FieldCorrectness,
    StepInfo,
    EnvState,
    ActionSpaceInfo,
)

# ─── Action space definition ──────────────────────────────────────────────────

CATEGORIES  = ["hardware", "software", "network", "access", "security"]
PRIORITIES  = ["P1", "P2", "P3", "P4"]
ROUTES      = ["auth_team", "network_team", "hardware_team", "security_team", "dev_team"]

# All possible (category, priority, route) combinations: 5 × 4 × 5 = 100
ALL_ACTIONS: List[Dict[str, str]] = [
    {"category": c, "priority": p, "route": r}
    for c, p, r in itertools.product(CATEGORIES, PRIORITIES, ROUTES)
]


# ─── Environment ──────────────────────────────────────────────────────────────

class TicketTriageEnv:
    """
    OpenEnv-compliant IT Ticket Triage Environment.

    Usage
    -----
    env = TicketTriageEnv()
    obs = env.reset(task="easy")

    action = {"category": "hardware", "priority": "P1", "route": "hardware_team"}
    obs, reward, done, info = env.step(action)

    print(info["agent_score"])        # 0.0 – 1.0 correctness score
    print(info["best_possible_score"])# best achievable score
    print(reward)                     # regret = agent_score - best (≤ 0)
    print(info["shadow_scores"])      # list of all 100 action scores
    """

    VALID_TASKS = {"easy", "medium", "hard"}

    def __init__(self, seed: int = 42):
        self._seed = seed
        random.seed(seed)

        # Fit TF-IDF vectoriser on the full corpus at init time
        corpus = [t["text"] for t in get_all_tickets()]
        fit_vectorizer(corpus)

        # Internal state
        self._task: Optional[str] = None
        self._ticket: Optional[Dict] = None
        self._observation: Optional[TriageObservation] = None
        self._done: bool = True
        self._step_count: int = 0
        self._episode_history: List[Dict] = []

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        task: str = "easy",
        ticket_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.

        Parameters
        ----------
        task      : difficulty level — "easy" | "medium" | "hard"
        ticket_id : (optional) specific ticket ID to load. If None, random.

        Returns
        -------
        observation : structured feature dict (compliant with openenv.yaml)
        """
        if task not in self.VALID_TASKS:
            raise ValueError(f"Invalid task {task!r}. Choose from {self.VALID_TASKS}")

        self._task = task

        if ticket_id is not None:
            ticket = get_ticket_by_id(ticket_id)
            if ticket is None:
                raise ValueError(f"Ticket '{ticket_id}' not found.")
        else:
            pool = get_tickets_by_difficulty(task)
            if not pool:
                raise RuntimeError(f"No tickets available for task '{task}'")
            ticket = random.choice(pool)

        self._ticket = ticket

        # Build observation using Pydantic model
        raw_features = extract_features(ticket)
        self._observation = TriageObservation(**raw_features)
        self._done = False
        self._step_count = 0

        return self._observation.model_dump()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        action: Dict[str, str],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Apply agent action and compute regret-based reward.

        Parameters
        ----------
        action : dict with keys "category", "priority", "route"

        Returns
        -------
        observation       : same observation (episode is single-step)
        reward            : float ≤ 0.0  (regret-based RL signal)
        done              : True (each ticket is a one-shot decision)
        info              : rich diagnostic dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Validate action using Pydantic model
        validated_action = TriageAction(**action)
        action_dict = validated_action.model_dump()

        label   = self._ticket["label"]
        grader  = get_grader(self._task)

        # 1. Score the actual action
        agent_score = grader.score(action_dict, label)

        # 2. Shadow RL: evaluate all possible actions
        shadow_scores = self._shadow_evaluate(grader, label)

        # 3. Best achievable score
        best_possible_score = max(shadow_scores.values())
        best_action = max(shadow_scores, key=shadow_scores.get)

        # 4. Regret-based reward (always ≤ 0)
        reward = agent_score - best_possible_score

        # 5. Build typed reward
        reward_obj = TriageReward(
            reward=reward,
            agent_score=agent_score,
            best_possible_score=best_possible_score,
            is_optimal=(reward == 0.0),
        )

        # 6. Build field correctness
        field_correct = FieldCorrectness(
            category=action_dict.get("category") == label["category"],
            priority=action_dict.get("priority") == label["priority"],
            route=action_dict.get("route") == label["route"],
        )

        # 7. Build info payload
        info = {
            "ticket_id":          self._ticket["id"],
            "ticket_text":        self._ticket["text"],
            "task":               self._task,
            "agent_action":       action_dict,
            "ground_truth":       label,
            "agent_score":        agent_score,
            "best_possible_score": best_possible_score,
            "best_action":        _action_key_to_dict(best_action),
            "reward":             reward,
            "is_optimal":         reward == 0.0,
            "reward_details":     reward_obj.model_dump(),
            "shadow_scores":      {
                k: round(v, 4) for k, v in shadow_scores.items()
            },
            "field_correct": field_correct.model_dump(),
        }

        self._done = True
        self._step_count += 1
        self._episode_history.append(info)

        return self._observation.model_dump(), reward, True, info

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state snapshot.

        Includes the current observation plus meta-information about
        the episode and environment configuration.
        """
        env_state = EnvState(
            task=self._task,
            ticket_id=self._ticket["id"] if self._ticket else None,
            done=self._done,
            step_count=self._step_count,
            observation=self._observation,
            action_space=ActionSpaceInfo(),
            episode_history_length=len(self._episode_history),
        )
        return env_state.model_dump()

    # ── Shadow RL (private) ───────────────────────────────────────────────────

    def _shadow_evaluate(
        self,
        grader,
        label: Dict[str, str],
    ) -> Dict[str, float]:
        """
        Score every possible action and return a mapping:
            "category|priority|route" -> score
        """
        results = {}
        for action in ALL_ACTIONS:
            key = _dict_to_action_key(action)
            results[key] = grader.score(action, label)
        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def episode_summary(self) -> Dict[str, Any]:
        """Aggregate stats over the current session's episodes."""
        if not self._episode_history:
            return {"episodes": 0}
        scores  = [ep["agent_score"] for ep in self._episode_history]
        rewards = [ep["reward"]      for ep in self._episode_history]
        optimal = [ep["is_optimal"]  for ep in self._episode_history]
        return {
            "episodes":            len(self._episode_history),
            "mean_agent_score":    round(sum(scores)  / len(scores),  4),
            "mean_reward":         round(sum(rewards) / len(rewards), 4),
            "optimal_rate":        round(sum(optimal) / len(optimal), 4),
        }


# ─── Helper functions ─────────────────────────────────────────────────────────

def _dict_to_action_key(action: Dict[str, str]) -> str:
    return f"{action['category']}|{action['priority']}|{action['route']}"


def _action_key_to_dict(key: str) -> Dict[str, str]:
    parts = key.split("|")
    return {"category": parts[0], "priority": parts[1], "route": parts[2]}
