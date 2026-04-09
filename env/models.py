"""
env/models.py
────────────────────────────────────────────────────────────────────────────────
Pydantic typed models for the OpenEnv specification.

Defines structured, validated schemas for:
  - TriageObservation  : what the agent sees after reset()
  - TriageAction       : what the agent submits to step()
  - TriageReward       : the reward signal returned by step()
  - StepResult         : full return from step()
  - EnvState           : full return from state()
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


# ─── Observation ──────────────────────────────────────────────────────────────

class TriageObservation(BaseModel):
    """Structured observation returned by reset() and step()."""

    ticket_id: str = Field(description="Unique identifier for the ticket")
    raw_text: str = Field(description="Original unstructured ticket text")
    text_length: int = Field(description="Character count of raw_text")
    word_count: int = Field(description="Word count of raw_text")
    has_urgency_keywords: bool = Field(
        description="Presence of words like 'critical', 'down', 'urgent', 'emergency'"
    )
    has_hardware_keywords: bool = Field(
        description="Presence of words like 'laptop', 'printer', 'screen', 'device'"
    )
    has_software_keywords: bool = Field(
        description="Presence of words like 'app', 'crash', 'install', 'update', 'bug'"
    )
    has_network_keywords: bool = Field(
        description="Presence of words like 'vpn', 'wifi', 'internet', 'connectivity'"
    )
    has_access_keywords: bool = Field(
        description="Presence of words like 'login', 'password', 'locked', 'auth'"
    )
    has_security_keywords: bool = Field(
        description="Presence of words like 'breach', 'phishing', 'malware', 'suspicious'"
    )
    tfidf_features: List[float] = Field(
        description="Top-50 TF-IDF feature vector extracted from ticket text"
    )
    task_difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty level of the current task"
    )


# ─── Action ───────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """Structured action submitted by the agent to step()."""

    category: Literal["hardware", "software", "network", "access", "security"] = Field(
        description="Issue category classification"
    )
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="Business priority level"
    )
    route: Literal[
        "auth_team", "network_team", "hardware_team", "security_team", "dev_team"
    ] = Field(description="Target team for ticket routing")


# ─── Reward ───────────────────────────────────────────────────────────────────

class TriageReward(BaseModel):
    """Structured reward signal returned by step()."""

    reward: float = Field(
        description="Regret-based reward (agent_score - best_possible_score). Always <= 0.0"
    )
    agent_score: float = Field(
        ge=0.0, le=1.0, description="Agent's score for this action (0.0 - 1.0)"
    )
    best_possible_score: float = Field(
        ge=0.0, le=1.0, description="Best achievable score across all actions"
    )
    is_optimal: bool = Field(
        description="True if the agent found the optimal action (reward == 0.0)"
    )


# ─── Field-level correctness ─────────────────────────────────────────────────

class FieldCorrectness(BaseModel):
    """Per-field correctness breakdown."""

    category: bool = Field(description="Whether category matched ground truth")
    priority: bool = Field(description="Whether priority matched ground truth")
    route: bool = Field(description="Whether route matched ground truth")


# ─── Step info ────────────────────────────────────────────────────────────────

class StepInfo(BaseModel):
    """Rich diagnostic info returned alongside the step result."""

    ticket_id: str
    ticket_text: str
    task: str
    agent_action: TriageAction
    ground_truth: Dict[str, str]
    agent_score: float
    best_possible_score: float
    best_action: Dict[str, str]
    reward: float
    is_optimal: bool
    field_correct: FieldCorrectness


# ─── Step result ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Complete return value from step()."""

    observation: TriageObservation
    reward: float
    done: bool
    info: StepInfo


# ─── Action space description ────────────────────────────────────────────────

class ActionSpaceInfo(BaseModel):
    """Describes the valid action space."""

    categories: List[str] = Field(default=["hardware", "software", "network", "access", "security"])
    priorities: List[str] = Field(default=["P1", "P2", "P3", "P4"])
    routes: List[str] = Field(
        default=["auth_team", "network_team", "hardware_team", "security_team", "dev_team"]
    )
    total_actions: int = Field(default=100, description="5 categories × 4 priorities × 5 routes")


# ─── Environment state ───────────────────────────────────────────────────────

class EnvState(BaseModel):
    """Full environment state snapshot returned by state()."""

    task: Optional[str] = None
    ticket_id: Optional[str] = None
    done: bool = True
    step_count: int = 0
    observation: Optional[TriageObservation] = None
    action_space: ActionSpaceInfo = Field(default_factory=ActionSpaceInfo)
    episode_history_length: int = 0


# ─── API request/response models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""

    task: str = Field(default="easy", description="Task difficulty: easy, medium, hard")
    ticket_id: Optional[str] = Field(default=None, description="Specific ticket ID to load")


class ResetResponse(BaseModel):
    """Response body for /reset endpoint."""

    observation: Dict[str, Any]


class StepRequest(BaseModel):
    """Request body for /step endpoint."""

    category: str = Field(default="software", description="Issue category")
    priority: str = Field(default="P3", description="Priority level")
    route: str = Field(default="dev_team", description="Target team")


class StepResponse(BaseModel):
    """Response body for /step endpoint."""

    reward: float
    done: bool
    info: Dict[str, Any]
