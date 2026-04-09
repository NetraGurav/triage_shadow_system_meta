"""
model/schema.py
────────────────────────────────────────────────────────────────────────────────
Pydantic schema for structured LLM agent responses.
"""

from pydantic import BaseModel, Field
from typing import Literal


class TriageResponse(BaseModel):
    """Schema for the LLM agent's structured triage response."""

    category: Literal["hardware", "software", "security", "network", "access"] = Field(
        description="The IT category of the issue (lowercase)"
    )
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="P1 is critical/urgent, P4 is low priority"
    )
    route: Literal[
        "auth_team", "network_team", "hardware_team", "security_team", "dev_team"
    ] = Field(description="The specific team handle for routing")
    reasoning: str = Field(description="Brief explanation of why this was chosen")