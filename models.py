"""
models.py — Data models for CustomerSupportEnv (v2)

Upgrades:
- Explicit CustomerState: emotion + patience + value tier (trackable, explainable)
- Structured SupportAction: action_type forces intentional decisions
- Failure state tracking: customer_left, patience_exhausted
- Reward breakdown logged every step so judges can see scoring logic
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class CustomerState(BaseModel):
    """
    Explicit customer emotional state — changes every turn.

    bad response  → patience ↓, emotion worsens
    good response → patience ↑, emotion improves

    Judges can watch this change in real time via /state endpoint.
    """
    emotion: Literal["furious", "angry", "frustrated", "neutral", "satisfied", "happy"]
    patience: float = Field(ge=0.0, le=1.0, description="0.0 = leaving, 1.0 = fully patient")
    value_tier: Literal["standard", "premium", "vip"]
    issue_resolved: bool = False
    will_leave: bool = False


class SupportObservation(BaseModel):
    """What the AI agent sees each step."""
    ticket_id: str
    customer_message: str
    customer_name: str
    customer_state: CustomerState
    issue_category: str
    conversation_turn: int
    history: List[str] = []
    previous_agent_response: Optional[str] = None


class SupportAction(BaseModel):
    """
    Structured action space — agent declares WHAT it is doing, not just text.

    action_type options:
      apologize   — acknowledge problem empathetically
      refund      — offer money back
      replace     — offer replacement product
      escalate    — hand off to senior/human agent
      inform      — give status update
      offer       — make retention offer (discount, credit)
      deescalate  — calm down angry customer

    tone options:
      formal | empathetic | urgent | reassuring
    """
    action_type: Literal[
        "apologize", "refund", "replace",
        "escalate", "inform", "offer", "deescalate"
    ]
    tone: Literal["formal", "empathetic", "urgent", "reassuring"]
    classification: Literal[
        "refund_request", "damaged_product", "account_issue",
        "escalation_needed", "general_inquiry"
    ]
    department_route: Literal[
        "billing", "logistics", "technical", "management", "retention"
    ]
    urgency_level: Literal["low", "medium", "high", "critical"]
    message: str = Field(description="Full response message to customer")
    proposed_solution: str = Field(description="Concrete solution with timeline/amount")
    escalate_to_human: bool = False
    retention_offer: Optional[str] = None


class SupportState(BaseModel):
    """Internal environment state — exposed via /state for debugging."""
    task_name: str
    current_turn: int = 0
    customer_state: CustomerState = Field(
        default_factory=lambda: CustomerState(
            emotion="neutral", patience=0.8,
            value_tier="standard", issue_resolved=False, will_leave=False
        )
    )
    total_reward: float = 0.0
    episode_done: bool = False
    failure_reason: Optional[str] = None  # 'customer_left' | 'patience_exhausted' | None
    max_turns: int = 3
    last_reward_breakdown: dict = Field(
        default_factory=lambda: {
            "classification": 0.0,
            "tone_empathy": 0.0,
            "solution_quality": 0.0,
            "retention": 0.0,
            "escalation_bonus": 0.0,
            "failure_penalty": 0.0,
            "raw_before_clamp": 0.0,
            "total": 0.0
        }
    )