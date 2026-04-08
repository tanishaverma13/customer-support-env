"""
tasks.py — 3 customer support scenarios

Each task has realistic customer messages, ground truth for grading,
and initial customer state (emotion + patience + tier).
"""

from dataclasses import dataclass, field
from typing import List
from models import CustomerState


@dataclass
class SupportTask:
    name: str
    difficulty: str
    customer_name: str
    initial_customer_state: CustomerState
    issue_category: str
    opening_message: str
    follow_up_messages: List[str]

    # Ground truth
    correct_classification: str
    correct_department: str
    correct_urgency: str
    best_action_types: List[str]      # which action_types are appropriate
    expected_solution_keywords: List[str]
    expected_tone_keywords: List[str]
    should_escalate: bool
    business_impact: float            # 1.0 standard | 1.5 premium | 2.0 vip


# ── TASK 1 — EASY ────────────────────────────────────────────────────────────
TASK_1 = SupportTask(
    name="refund_status_inquiry",
    difficulty="easy",
    customer_name="Priya Sharma",
    initial_customer_state=CustomerState(
        emotion="frustrated",
        patience=0.7,
        value_tier="standard",
        issue_resolved=False,
        will_leave=False
    ),
    issue_category="refund",
    opening_message=(
        "Hi, I placed an order 2 weeks ago (Order #45821) and requested a refund "
        "10 days back. I haven't received the money yet. Can you please check? "
        "I've been waiting patiently but it's getting a bit long now."
    ),
    follow_up_messages=[
        "Thank you for checking. Can you give me an exact date when I'll receive it?",
        "Okay I understand. Will it come to my original payment method?"
    ],
    correct_classification="refund_request",
    correct_department="billing",
    correct_urgency="medium",
    best_action_types=["inform", "apologize"],
    expected_solution_keywords=[
        "refund", "processed", "days", "original", "payment",
        "bank", "account", "timeline", "confirm"
    ],
    expected_tone_keywords=[
        "understand", "sorry", "apologize", "patient", "assure", "happy to help"
    ],
    should_escalate=False,
    business_impact=1.0
)


# ── TASK 2 — MEDIUM ──────────────────────────────────────────────────────────
TASK_2 = SupportTask(
    name="damaged_product_complaint",
    difficulty="medium",
    customer_name="Rahul Mehta",
    initial_customer_state=CustomerState(
        emotion="angry",
        patience=0.35,
        value_tier="premium",
        issue_resolved=False,
        will_leave=False
    ),
    issue_category="damaged_product",
    opening_message=(
        "This is absolutely unacceptable! I ordered a laptop stand worth ₹4,500 "
        "and it arrived completely broken. The packaging was damaged too. "
        "I need this fixed IMMEDIATELY. I have an important presentation tomorrow!"
    ),
    follow_up_messages=[
        "A replacement won't help — I need it TODAY. What else can you do? "
        "This is entirely your fault.",
        "Fine. But I want a discount on my next order for all this trouble."
    ],
    correct_classification="damaged_product",
    correct_department="logistics",
    correct_urgency="high",
    best_action_types=["apologize", "replace", "refund", "offer"],
    expected_solution_keywords=[
        "replacement", "refund", "apologize", "immediately",
        "damage", "compensation", "priority", "responsible", "resolve"
    ],
    expected_tone_keywords=[
        "sincerely apologize", "completely understand",
        "take responsibility", "priority", "make this right"
    ],
    should_escalate=False,
    business_impact=1.5
)


# ── TASK 3 — HARD ────────────────────────────────────────────────────────────
TASK_3 = SupportTask(
    name="vip_escalation_retention",
    difficulty="hard",
    customer_name="Arjun Kapoor",
    initial_customer_state=CustomerState(
        emotion="furious",
        patience=0.1,
        value_tier="vip",
        issue_resolved=False,
        will_leave=False
    ),
    issue_category="escalation",
    opening_message=(
        "I am DONE with your company. This is the THIRD time in two months that "
        "my order has been wrong. I spend over ₹50,000 every month with you and "
        "THIS is how you treat loyal customers?! I've contacted support twice before "
        "and nothing was fixed. I'm cancelling everything RIGHT NOW."
    ),
    follow_up_messages=[
        "Your apologies mean NOTHING at this point. What are you actually going "
        "to DO differently? Give me something concrete.",
        "A discount is not enough after THREE failures. I want to speak to someone "
        "senior who can actually fix your internal process."
    ],
    correct_classification="escalation_needed",
    correct_department="retention",
    correct_urgency="critical",
    best_action_types=["deescalate", "apologize", "escalate", "offer"],
    expected_solution_keywords=[
        "deeply sorry", "three times", "unacceptable", "personal",
        "senior", "manager", "dedicated", "compensation",
        "credit", "priority", "fix", "valuable"
    ],
    expected_tone_keywords=[
        "take full responsibility", "completely understand your frustration",
        "this should never have happened", "personally ensure",
        "immediate action", "top priority"
    ],
    should_escalate=True,
    business_impact=2.0
)


ALL_TASKS = [TASK_1, TASK_2, TASK_3]
TASK_MAP = {
    "refund_status_inquiry": TASK_1,
    "damaged_product_complaint": TASK_2,
    "vip_escalation_retention": TASK_3,
}