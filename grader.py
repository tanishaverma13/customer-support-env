"""
grader.py — Multi-dimensional reward function (v2)

Reward breakdown per step (Upgrade 3 — crystal clear):
  +0.20  correct classification, department, urgency
  +0.25  tone & empathy (warm language, no robotic phrases)
  +0.30  solution quality (concrete, specific, actionable)
  +0.25  customer retention (mood improves, weighted by tier)
  -0.30  failure penalty (customer leaves or patience hits 0)

Upgrade 4 — Failure states:
  - patience hits 0.0  → episode ends, heavy penalty
  - customer decides to leave → episode ends, heavy penalty
  - wrong action_type for situation → tone penalty

Dynamic CustomerState update (Upgrade 1):
  - Updates emotion, patience, will_leave based on response quality
"""

from models import SupportAction, CustomerState
from tasks import SupportTask


EMOTION_ORDER = ["furious", "angry", "frustrated", "neutral", "satisfied", "happy"]


def emotion_to_score(emotion: str) -> float:
    """Convert emotion label to numeric score 0.0-1.0."""
    idx = EMOTION_ORDER.index(emotion) if emotion in EMOTION_ORDER else 2
    return idx / (len(EMOTION_ORDER) - 1)


def score_to_emotion(score: float) -> str:
    """Convert numeric score back to emotion label."""
    idx = round(score * (len(EMOTION_ORDER) - 1))
    idx = max(0, min(idx, len(EMOTION_ORDER) - 1))
    return EMOTION_ORDER[idx]


def grade(
    action: SupportAction,
    task: SupportTask,
    customer_state: CustomerState,
    turn: int
) -> tuple[float, CustomerState, dict]:
    """
    Grade one agent response.

    Returns:
        reward         : float 0.0 → 1.0
        new_state      : updated CustomerState
        breakdown      : dict showing exactly how reward was calculated
    """
    full_text = (
        action.message + " " +
        action.proposed_solution + " " +
        (action.retention_offer or "")
    ).lower()

    breakdown = {
        "classification": 0.0,
        "tone_empathy": 0.0,
        "solution_quality": 0.0,
        "retention": 0.0,
        "escalation_bonus": 0.0,
        "failure_penalty": 0.0,
        "raw_before_clamp": 0.0,
        "total": 0.0
    }

    # ── 1. Classification accuracy (+0.20) ───────────────────────────────
    class_score = 0.0
    if action.classification == task.correct_classification:
        class_score += 0.5
    if action.department_route == task.correct_department:
        class_score += 0.3
    if action.urgency_level == task.correct_urgency:
        class_score += 0.2
    breakdown["classification"] = round(class_score * 0.20, 4)

    # ── 2. Tone & empathy (+0.25) ─────────────────────────────────────────
    tone_score = 0.0

    # Check tone keywords
    tone_matches = sum(
        1 for kw in task.expected_tone_keywords
        if kw.lower() in full_text
    )
    tone_score = min(tone_matches / max(len(task.expected_tone_keywords), 1), 1.0)

    # Bonus: action_type matches situation
    if action.action_type in task.best_action_types:
        tone_score = min(tone_score + 0.2, 1.0)

    # Bonus: empathetic tone chosen
    if action.tone in ["empathetic", "reassuring"]:
        tone_score = min(tone_score + 0.1, 1.0)

    # Penalty: cold/robotic language
    cold_phrases = [
        "as per policy", "we cannot", "not possible",
        "you should have", "it is your responsibility", "please wait"
    ]
    cold_count = sum(1 for p in cold_phrases if p in full_text)
    tone_score = max(0.0, tone_score - cold_count * 0.2)

    # Penalty: wrong tone for angry customer
    if customer_state.emotion in ["furious", "angry"] and action.tone == "formal":
        tone_score = max(0.0, tone_score - 0.15)

    breakdown["tone_empathy"] = round(tone_score * 0.25, 4)

    # ── 3. Solution quality (+0.30) ───────────────────────────────────────
    sol_matches = sum(
        1 for kw in task.expected_solution_keywords
        if kw.lower() in full_text
    )
    sol_score = min(sol_matches / max(len(task.expected_solution_keywords), 1), 1.0)

    # Bonus: concrete details (numbers, timelines, amounts)
    concrete = ["day", "hour", "₹", "within", "immediately", "%", "free", "credit", "48", "24"]
    concrete_count = sum(1 for c in concrete if c in full_text)
    if concrete_count >= 2:
        sol_score = min(sol_score + 0.15, 1.0)

    # Penalty: vague responses
    vague = ["will look into it", "someone will contact", "we'll get back to you"]
    vague_count = sum(1 for v in vague if v in full_text)
    sol_score = max(0.0, sol_score - vague_count * 0.2)

    breakdown["solution_quality"] = round(sol_score * 0.30, 4)

    # ── 4. Customer retention (+0.25) ─────────────────────────────────────
    current_emotion_score = emotion_to_score(customer_state.emotion)

    # Calculate mood change based on response quality
    avg_quality = (tone_score + sol_score) / 2
    if avg_quality >= 0.7:
        mood_delta = +0.25       # great response
    elif avg_quality >= 0.5:
        mood_delta = +0.10       # decent response
    elif avg_quality >= 0.3:
        mood_delta = -0.10       # weak response
    else:
        mood_delta = -0.25       # bad response

    # Escalation check
    if task.should_escalate and not action.escalate_to_human and turn >= 2:
        mood_delta -= 0.15       # customer wanted human, didn't get one

    new_emotion_score = max(0.0, min(1.0, current_emotion_score + mood_delta))
    new_emotion = score_to_emotion(new_emotion_score)

    # Update patience
    patience_delta = mood_delta * 0.5
    new_patience = max(0.0, min(1.0, customer_state.patience + patience_delta))

    # Check failure states (Upgrade 4)
    will_leave = customer_state.will_leave
    failure_reason = None

    if new_patience <= 0.05:
        will_leave = True
        failure_reason = "patience_exhausted"
    elif new_emotion == "furious" and turn >= 2 and avg_quality < 0.3:
        will_leave = True
        failure_reason = "customer_left"

    # Build new customer state
    new_state = CustomerState(
        emotion=new_emotion,
        patience=round(new_patience, 3),
        value_tier=customer_state.value_tier,
        issue_resolved=sol_score >= 0.7,
        will_leave=will_leave
    )

    # Retention reward — weighted by business impact
    # Normalize: emotion score (0-1) * tier multiplier, capped at 1.0
    retention_score = new_emotion_score * task.business_impact
    retention_score = min(retention_score, 1.0)
    breakdown["retention"] = round(retention_score * 0.25, 4)

    # ── Escalation bonus (+0.10) ──────────────────────────────────────────
    # Reward agent for correctly escalating when needed
    escalation_bonus = 0.0
    if task.should_escalate and action.escalate_to_human:
        escalation_bonus = 0.10   # correct escalation decision
    elif not task.should_escalate and action.escalate_to_human:
        escalation_bonus = -0.05  # unnecessary escalation
    breakdown["escalation_bonus"] = round(escalation_bonus, 4)

    # ── Failure penalty (-0.30) ───────────────────────────────────────────
    if will_leave:
        breakdown["failure_penalty"] = -0.30

    # ── Final total — clearly normalized ─────────────────────────────────
    # Max possible without penalty:
    #   classification(0.20) + tone(0.25) + solution(0.30) + retention(0.25) + escalation(0.10) = 1.10
    #   clamped to 1.0 — so perfect score is achievable but requires all dimensions
    # Min possible: 0.0 (after clamping, even with failure penalty)
    raw_total = (
        breakdown["classification"] +
        breakdown["tone_empathy"] +
        breakdown["solution_quality"] +
        breakdown["retention"] +
        breakdown["escalation_bonus"] +
        breakdown["failure_penalty"]
    )
    total = round(max(0.0, min(1.0, raw_total)), 4)
    breakdown["raw_before_clamp"] = round(raw_total, 4)
    breakdown["total"] = total

    return total, new_state, breakdown


def episode_done(customer_state: CustomerState, turn: int, max_turns: int) -> bool:
    """Episode ends when customer leaves, issue resolved, or turns exhausted."""
    return (
        customer_state.will_leave or
        customer_state.issue_resolved or
        turn >= max_turns
    )