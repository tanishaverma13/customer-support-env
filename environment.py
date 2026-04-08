"""
environment.py — CustomerSupportEnvironment (v2)

All 4 upgrades integrated:
1. Explicit CustomerState (emotion + patience + tier) tracked every step
2. Structured SupportAction with action_type
3. Reward breakdown logged every step
4. Failure states — customer leaving ends episode with penalty
"""

import random
from typing import Optional
from models import SupportObservation, SupportAction, SupportState, CustomerState
from tasks import ALL_TASKS, TASK_MAP, SupportTask
from grader import grade, episode_done


class StepResult:
    def __init__(
        self,
        observation: SupportObservation,
        reward: float,
        done: bool,
        info: dict
    ):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class CustomerSupportEnvironment:
    """
    RL environment for training customer support AI agents.

    Unique features:
    - Explicit customer state: emotion + patience + tier (trackable, explainable)
    - Structured action space: action_type forces intentional decisions
    - Dynamic state transitions: bad response = angrier customer = harder next turn
    - Failure states: customer leaving ends episode with -0.30 penalty
    - Reward breakdown exposed every step so judges see exact scoring
    """

    def __init__(self):
        self._state: Optional[SupportState] = None
        self._current_task: Optional[SupportTask] = None
        self._history = []

    def reset(self, task_name: Optional[str] = None) -> SupportObservation:
        """Start a fresh episode."""
        if task_name and task_name in TASK_MAP:
            task = TASK_MAP[task_name]
        else:
            # Random task when none specified — more robust evaluation
            task = random.choice(ALL_TASKS)

        self._current_task = task
        self._history = []
        self._trust = 0.5          # trust score: 0.0 = no trust, 1.0 = full trust
        self._last_action_type = None  # for repeated action penalty

        self._state = SupportState(
            task_name=task.name,
            current_turn=0,
            customer_state=task.initial_customer_state.model_copy(),
            total_reward=0.0,
            episode_done=False,
            failure_reason=None,
            max_turns=3,
        )

        return SupportObservation(
            ticket_id=f"TKT-{task.name[:8].upper()}-001",
            customer_message=task.opening_message,
            customer_name=task.customer_name,
            customer_state=task.initial_customer_state.model_copy(),
            issue_category=task.issue_category,
            conversation_turn=1,
            history=[],
            previous_agent_response=None,
        )

    def step(self, action: SupportAction) -> StepResult:
        """Agent submits response. Environment grades it, updates customer state."""
        if self._state is None or self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        self._state.current_turn += 1
        turn = self._state.current_turn

        # ── Trust Score update ───────────────────────────────────────────
        # Good tone builds trust, bad tone erodes it
        if action.tone in ["empathetic", "reassuring"]:
            self._trust = min(1.0, self._trust + 0.15)
        elif action.tone == "formal":
            self._trust = max(0.0, self._trust - 0.10)

        # ── Repeated Action Penalty ──────────────────────────────────────
        # Penalise agent for spamming same action type
        repeated_penalty = 0.0
        if self._last_action_type == action.action_type:
            repeated_penalty = -0.15
        self._last_action_type = action.action_type

        # Grade response — get reward, updated customer state, breakdown
        reward, new_customer_state, breakdown = grade(
            action=action,
            task=self._current_task,
            customer_state=self._state.customer_state,
            turn=turn,
        )

        # Apply trust bonus and repeated action penalty
        trust_bonus = round(self._trust * 0.08, 4)   # max +0.08 for full trust
        reward = round(max(0.0, min(1.0, reward + trust_bonus + repeated_penalty)), 4)
        breakdown["trust_bonus"] = trust_bonus
        breakdown["repeated_penalty"] = repeated_penalty
        breakdown["trust_score"] = round(self._trust, 3)
        breakdown["total"] = reward

        # Update internal state
        self._state.customer_state = new_customer_state
        self._state.total_reward += reward
        self._state.last_reward_breakdown = breakdown

        # Check failure states
        if new_customer_state.will_leave:
            self._state.failure_reason = breakdown.get("failure_reason", "customer_left")

        # Check done
        done = episode_done(new_customer_state, turn, self._state.max_turns)
        self._state.episode_done = done

        # Get current customer message (what customer said THIS turn)
        follow_ups = self._current_task.follow_up_messages
        if turn == 1:
            current_customer_msg = self._current_task.opening_message
        elif turn - 1 <= len(follow_ups):
            current_customer_msg = follow_ups[turn - 2] if turn > 1 else self._current_task.opening_message
        else:
            current_customer_msg = "..."

        # Build history with ACTUAL evolving conversation (bug fix)
        self._history.append(f"[Turn {turn}] Customer: {current_customer_msg[:70]}...")
        self._history.append(f"[Turn {turn}] Agent [{action.action_type}/{action.tone}]: {action.message[:70]}...")

        # Pick next customer message
        if not done and turn < len(follow_ups):
            next_msg = follow_ups[turn - 1]
        elif new_customer_state.will_leave:
            next_msg = "I've had enough. I'm taking my business elsewhere. Goodbye."
        elif new_customer_state.issue_resolved:
            next_msg = "Thank you, that resolves my issue. I appreciate your help."
        else:
            next_msg = "Is there anything else you can do for me?"

        next_obs = SupportObservation(
            ticket_id=f"TKT-{self._current_task.name[:8].upper()}-001",
            customer_message=next_msg,
            customer_name=self._current_task.customer_name,
            customer_state=new_customer_state,
            issue_category=self._current_task.issue_category,
            conversation_turn=turn + 1,
            history=self._history.copy(),
            previous_agent_response=action.message,
        )

        return StepResult(
            observation=next_obs,
            reward=reward,
            done=done,
            info={
                "task": self._current_task.name,
                "difficulty": self._current_task.difficulty,
                "turn": turn,
                "customer_emotion": new_customer_state.emotion,
                "customer_patience": new_customer_state.patience,
                "customer_retained": not new_customer_state.will_leave,
                "issue_resolved": new_customer_state.issue_resolved,
                "failure_reason": self._state.failure_reason,
                "trust_score": round(self._trust, 3),
                "repeated_penalty": repeated_penalty,
                "reward_breakdown": breakdown,
                "total_reward": self._state.total_reward,
            }
        )

    def state(self) -> SupportState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def get_all_tasks(self) -> list:
        return [
            {"name": t.name, "difficulty": t.difficulty}
            for t in ALL_TASKS
        ]