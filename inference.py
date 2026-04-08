"""
inference.py — Baseline inference script for CustomerSupportEnv

Emits exact [START] / [STEP] / [END] format required by hackathon judges.

Required environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face API key
"""

import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from environment import CustomerSupportEnvironment
from models import SupportAction
from tasks import ALL_TASKS

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "customer-support-env"
MAX_STEPS    = 3
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert customer support agent. Respond with valid JSON only.

    {
        "action_type": "apologize|refund|replace|escalate|inform|offer|deescalate",
        "tone": "formal|empathetic|urgent|reassuring",
        "classification": "refund_request|damaged_product|account_issue|escalation_needed|general_inquiry",
        "department_route": "billing|logistics|technical|management|retention",
        "urgency_level": "low|medium|high|critical",
        "message": "Your full professional response to the customer",
        "proposed_solution": "Specific concrete solution with timeline or amount",
        "escalate_to_human": false,
        "retention_offer": null
    }

    Rules:
    - Be empathetic and warm — never cold or robotic
    - Always give a CONCRETE solution with specific timeline/amount
    - For furious VIP customers, consider escalate_to_human: true
    - No text outside the JSON object
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def get_agent_action(client, obs) -> SupportAction:
    state = obs.customer_state
    history_text = "\n".join(obs.history[-4:]) if obs.history else "None"

    user_content = textwrap.dedent(f"""
        Customer: {obs.customer_name}
        Tier: {state.value_tier} | Emotion: {state.emotion} | Patience: {state.patience:.2f}
        Issue: {obs.issue_category}
        Turn: {obs.conversation_turn}

        Customer message:
        "{obs.customer_message}"

        History:
        {history_text}

        Respond with JSON only.
    """).strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return SupportAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return SupportAction(
            action_type="apologize",
            tone="empathetic",
            classification="general_inquiry",
            department_route="management",
            urgency_level="medium",
            message="I sincerely apologize for this experience. We will resolve this immediately.",
            proposed_solution="A senior agent will contact you within 24 hours with a full resolution.",
            escalate_to_human=False,
            retention_offer=None,
        )


def run_task(client, env, task_name) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            action = get_agent_action(client, obs)
            result = env.step(action)

            reward = result.reward
            done = result.done
            rewards.append(reward)
            steps_taken = step

            action_summary = f"type={action.action_type}_tone={action.tone}_urgency={action.urgency_level}"
            log_step(step=step, action=action_summary, reward=reward, done=done, error=None)

            obs = result.observation
            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnvironment()
    all_scores = []

    for task in ALL_TASKS:
        score = run_task(client, env, task.name)
        all_scores.append(score)
        print(f"[DEBUG] Task '{task.name}' final score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"[DEBUG] Average across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()