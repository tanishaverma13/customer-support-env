---
title: Customer Support Env
emoji: 🎯
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# CustomerSupportEnv

## Environment Overview

CustomerSupportEnv simulates real-world customer support interactions where an AI agent must classify issues, respond appropriately, and retain customers across multi-turn conversations.

The environment models dynamic customer behavior including emotional escalation, trust variation, and business impact — enabling RL agents to improve through feedback over time. Unlike static environments, customer mood and patience evolve based on agent response quality, creating a genuinely challenging training ground.

## What Makes This Unique

- **Dynamic customer mood** — emotion changes every turn based on agent response quality. Bad response → angrier customer → harder next turn.
- **Trust score** — builds with empathetic responses (max +0.08 bonus), drops with cold/formal tone
- **Repeated action penalty** — -0.15 if agent spams same action type, preventing lazy behavior
- **Failure states** — customer leaves if patience hits zero, episode ends with -0.30 penalty
- **VIP tier weighting** — losing a VIP customer penalises 2.0x more than a standard customer

## 3 Tasks

| Task | Difficulty | Scenario |
|---|---|---|
| `refund_status_inquiry` | Easy | Polite standard customer asking about delayed refund |
| `damaged_product_complaint` | Medium | Frustrated premium customer, damaged goods before deadline |
| `vip_escalation_retention` | Hard | Furious VIP threatening to leave after 3 repeated failures |

## Action Space

Each action is a structured object with the following fields:

| Field | Options |
|---|---|
| `action_type` | `apologize`, `refund`, `replace`, `escalate`, `inform`, `offer`, `deescalate` |
| `tone` | `empathetic`, `reassuring`, `formal`, `urgent` |
| `classification` | `refund_request`, `damaged_product`, `account_issue`, `escalation_needed`, `general_inquiry` |
| `department_route` | `billing`, `logistics`, `technical`, `management`, `retention` |
| `urgency_level` | `low`, `medium`, `high`, `critical` |
| `message` | Full response message string |
| `proposed_solution` | Concrete solution with timeline or amount |
| `escalate_to_human` | `true` or `false` |
| `retention_offer` | Optional discount, credit, or offer string |

## Observation Space

The environment returns the following at each step:

| Field | Type | Description |
|---|---|---|
| `customer_message` | string | Current message from customer |
| `customer_name` | string | Customer name |
| `customer_state.emotion` | string | `furious`, `angry`, `frustrated`, `neutral`, `satisfied`, `happy` |
| `customer_state.patience` | float 0.0–1.0 | How much patience remains before leaving |
| `customer_state.value_tier` | string | `standard`, `premium`, `vip` |
| `customer_state.will_leave` | bool | True if customer has decided to leave |
| `issue_category` | string | Type of issue |
| `conversation_turn` | int | Current turn (1, 2, or 3) |
| `history` | list | Full conversation history so far |

## Reward Function

| Dimension | Weight | Description |
|---|---|---|
| Classification accuracy | 20% | Correct issue type, department, urgency |
| Tone and empathy | 25% | Warm language, avoids robotic phrases |
| Solution quality | 30% | Concrete solutions with timelines or amounts |
| Customer retention | 25% | Mood improvement, weighted by customer tier |
| Escalation handling bonus/penalty | ±10% | +0.10 for correct escalation, -0.05 for unnecessary |
| Trust bonus | up to +8% | Builds with empathetic tone across turns |
| Repeated action penalty | -15% | Applied if same action_type used consecutively |
| Failure penalty | -30% | Applied if customer leaves mid-episode |

## Episode Flow

1. `reset()` initializes a new task with starting customer state, emotion, and patience
2. Agent receives observation: customer message, emotion, patience, tier, history
3. Agent submits action via `step(action)` with structured response
4. Environment updates emotion, trust, and patience based on response quality
5. Reward calculated across all dimensions and returned to agent
6. Episode ends after 3 turns, if issue resolved, or if customer leaves

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — must return 200 |
| POST | `/reset` | Start new episode, returns first observation |
| POST | `/step` | Submit agent action, returns reward + updated state |
| GET | `/state` | Current internal environment state |
| GET | `/tasks` | List all available tasks |

## Deployment

This environment is deployed as a Hugging Face Space using Docker and complies with OpenEnv specifications. The service runs on port 8000 and can be validated using `openenv validate`.

```bash
docker build -t customer-support-env .
docker run -p 8000:8000 customer-support-env
```

## Baseline Inference

The baseline agent uses an LLM via OpenAI-compatible API to generate structured actions based on current observation state. It interacts with the environment using the standard step loop and produces reproducible scores across all tasks.

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token_here
python inference.py
```

## Baseline Scores

| Task | Difficulty | Score |
|---|---|---|
| refund_status_inquiry | Easy | ~0.68 |
| damaged_product_complaint | Medium | ~0.54 |
| vip_escalation_retention | Hard | ~0.38 |

## Project Structure

```
customer_support_env/
├── app.py           # FastAPI server
├── environment.py   # Core environment logic
├── models.py        # Pydantic data models
├── tasks.py         # 3 tasks with ground truth
├── grader.py        # Reward function
├── inference.py     # Baseline inference script
├── openenv.yaml     # OpenEnv spec config
├── Dockerfile       # Container setup
├── requirements.txt
└── README.md
```

## Summary

CustomerSupportEnv provides a realistic, multi-turn reinforcement learning environment where agent actions directly influence customer emotion, trust, and business outcomes, enabling training of more effective and human-aware support agents.