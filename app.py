from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
from models import SupportObservation, SupportAction, SupportState
from environment import CustomerSupportEnvironment, StepResult

app = FastAPI(
    title="CustomerSupportEnv",
    description="OpenEnv RL environment for training customer support AI agents. Features dynamic customer state, structured action space, and failure state detection.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

env = CustomerSupportEnvironment()

class ResetRequest(BaseModel):
    task_name: Optional[str] = None

class ResetResponse(BaseModel):
    observation: SupportObservation

class StepRequest(BaseModel):
    action: SupportAction

class StepResponse(BaseModel):
    observation: SupportObservation
    reward: float
    done: bool
    info: dict

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Live training dashboard — see your environment in action."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(dashboard_path, encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "environment": "CustomerSupportEnv", "version": "2.0.0"}

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_name=request.task_name)
        return ResetResponse(observation=obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    try:
        result: StepResult = env.step(request.action)
        return StepResponse(observation=result.observation, reward=result.reward, done=result.done, info=result.info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=SupportState)
def state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tasks")
def list_tasks():
    return {"tasks": env.get_all_tasks()}