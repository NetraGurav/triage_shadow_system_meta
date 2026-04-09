"""
inference.py
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import io
import json
from typing import List, Optional, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

# Local imports
from env.environment import TicketTriageEnv
from model.llm_agent import LLMTriageAgent
from data.tickets import get_tickets_by_difficulty

# Load environment variables
load_dotenv()

# --- Configuration (Mandatory) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") # No default allowed per requirements

# Optional
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "it-ticket-triage"
SCORE_THRESHOLD = 0.5

# Fix Windows encoding issues for emojis in console
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

# --- Logging helpers (Strict Format) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --- Evaluation logic ---

async def run_episode(env: TicketTriageEnv, agent: LLMTriageAgent, ticket: Dict[str, Any], task_name: str) -> bool:
    """Run a single-step triage episode."""
    rewards: List[float] = []
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # 1. Reset
        obs = env.reset(task=ticket["difficulty"], ticket_id=ticket["id"])
        
        # 2. Predict
        prediction = agent.predict(ticket["text"])
        action = prediction["action"]
        
        # 3. Step
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        score = info["agent_score"]
        success = score >= SCORE_THRESHOLD
        
        log_step(step=1, action=json.dumps(action), reward=reward, done=True, error=None)
        
        log_end(success=success, steps=1, score=score, rewards=rewards)
        return success
        
    except Exception as e:
        log_step(step=1, action="{}", reward=0.00, done=True, error=str(e))
        log_end(success=False, steps=1, score=0.00, rewards=[0.00])
        return False

async def main() -> None:
    # Validate token
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable is missing.", file=sys.stderr)
        # We still proceed to initialization to let the Log helpers work if called, 
        # but the agent will likely crash or fallback.
    
    # Initialize components
    env = TicketTriageEnv(seed=42)
    
    # The requirement says "Participants must use OpenAI Client for all LLM calls"
    # Our LLMTriageAgent uses the OpenAI library internally.
    agent = LLMTriageAgent(use_fallback=True)
    
    # For evaluation, we typically run across a sample or a specific task.
    # Here we default to one ticket from the "easy" task to demonstrate compliance.
    task_to_run = os.getenv("TASK_NAME", "easy")
    tickets = get_tickets_by_difficulty(task_to_run)
    
    if not tickets:
        print(f"[ERROR] No tickets found for task {task_to_run}", file=sys.stderr)
        return

    # Run the first ticket in the set as the representative evaluation
    await run_episode(env, agent, tickets[0], task_to_run)

if __name__ == "__main__":
    asyncio.run(main())
