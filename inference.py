"""
inference.py
────────────────────────────────────────────────────────────────────────────────
Full evaluation pipeline for the IT Ticket Triage OpenEnv system.

This script:
  1. Initialises the environment (reset / step / state)
  2. Trains the SFT baseline model
  3. Runs evaluation across ALL tickets in ALL three tasks (easy/medium/hard)
  4. Prints structured per-ticket logs with reward and field-level correctness
  5. Outputs aggregate scores per task (0.0 – 1.0)
  6. Optionally runs the LLM agent if API credentials are provided

Usage
─────
    python inference.py                      # SFT model only
    python inference.py --agent llm          # LLM agent (requires env vars)
    python inference.py --task easy          # single task
    python inference.py --ticket E001        # single specific ticket

Environment variables (for LLM mode):
    API_BASE_URL  – e.g. https://api.openai.com/v1
    MODEL_NAME    – e.g. gpt-4o-mini
    HF_TOKEN      – your API token
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from typing import Dict, List, Any

from dotenv import load_dotenv

from env.environment import TicketTriageEnv
from model.sft_model import TriageModel
from data.tickets import get_tickets_by_difficulty, get_ticket_by_id, TICKETS


# ─── ANSI colour helpers ──────────────────────────────────────────────────────

USE_COLOUR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text

GREEN  = lambda s: _c(s, "92")
RED    = lambda s: _c(s, "91")
YELLOW = lambda s: _c(s, "93")
CYAN   = lambda s: _c(s, "96")
BOLD   = lambda s: _c(s, "1")
DIM    = lambda s: _c(s, "2")


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width

load_dotenv()

def _header(title: str) -> None:
    print()
    print(BOLD(_sep("═")))
    print(BOLD(f"  {title}"))
    print(BOLD(_sep("═")))

def _section(title: str) -> None:
    print()
    print(CYAN(_sep("─")))
    print(CYAN(f"  {title}"))
    print(CYAN(_sep("─")))

def _tick_cross(ok: bool) -> str:
    return GREEN("✓") if ok else RED("✗")


# ─── Evaluation per ticket ────────────────────────────────────────────────────

def evaluate_ticket(
    env: TicketTriageEnv,
    agent_predict_fn,
    ticket: Dict,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one episode: reset → predict → step → collect results.
    """
    task = ticket["difficulty"]
    model_name = os.environ.get("MODEL_NAME", "sft-baseline")

    print(f"[START] task={task} env=it-ticket-triage model={model_name}", flush=True)

    # ── reset ─────────────────────────────────────────────────────────────────
    obs = env.reset(task=task, ticket_id=ticket["id"])

    # ── predict ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    prediction = agent_predict_fn(ticket["text"])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Normalise prediction shape (LLM agent returns {"action": {...}} wrapper)
    if "action" in prediction:
        action    = prediction["action"]
        reasoning = prediction.get("reasoning", "")
        source    = prediction.get("source", "unknown")
        confidence = prediction.get("confidence", {})
    else:
        action    = prediction
        reasoning = ""
        source    = "sft"
        confidence = {}

    # ── step ──────────────────────────────────────────────────────────────────
    _, reward, done, info = env.step(action)

    info["reasoning"]   = reasoning
    info["source"]      = source
    info["confidence"]  = confidence
    info["latency_ms"]  = round(elapsed_ms, 2)

    action_str = json.dumps(action)
    print(f"[STEP] step=1 action={action_str} reward={reward:.2f} done=true error=null", flush=True)

    score = info["agent_score"]
    success = "true" if score >= 0.5 else "false"
    print(f"[END] success={success} steps=1 score={score:.3f} reward={reward:.2f}", flush=True)

    return info


# ─── Task-level evaluation ────────────────────────────────────────────────────

def evaluate_task(
    env: TicketTriageEnv,
    agent_predict_fn,
    task: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all tickets for a given task level.
    """
    tickets = get_tickets_by_difficulty(task)
    if not tickets:
        return {"task": task, "score": 0.0, "tickets": []}

    results = []
    for ticket in tickets:
        info = evaluate_ticket(env, agent_predict_fn, ticket, verbose=verbose)
        results.append(info)

    # Aggregate
    scores  = [r["agent_score"] for r in results]
    rewards = [r["reward"]      for r in results]
    optimal = [r["is_optimal"]  for r in results]

    cat_acc   = sum(r["field_correct"]["category"] for r in results) / len(results)
    pri_acc   = sum(r["field_correct"]["priority"]  for r in results) / len(results)
    route_acc = sum(r["field_correct"]["route"]     for r in results) / len(results)

    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "task":          task,
        "score":         round(mean_score, 4),
        "num_tickets":   len(results),
        "mean_reward":   round(sum(rewards)/len(rewards), 4) if rewards else 0.0,
        "optimal_rate":  round(sum(optimal)/len(optimal), 4) if optimal else 0.0,
        "accuracy": {
            "category": round(cat_acc, 4),
            "priority":  round(pri_acc, 4),
            "route":     round(route_acc, 4),
        },
        "tickets": results,
    }


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_full_evaluation(
    agent: str = "sft",
    tasks: List[str] = ["easy", "medium", "hard"],
    ticket_id: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full OpenEnv evaluation pipeline.
    """
    env = TicketTriageEnv(seed=42)
    state = env.state()

    if agent == "sft":
        model = TriageModel().fit()
        predict_fn = model.predict_with_confidence
    elif agent == "llm":
        from model.llm_agent import LLMTriageAgent
        llm_agent = LLMTriageAgent(use_fallback=True)
        predict_fn = llm_agent.predict
    else:
        raise ValueError(f"Unknown agent: {agent!r}. Choose 'sft' or 'llm'.")

    if ticket_id is not None:
        ticket = get_ticket_by_id(ticket_id)
        if ticket is None:
            sys.exit(1)
        info = evaluate_ticket(env, predict_fn, ticket, verbose=True)
        return {"single_ticket": info}

    task_results = {}
    for task in tasks:
        result = evaluate_task(env, predict_fn, task, verbose=verbose)
        task_results[task] = result

    overall_scores = []
    for task, result in task_results.items():
        score = result["score"]
        overall_scores.append(score)

    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    summary = {
        "overall_score": round(overall, 4),
        "tasks": {
            task: {
                "score":        result["score"],
                "optimal_rate": result["optimal_rate"],
                "accuracy":     result["accuracy"],
            }
            for task, result in task_results.items()
        },
    }

    return {"summary": summary, "tasks": task_results}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IT Ticket Triage — OpenEnv Inference & Evaluation"
    )
    parser.add_argument(
        "--agent",
        choices=["sft", "llm"],
        default="sft",
        help="Agent to use: 'sft' (default) or 'llm' (requires API env vars)",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task level to evaluate (default: all)",
    )
    parser.add_argument(
        "--ticket",
        default=None,
        help="Evaluate a specific ticket by ID (e.g. E001, M003, H007)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-ticket logs; only print summary",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    tasks = (
        ["easy", "medium", "hard"]
        if args.task == "all"
        else [args.task]
    )

    run_full_evaluation(
        agent=args.agent,
        tasks=tasks,
        ticket_id=args.ticket,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
