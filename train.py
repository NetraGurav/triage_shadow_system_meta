"""
train.py
────────────────────────────────────────────────────────────────────────────────
Standalone training script for the SFT baseline model.

Run this ONCE before inference to pre-train and serialise the model.
(inference.py also trains the model in-memory if no saved model exists,
but running train.py first is recommended for faster startup.)

Usage:
    python train.py
    python train.py --output model/saved_model.pkl
    python train.py --eval     # also run leave-one-out cross-validation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from model.sft_model import TriageModel
from data.tickets import TICKETS

load_dotenv()


def train(output_path: str = "model/saved_model.pkl", run_eval: bool = False):
    print("=" * 60)
    print("  IT Ticket Triage — SFT Model Training")
    print("=" * 60)
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"[1/3] Training on {len(TICKETS)} tickets...")
    model = TriageModel()
    model.fit()
    print("      Training complete.")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"[2/3] Saving model to '{output_path}'...")
    model.save(output_path)

    # ── Quick evaluation ──────────────────────────────────────────────────────
    print("[3/3] Quick in-sample evaluation...")
    correct = {"category": 0, "priority": 0, "route": 0}
    total   = len(TICKETS)

    for ticket in TICKETS:
        result = model.predict_with_confidence(ticket["text"])
        action = result["action"]
        label  = ticket["label"]
        for field in correct:
            if action[field] == label[field]:
                correct[field] += 1

    print()
    print("  In-sample accuracy (training data):")
    for field, count in correct.items():
        acc = count / total * 100
        print(f"    {field:<12}: {count}/{total} = {acc:.1f}%")

    print()
    print("  NOTE: In-sample accuracy is optimistic.")
    print("  Use the OpenEnv evaluation pipeline (inference.py) for true metrics.")

    # ── Optional leave-one-out evaluation ─────────────────────────────────────
    if run_eval:
        print()
        print("  Running leave-one-out cross-validation...")
        loo_correct = {"category": 0, "priority": 0, "route": 0}

        for i, held_out in enumerate(TICKETS):
            # Train on all except held-out
            train_tickets = [t for j, t in enumerate(TICKETS) if j != i]
            loo_model = TriageModel()

            # Monkey-patch: temporarily override get_all_tickets
            import data.tickets as dt
            original = dt.TICKETS
            dt.TICKETS = train_tickets
            loo_model.fit()
            dt.TICKETS = original

            action = loo_model.predict(held_out["text"])
            label  = held_out["label"]
            for field in loo_correct:
                if action[field] == label[field]:
                    loo_correct[field] += 1

        print()
        print("  Leave-one-out accuracy:")
        for field, count in loo_correct.items():
            acc = count / total * 100
            print(f"    {field:<12}: {count}/{total} = {acc:.1f}%")

    print()
    print("  Training complete. Run `python inference.py` to evaluate.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train the SFT triage model")
    parser.add_argument("--output", default="model/saved_model.pkl",
                        help="Path to save the trained model")
    parser.add_argument("--eval", action="store_true",
                        help="Run leave-one-out cross-validation")
    args = parser.parse_args()
    train(output_path=args.output, run_eval=args.eval)


if __name__ == "__main__":
    main()
