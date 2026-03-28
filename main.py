#!/usr/bin/env python3
"""
Medical X-ray Adversarial Robustness Evaluation Project

Main entry point for running different experiments:
    - Training baseline model
    - Evaluating model accuracy
    - Running adversarial attacks (FGSM, PGD, C&W)
    - Generating robustness reports
    - Visualization (Grad-CAM, adversarial examples)

Usage:
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode attack --attack_type fgsm --eps 0.01
    python main.py --mode robustness
    python main.py --mode visualize
"""

import os
import sys
import argparse
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import (
    DEVICE, MODEL_PATH, RESULTS_DIR, DATA_DIR,
    BATCH_SIZE, EPOCHS, LR, EPS_LIST,
    ensure_dirs, print_config
)


def run_train(args):
    """Run model training."""
    print("\n" + "=" * 50)
    print("Running Training...")
    print("=" * 50)
    
    from training.train_baseline import main as train_main
    train_main()


def run_eval(args):
    """Run model evaluation on clean data."""
    print("\n" + "=" * 50)
    print("Running Evaluation on Clean Data...")
    print("=" * 50)
    
    from training.eval_clean import main as eval_clean_main
    eval_clean_main()
    
    print("\nGenerating ROC curve...")
    from training.ROC import main as roc_main
    roc_main()
    
    print("\nComputing confusion matrix...")
    from training.eval_confusion import main as confusion_main
    confusion_main()


def run_attack(args):
    """Run adversarial attack evaluation."""
    print("\n" + "=" * 50)
    print(f"Running {args.attack_type.upper()} Attack...")
    print("=" * 50)
    
    if args.attack_type == "fgsm":
        from attacks.fgsm import main as fgsm_main
        fgsm_main()
    elif args.attack_type == "pgd":
        from attacks.pgd import main as pgd_main
        pgd_main()
    elif args.attack_type == "all":
        print("Running all attacks...")
        from attacks.attack_runner import run_all_attacks
        run_all_attacks()
    else:
        print(f"Unknown attack type: {args.attack_type}")
        print("Available: fgsm, pgd, all")


def run_robustness(args):
    """Run comprehensive robustness evaluation."""
    print("\n" + "=" * 50)
    print("Running Robustness Evaluation...")
    print("=" * 50)
    
    from robustness.eval_robustness import main as robustness_main
    robustness_main()


def run_visualize(args):
    """Run visualization."""
    print("\n" + "=" * 50)
    print("Running Visualization...")
    print("=" * 50)
    
    from run_visualization import main as viz_main
    viz_main()
    print(f"Visualization complete. Results saved to {RESULTS_DIR}")


def run_pipeline(args):
    """Run the complete evaluation pipeline (FGSM + PGD detailed analysis)."""
    print("\n" + "=" * 50)
    print("Running Complete Pipeline...")
    print("=" * 50)
    
    print("\n[1/2] Running FGSM detailed evaluation...")
    from robustness.eval_fgsm import main as fgsm_main
    fgsm_main()
    
    print("\n[2/2] Running PGD detailed evaluation...")
    from robustness.eval_pgd import main as pgd_main
    pgd_main()


def run_full(args):
    """Run all experiments in sequence."""
    print("\n" + "=" * 50)
    print("Running Full Experiment Suite...")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print("\n[1/4] Training model (no checkpoint found)...")
        run_train(args)
    else:
        print(f"\n[1/4] Using existing model: {MODEL_PATH}")
    
    print("\n[2/4] Evaluating on clean data...")
    run_eval(args)
    
    print("\n[3/4] Running adversarial attacks...")
    args.attack_type = "all"
    run_attack(args)
    
    print("\n[4/4] Generating robustness report...")
    run_robustness(args)
    
    print("\n" + "=" * 50)
    print("All experiments completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Medical X-ray Adversarial Robustness Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode train
    python main.py --mode eval
    python main.py --mode attack --attack_type fgsm
    python main.py --mode attack --attack_type all
    python main.py --mode robustness
    python main.py --mode visualize
    python main.py --mode full
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "attack", "robustness", "visualize", "pipeline", "full", "info"],
        help="Operation mode"
    )
    
    parser.add_argument(
        "--attack_type",
        type=str,
        default="fgsm",
        choices=["fgsm", "pgd", "all"],
        help="Type of adversarial attack"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Perturbation budget (epsilon)"
    )
    
    parser.add_argument(
        "--viz_type",
        type=str,
        default="all",
        help="[Deprecated] All visualizations are now run together"
    )
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    if args.mode == "info":
        print_config()
        return
    
    print(f"\nDevice: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    
    mode_handlers = {
        "train": run_train,
        "eval": run_eval,
        "attack": run_attack,
        "robustness": run_robustness,
        "visualize": run_visualize,
        "pipeline": run_pipeline,
        "full": run_full
    }
    
    handler = mode_handlers.get(args.mode)
    if handler:
        handler(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
