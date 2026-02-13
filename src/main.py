"""
FakeNewsVerifier — Entry point: text → extract → verify → correct → report.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path when run as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.claim_extractor import extract_claims
from src.agents.verifier import verify_claims
from src.agents.corrector import correct_results
from src.graph import graph_summary


def run_verification(input_text, correction_params=None, verbose=True):
    """
    Run extract → verify → correct → report.
    correction_params: optional dict (intervention_threshold, causal_bias, causal_truth) from RL tuner.
    verbose: if False, no console output (for before/after comparison).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("FAKE NEWS VERIFIER - Phase 1 Demo")
        print("=" * 60)
        if correction_params:
            print("(Using RL-tuned correction params)")
        print(f"Input: {input_text}\n")

    # Step 1: Extract
    claims = extract_claims(input_text)
    if verbose:
        print(f"Extracted {len(claims)} claim(s):")
        for i, c in enumerate(claims, 1):
            print(f"  {i}. {c}")
        print()

    # Step 2: Verify
    verification_results, graph = verify_claims(claims)
    if verbose:
        print("Initial Verification:")
        for r in verification_results:
            print(f"• {r['claim'][:60]}{'...' if len(r['claim']) > 60 else ''} → {r['status']} ({r['confidence']})")
        print(f"\n{graph_summary(graph)}\n")

    # Step 3: Correct (with optional RL-tuned params)
    corrected_results, _ = correct_results(verification_results, graph, params=correction_params)
    if verbose:
        print("After Self-Correction:")
        for r in corrected_results:
            corr = f" ({r['correction_applied']})" if r.get("correction_applied") else ""
            print(f"• {r['claim'][:60]}{'...' if len(r['claim']) > 60 else ''} → {r['status']} ({r['confidence']}){corr}")
        print()

    report = {
        "original_input": input_text,
        "claims": claims,
        "final_verification": corrected_results,
        "graph_summary": graph_summary(graph),
    }
    if correction_params:
        report["correction_params"] = correction_params
    return report


def _build_rl_dataset(train_path):
    """Load train_claims.json and run extract+verify for each; return list of (v_res, graph, gt)."""
    from src.agents.claim_extractor import extract_claims
    from src.agents.verifier import verify_claims

    data = json.loads(Path(train_path).read_text())
    dataset = []
    for item in data:
        claim = item.get("claim", "")
        gt = item.get("ground_truth", {})
        claims = extract_claims(claim)
        v_res, graph = verify_claims(claims)
        dataset.append((v_res, graph, gt))
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeNewsVerifier Phase 1 Simulator")
    parser.add_argument(
        "--input",
        type=str,
        default="Breaking: 7.2 magnitude earthquake hits Nashville today!",
        help="Claim or post to verify",
    )
    parser.add_argument("--output", type=str, default=None, help="Save report to JSON file")
    parser.add_argument(
        "--rl-tune",
        action="store_true",
        help="Run RL tuner on train data, then verify with tuned params",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to train_claims.json (default: data/train_claims.json)",
    )
    parser.add_argument(
        "--rl-iterations",
        type=int,
        default=5,
        help="PPO training iterations for RL tuner (default: 5)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Save RL checkpoint to this dir when using --rl-tune (e.g. checkpoints/ppo)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test/holdout claims JSON (default: 20%% of train data)",
    )
    parser.add_argument(
        "--complex-mode",
        action="store_true",
        help="Run complex multi-agent pipeline (Understand → Judge → Search Decider → Comparator → Reporter)",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        pass

    correction_params = None
    if args.rl_tune and not args.complex_mode:
        train_path = args.train_data or str(ROOT / "data" / "train_claims.json")
        if not Path(train_path).exists():
            print(f"Train data not found: {train_path}. Skipping RL tune.")
        else:
            print("Phase 3: RL tuner — building dataset and training...")
            full_dataset = _build_rl_dataset(train_path)
            # 80/20 train/test split for evaluation (RL getting better metrics)
            split = max(1, int(len(full_dataset) * 0.8))
            train_dataset = full_dataset[:split]
            test_dataset = full_dataset[split:]
            if args.test_data and Path(args.test_data).exists():
                test_dataset = _build_rl_dataset(args.test_data)
            print(f"  Train size: {len(train_dataset)}, test (holdout) size: {len(test_dataset)}")
            from src.rl_tuner import tune_correction
            correction_params = tune_correction(
                train_dataset,
                num_iterations=args.rl_iterations,
                checkpoint_dir=args.checkpoint,
                test_dataset=test_dataset if test_dataset else None,
            )
            print(f"  Tuned params: {correction_params}")
            # Before/after report for test claim (e.g. initial conf 0.6 → tuned 0.2 for fake claim)
            before_report = run_verification(args.input, correction_params=None, verbose=False)
            after_report = run_verification(args.input, correction_params=correction_params, verbose=False)
            before_confs = [r["confidence"] for r in before_report["final_verification"]]
            after_confs = [r["confidence"] for r in after_report["final_verification"]]
            before_mean = sum(before_confs) / len(before_confs) if before_confs else 0
            after_mean = sum(after_confs) / len(after_confs) if after_confs else 0
            print("\n--- Before/After (test claim) ---")
            print(f"  Before (default params): mean final confidence = {before_mean:.2f}")
            print(f"  After (tuned params):    mean final confidence = {after_mean:.2f}")
            for i, r in enumerate(before_report["final_verification"]):
                snippet = r["claim"][:50] + "..." if len(r["claim"]) > 50 else r["claim"]
                print(f"  Claim {i+1}: {before_confs[i]:.2f} → {after_confs[i]:.2f}  ({snippet})")

    if args.complex_mode:
        from src.complex_pipeline import run_complex_pipeline
        report = run_complex_pipeline(args.input, verbose=True)
        print("\nFinal Report (complex mode):")
        print(json.dumps(report, indent=2, default=str))
    else:
        report = run_verification(args.input, correction_params=correction_params)
        print("\nFinal Report Summary:")
        print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to: {args.output}")
