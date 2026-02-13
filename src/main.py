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


def run_verification(input_text, correction_params=None):
    """
    Run extract → verify → correct → report.
    correction_params: optional dict (intervention_threshold, causal_bias, causal_truth) from RL tuner.
    """
    print("\n" + "=" * 60)
    print("FAKE NEWS VERIFIER - Phase 1 Demo")
    print("=" * 60)
    if correction_params:
        print("(Using RL-tuned correction params)")
    print(f"Input: {input_text}\n")

    # Step 1: Extract
    claims = extract_claims(input_text)
    print(f"Extracted {len(claims)} claim(s):")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c}")
    print()

    # Step 2: Verify
    verification_results, graph = verify_claims(claims)
    print("Initial Verification:")
    for r in verification_results:
        print(f"• {r['claim'][:60]}{'...' if len(r['claim']) > 60 else ''} → {r['status']} ({r['confidence']})")
    print(f"\n{graph_summary(graph)}\n")

    # Step 3: Correct (with optional RL-tuned params)
    corrected_results, _ = correct_results(verification_results, graph, params=correction_params)
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

    args, unknown = parser.parse_known_args()
    if unknown:
        pass

    correction_params = None
    if args.rl_tune:
        train_path = args.train_data or str(ROOT / "data" / "train_claims.json")
        if not Path(train_path).exists():
            print(f"Train data not found: {train_path}. Skipping RL tune.")
        else:
            print("Phase 3: RL tuner — building dataset and training...")
            dataset = _build_rl_dataset(train_path)
            print(f"  Dataset size: {len(dataset)}")
            from src.rl_tuner import tune_correction
            correction_params = tune_correction(
                dataset,
                num_iterations=args.rl_iterations,
                checkpoint_dir=args.checkpoint,
            )
            print(f"  Tuned params: {correction_params}")

    report = run_verification(args.input, correction_params=correction_params)

    print("\nFinal Report Summary:")
    print(json.dumps(report, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to: {args.output}")
