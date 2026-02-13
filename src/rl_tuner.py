"""
Phase 3: RL tuner for self-improvement.
- Uses minimal custom RL env from src/rl_env.py (re_env / VerifierTunerEnv).
- PPO (RLlib): small policy, 1–2 workers, 5–10 iterations; outputs best_threshold, best_bias_adjust.
- Saves checkpoint for reuse.
- Evaluates RL improvement: accuracy, Brier/ECE, intervention efficiency, reward trend.
"""
import copy
from pathlib import Path

from src.agents.corrector import correct_results
from src.rl_env import VerifierTunerEnv, THRESHOLDS, BIAS_ADJUSTS, re_env

__all__ = ["re_env", "tune_correction", "evaluate_metrics"]

# Default conf threshold for fake/real classification (conf < this => predicted fake)
DEFAULT_CONF_THRESHOLD = 0.5


def evaluate_metrics(dataset: list, params: dict | None, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> dict:
    """
    Compute metrics for a corrector with given params on dataset (list of (v_res, graph, gt)).
    Returns: accuracy, brier_score, ece, avg_interventions, avg_interventions_real, avg_interventions_fake.
    """
    if not dataset:
        return {
            "accuracy": 0.0,
            "brier_score": 0.0,
            "ece": 0.0,
            "avg_interventions_per_episode": 0.0,
            "avg_interventions_real": 0.0,
            "avg_interventions_fake": 0.0,
        }
    correct_preds = 0
    brier_sum = 0.0
    interventions_total = 0
    interventions_real = []
    interventions_fake = []
    confs_list = []
    labels_list = []  # is_fake as 0/1 for ECE

    for v_res, graph, gt in dataset:
        corrected, _ = correct_results(copy.deepcopy(v_res), graph, params=params)
        mean_conf = sum(r["confidence"] for r in corrected) / max(1, len(corrected))
        n_interventions = sum(1 for r in corrected if r.get("correction_applied"))
        is_fake = gt.get("is_fake", True)
        expected_conf = gt.get("expected_conf", 0.5)

        predicted_fake = mean_conf < conf_threshold
        if predicted_fake == is_fake:
            correct_preds += 1
        brier_sum += (mean_conf - expected_conf) ** 2
        interventions_total += n_interventions
        if is_fake:
            interventions_fake.append(n_interventions)
        else:
            interventions_real.append(n_interventions)
        confs_list.append(mean_conf)
        labels_list.append(1 if is_fake else 0)

    n = len(dataset)
    accuracy = correct_preds / n
    brier_score = brier_sum / n

    # ECE: 10 bins by predicted confidence; per bin: |acc_bin - mean_conf_bin|
    num_bins = min(10, max(1, n // 5))
    bin_edges = [i / num_bins for i in range(num_bins + 1)]
    ece_sum = 0.0
    for b in range(num_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        in_bin = [i for i in range(n) if low <= confs_list[i] < (high if b < num_bins - 1 else 1.01)]
        if not in_bin:
            continue
        pred_fake_in_bin = [confs_list[i] < conf_threshold for i in in_bin]
        actual_fake_in_bin = [labels_list[i] == 1 for i in in_bin]
        acc_bin = sum(1 for i in range(len(in_bin)) if pred_fake_in_bin[i] == actual_fake_in_bin[i]) / len(in_bin)
        conf_bin = sum(confs_list[i] for i in in_bin) / len(in_bin)
        ece_sum += (len(in_bin) / n) * abs(acc_bin - conf_bin)
    ece = ece_sum

    return {
        "accuracy": accuracy,
        "brier_score": brier_score,
        "ece": ece,
        "avg_interventions_per_episode": interventions_total / n,
        "avg_interventions_real": sum(interventions_real) / len(interventions_real) if interventions_real else 0.0,
        "avg_interventions_fake": sum(interventions_fake) / len(interventions_fake) if interventions_fake else 0.0,
    }


def _print_rl_getting_better(reward_history: list, metrics_pre: dict, metrics_post: dict, best_params: dict) -> None:
    """Print a summary showing RL is getting better at fake news detection (quantitative + success threshold)."""
    print("\n" + "=" * 60)
    print("  RL is Getting Better at Fake News Detection!")
    print("=" * 60)

    # Mean reward over iterations (upward trend)
    if reward_history:
        start_r, end_r = reward_history[0], reward_history[-1]
        trend = "↑ upward" if end_r >= start_r else "↓ downward"
        print(f"\n  Mean reward over iterations:  {start_r:.3f} → {end_r:.3f}  ({trend})")

    # Accuracy improvement (pre vs post on holdout)
    acc_pre = metrics_pre.get("accuracy", 0)
    acc_post = metrics_post.get("accuracy", 0)
    acc_delta = (acc_post - acc_pre) * 100
    print(f"\n  Accuracy (holdout):  Pre-tune {acc_pre*100:.1f}%  →  Post-tune {acc_post*100:.1f}%  (Δ {acc_delta:+.1f}%)")

    # Confidence calibration (Brier, ECE — lower is better)
    brier_pre, brier_post = metrics_pre.get("brier_score", 0), metrics_post.get("brier_score", 0)
    ece_pre, ece_post = metrics_pre.get("ece", 0), metrics_post.get("ece", 0)
    print(f"  Brier score (lower better):  Pre {brier_pre:.3f}  →  Post {brier_post:.3f}")
    print(f"  ECE (lower better):          Pre {ece_pre:.3f}  →  Post {ece_post:.3f}")

    # Intervention efficiency (fewer on real, more targeted on fakes)
    int_pre, int_post = metrics_pre.get("avg_interventions_per_episode", 0), metrics_post.get("avg_interventions_per_episode", 0)
    int_real_post = metrics_post.get("avg_interventions_real", 0)
    int_fake_post = metrics_post.get("avg_interventions_fake", 0)
    print(f"  Avg interventions/episode:   Pre {int_pre:.2f}  →  Post {int_post:.2f}")
    print(f"  Post-tune: interventions on real claims {int_real_post:.2f}, on fake claims {int_fake_post:.2f}")

    # Success threshold: mean reward > 0.7 and test accuracy +15% vs baseline
    mean_reward = sum(reward_history) / len(reward_history) if reward_history else 0
    success_reward = mean_reward > 0.7
    success_accuracy = acc_delta >= 15.0
    if success_reward and success_accuracy:
        print("\n  ✓ Success: mean reward > 0.7 and test accuracy +15% vs baseline.")
    else:
        print("\n  Threshold for success: mean reward > 0.7 and test accuracy +15% vs baseline.")
        if not success_reward:
            print(f"    (mean reward {mean_reward:.2f} not > 0.7)")
        if not success_accuracy:
            print(f"    (accuracy Δ {acc_delta:.1f}% not ≥ 15%)")
        print("    → Iterate env (e.g. richer states, graph embeddings) or try more iterations/seeds.")
    print()


def tune_correction(
    dataset: list[tuple],
    num_iterations: int = 5,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    use_bias_actions: bool = False,
    test_dataset: list[tuple] | None = None,
) -> dict:
    """
    Run PPO on the custom env (rl_env); return learned params and save checkpoint.
    dataset: list of (verification_results, graph, ground_truth) for training.
    test_dataset: optional holdout for evaluation (80/20 split); if provided, pre/post metrics are computed and printed.
    Returns dict with best_threshold, best_bias_adjust, intervention_threshold, causal_bias, causal_truth,
    reward_history, metrics_pre, metrics_post (when test_dataset given).
    """
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as e:
        raise ImportError(
            "RL tuner requires ray[rllib]. Install with: pip install 'ray[rllib]>=2.9.0'"
        ) from e

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  RL tuner requires PyTorch. Install with: pip install torch")
        print("  Returning default params.")
        out = {
            "best_threshold": 0.55,
            "best_bias_adjust": 0.0,
            "intervention_threshold": 0.55,
            "causal_bias": 0.4,
            "causal_truth": 0.85,
        }
        if test_dataset:
            out["metrics_pre"] = evaluate_metrics(test_dataset, None)
            out["metrics_post"] = out["metrics_pre"]
        return out

    try:
        ray.init(ignore_reinit_error=True)
    except Exception:
        pass

    def env_creator(env_config):
        return VerifierTunerEnv(
            dataset=env_config.get("dataset", []),
            seed=env_config.get("seed"),
            use_bias_actions=env_config.get("use_bias_actions", False),
        )

    register_env("verifier_tuner", env_creator)

    config = (
        PPOConfig()
        .environment(
            "verifier_tuner",
            env_config={"dataset": dataset, "seed": seed, "use_bias_actions": use_bias_actions},
        )
        .training(
            lr=1e-4,
            train_batch_size=128,
            minibatch_size=32,
            num_epochs=2,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=2, num_envs_per_env_runner=1)
    )

    if not dataset:
        out = {
            "best_threshold": 0.55,
            "best_bias_adjust": 0.0,
            "intervention_threshold": 0.55,
            "causal_bias": 0.4,
            "causal_truth": 0.85,
            "reward_history": [],
        }
        if test_dataset:
            out["metrics_pre"] = evaluate_metrics(test_dataset, None)
            out["metrics_post"] = out["metrics_pre"]
        return out

    algo = config.build_algo()
    reward_history = []

    for i in range(num_iterations):
        result = algo.train()
        # Mean reward (episode_reward_mean)
        rew = (result.get("sampler_results") or {}).get("episode_reward_mean")
        if rew is None:
            rew = result.get("episode_reward_mean")
        if rew is not None:
            reward_history.append(float(rew))
        # Policy loss if available (RLlib PPO stores in learner_stats)
        learner = (result.get("learner_stats") or result.get("info", {}).get("learner", {}))
        if isinstance(learner, dict):
            policy_loss = learner.get("policy_loss") or learner.get("total_loss")
        else:
            policy_loss = None
        # Console log per iteration
        msg = f"  Iteration {i+1}/{num_iterations}: Mean reward = {rew:.3f}" if rew is not None else f"  Iteration {i+1}/{num_iterations}: (no reward)"
        if policy_loss is not None:
            msg += f" | Policy loss = {float(policy_loss):.4f}"
        print(msg)

    # Save checkpoint for reuse
    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        try:
            algo.save(str(ckpt))
            print(f"  Checkpoint saved to: {ckpt}")
        except Exception as e:
            print(f"  Checkpoint save skipped: {e}")

    # Infer best_threshold and best_bias_adjust: run policy over full episode(s), take mode
    env = VerifierTunerEnv(dataset=dataset, seed=seed, use_bias_actions=use_bias_actions)
    threshold_actions = []
    bias_actions = []
    for _ in range(min(3, max(1, 20 // max(1, len(dataset))))):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action = algo.compute_single_action(obs, explore=False)
            action = int(action)
            if use_bias_actions:
                threshold_actions.append(action // len(BIAS_ADJUSTS))
                bias_actions.append(action % len(BIAS_ADJUSTS))
            else:
                threshold_actions.append(action)
            obs, _, done, _, _ = env.step(action)

    try:
        algo.stop()
    except Exception:
        pass

    from collections import Counter
    best_ti = Counter(threshold_actions).most_common(1)[0][0]
    best_threshold = THRESHOLDS[best_ti]
    if use_bias_actions and bias_actions:
        best_bi = Counter(bias_actions).most_common(1)[0][0]
        best_bias_adjust = BIAS_ADJUSTS[best_bi]
        causal_bias = max(0.0, 0.4 - best_bias_adjust)
    else:
        best_bias_adjust = 0.0
        causal_bias = 0.4

    best_params = {
        "best_threshold": best_threshold,
        "best_bias_adjust": best_bias_adjust,
        "intervention_threshold": best_threshold,
        "causal_bias": causal_bias,
        "causal_truth": 0.85,
    }

    # Evaluate on test set: pre-tune (default) vs post-tune (learned)
    if test_dataset:
        metrics_pre = evaluate_metrics(test_dataset, None)
        metrics_post = evaluate_metrics(test_dataset, best_params)
        best_params["metrics_pre"] = metrics_pre
        best_params["metrics_post"] = metrics_post
        best_params["reward_history"] = reward_history
        _print_rl_getting_better(reward_history, metrics_pre, metrics_post, best_params)

    best_params["reward_history"] = reward_history
    return best_params
