"""
Phase 3: RL tuner for self-improvement.
- Uses minimal custom RL env from src/rl_env.py (re_env / VerifierTunerEnv).
- PPO (RLlib): small policy, 1–2 workers, 5–10 iterations; outputs best_threshold, best_bias_adjust.
- Saves checkpoint for reuse.
"""
from pathlib import Path

from src.rl_env import VerifierTunerEnv, THRESHOLDS, BIAS_ADJUSTS, re_env

__all__ = ["re_env", "tune_correction"]


def tune_correction(
    dataset: list[tuple],
    num_iterations: int = 5,
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
    use_bias_actions: bool = False,
) -> dict:
    """
    Run PPO on the custom env (rl_env); return learned params and save checkpoint.
    dataset: list of (verification_results, graph, ground_truth).
    Returns dict with best_threshold, best_bias_adjust, intervention_threshold, causal_bias, causal_truth.
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
        return {
            "best_threshold": 0.55,
            "best_bias_adjust": 0.0,
            "intervention_threshold": 0.55,
            "causal_bias": 0.4,
            "causal_truth": 0.85,
        }

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
        return {
            "best_threshold": 0.55,
            "best_bias_adjust": 0.0,
            "intervention_threshold": 0.55,
            "causal_bias": 0.4,
            "causal_truth": 0.85,
        }

    algo = config.build_algo()

    for i in range(num_iterations):
        result = algo.train()
        rew = (result.get("sampler_results") or {}).get("episode_reward_mean")
        if rew is None:
            rew = result.get("episode_reward_mean")
        if (i + 1) % max(1, num_iterations // 2) == 0 and rew is not None:
            print(f"  RL iter {i+1}/{num_iterations}  mean_reward={rew:.3f}")

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

    return {
        "best_threshold": best_threshold,
        "best_bias_adjust": best_bias_adjust,
        "intervention_threshold": best_threshold,
        "causal_bias": causal_bias,
        "causal_truth": 0.85,
    }
