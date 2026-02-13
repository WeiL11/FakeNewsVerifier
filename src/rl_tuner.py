"""
Phase 3: RL tuner for self-improvement.
- Gym-like env: state = graph + verification features; action = correction threshold.
- PPO (RLlib): train briefly to optimize when to intervene; reward = truth-seeking vs over-correction.
- tune_correction() → optimized params (intervention_threshold, causal_bias, causal_truth).
"""
import copy
from typing import Any

import gymnasium as gym
import numpy as np

from src.agents.corrector import correct_results

# Discrete thresholds the agent can choose
THRESHOLDS = [0.4, 0.5, 0.55, 0.6, 0.7]
INTERVENTION_PENALTY = 0.05  # penalize unnecessary interventions (efficiency bonus when low)


def _mean_conf(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(r["confidence"] for r in results) / len(results)


def _count_interventions(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("correction_applied"))


class VerifierTunerEnv(gym.Env):
    """
    RLlib-compatible env: tune correction threshold to match ground-truth confidence
    while penalizing over-correction (hallucinations / false positives).
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset: list[tuple], seed=None):
        """
        dataset: list of (verification_results, graph, ground_truth).
        Each ground_truth: {"is_fake": bool, "expected_conf": float}.
        """
        super().__init__()
        self.dataset = dataset
        self._current = None
        self._rng = np.random.default_rng(seed)

        # State: [mean_verification_conf, n_edges_norm, n_claims_norm] in [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # Action: which threshold index (0..4)
        self.action_space = gym.spaces.Discrete(len(THRESHOLDS))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not self.dataset:
            raise ValueError("Empty dataset")
        idx = self._rng.integers(0, len(self.dataset))
        self._current = self.dataset[idx]
        v_res, graph, gt = self._current
        obs = self._obs(v_res, graph)
        return obs, {}

    def step(self, action: int):
        v_res, graph, gt = self._current
        threshold = THRESHOLDS[int(action)]
        params = {"intervention_threshold": threshold}
        corrected, _ = correct_results(copy.deepcopy(v_res), graph, params)
        mean_final = _mean_conf(corrected)
        n_interventions = _count_interventions(corrected)
        n_claims = max(1, len(v_res))

        expected = gt.get("expected_conf", 0.5)
        # Reward: match ground truth (truth-seeking), penalize over-intervention
        reward = -abs(mean_final - expected) - INTERVENTION_PENALTY * (
            n_interventions / n_claims
        )

        obs = self._obs(v_res, graph)
        return obs, float(reward), True, False, {}

    def _obs(self, v_res: list[dict], graph: Any) -> np.ndarray:
        mean_conf = _mean_conf(v_res)
        n_edges = graph.number_of_edges() if graph else 0
        n_claims = max(1, len(v_res))
        return np.array(
            [
                mean_conf,
                min(1.0, n_edges / 20.0),
                min(1.0, n_claims / 5.0),
            ],
            dtype=np.float32,
        )


def tune_correction(
    dataset: list[tuple],
    num_iterations: int = 5,
    seed: int | None = None,
) -> dict:
    """
    Run a short PPO training loop on the tuner env; return optimized params.
    dataset: list of (verification_results, graph, ground_truth).
    Returns dict with intervention_threshold (and optionally causal_bias, causal_truth).
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
        )

    register_env("verifier_tuner", env_creator)

    config = (
        PPOConfig()
        .environment("verifier_tuner", env_config={"dataset": dataset, "seed": seed})
        .training(
            lr=1e-4,
            train_batch_size=64,
            minibatch_size=16,
            num_epochs=2,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
    )

    if not dataset:
        return {
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

    # Infer best threshold: run policy on states and take mode
    env = VerifierTunerEnv(dataset=dataset, seed=seed)
    actions = []
    for _ in range(min(20, len(dataset) * 2)):
        obs, _ = env.reset(seed=seed)
        action = algo.compute_single_action(obs, explore=False)
        actions.append(int(action))
    try:
        algo.stop()
    except Exception:
        pass

    from collections import Counter
    best_action = Counter(actions).most_common(1)[0][0]
    best_threshold = THRESHOLDS[best_action]

    return {
        "intervention_threshold": best_threshold,
        "causal_bias": 0.4,
        "causal_truth": 0.85,
    }
