"""
Minimal custom RL environment (Gymnasium-style, RLlib compatible).
- State: vector of current confidence scores + graph stats [conf1, ..., num_edges, num_sources].
- Actions: discrete (e.g. set threshold 0.4/0.55/0.7, or bias removal 0.1/0.2/0.3).
- Reward: (1 - |final_conf - expected_conf|) + efficiency bonus - over-correction penalty.
- Episode: one full run over the batch of claims (10–20).
"""
import copy
from typing import Any

import gymnasium as gym
import numpy as np

from src.agents.corrector import correct_results

# Discrete choices: threshold (3–5) and optional bias adjustment
THRESHOLDS = [0.4, 0.5, 0.55, 0.6, 0.7]
BIAS_ADJUSTS = [0.0, 0.1, 0.2]  # increase bias removal (subtract from causal bias)
MAX_CLAIMS = 10  # pad confidence vector to fixed size
MAX_EDGES = 30
MAX_SOURCES = 10


def _mean_conf(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(r["confidence"] for r in results) / len(results)


def _count_interventions(results: list[dict]) -> int:
    return sum(1 for r in results if r.get("correction_applied"))


class VerifierTunerEnv(gym.Env):
    """
    Gymnasium-style Env (RLlib compatible).
    State: [conf1, conf2, ..., conf_MAX_CLAIMS, num_edges_norm, num_sources_norm].
    Actions: Discrete(5) threshold or Discrete(15) threshold × bias_adjust (start simple: 5).
    Reward: (1 - |final_conf - expected_conf|) + efficiency bonus - over-correction penalty.
    Episode: one full run over the dataset (one step per claim, done when batch finished).
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset: list[tuple], seed=None, use_bias_actions: bool = False):
        """
        dataset: list of (verification_results, graph, ground_truth).
        ground_truth: {"is_fake": bool, "expected_conf": float}.
        use_bias_actions: if True, actions are 15 (threshold × bias_adjust); else 5 (threshold only).
        """
        super().__init__()
        self.dataset = list(dataset)
        self.use_bias_actions = use_bias_actions
        self._index = 0
        self._rng = np.random.default_rng(seed)

        # State: confidence scores (padded) + num_edges_norm + num_sources_norm
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(MAX_CLAIMS + 2,), dtype=np.float32
        )
        if use_bias_actions:
            self.action_space = gym.spaces.Discrete(len(THRESHOLDS) * len(BIAS_ADJUSTS))
        else:
            self.action_space = gym.spaces.Discrete(len(THRESHOLDS))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if not self.dataset:
            raise ValueError("Empty dataset")
        self._index = 0
        # Optionally shuffle so episode order varies
        # self._order = self._rng.permutation(len(self.dataset))
        v_res, graph, _ = self.dataset[self._index]
        obs = self._obs(v_res, graph)
        return obs, {}

    def step(self, action: int):
        v_res, graph, gt = self.dataset[self._index]
        if self.use_bias_actions:
            ti = action // len(BIAS_ADJUSTS)
            bi = action % len(BIAS_ADJUSTS)
            threshold = THRESHOLDS[ti]
            bias_adjust = BIAS_ADJUSTS[bi]
            params = {
                "intervention_threshold": threshold,
                "causal_bias": max(0.0, 0.4 - bias_adjust),
                "causal_truth": 0.85,
            }
        else:
            threshold = THRESHOLDS[int(action)]
            params = {"intervention_threshold": threshold}

        corrected, _ = correct_results(copy.deepcopy(v_res), graph, params)
        mean_final = _mean_conf(corrected)
        n_interventions = _count_interventions(corrected)
        n_claims = max(1, len(v_res))
        expected = gt.get("expected_conf", 0.5)
        is_fake = gt.get("is_fake", True)

        # Reward: (1 - |final_conf - expected_conf|) normalized
        term1 = 1.0 - min(1.0, abs(mean_final - expected))

        # Bonus for few interventions (efficiency)
        efficiency_bonus = 0.1 * (1.0 - n_interventions / n_claims)

        # Penalty for over-correction (making real claims low-conf)
        over_correction_penalty = 0.0
        if not is_fake and mean_final < 0.5:
            over_correction_penalty = 0.2

        reward = term1 + efficiency_bonus - over_correction_penalty
        reward = float(np.clip(reward, -1.0, 1.0))

        self._index += 1
        done = self._index >= len(self.dataset)
        truncated = False
        if done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            next_v_res, next_graph, _ = self.dataset[self._index]
            next_obs = self._obs(next_v_res, next_graph)

        return next_obs, reward, done, truncated, {}

    def _obs(self, v_res: list[dict], graph: Any) -> np.ndarray:
        # Current confidence scores (padded to MAX_CLAIMS)
        confs = [r["confidence"] for r in v_res]
        if len(confs) < MAX_CLAIMS:
            confs = confs + [0.0] * (MAX_CLAIMS - len(confs))
        else:
            confs = confs[:MAX_CLAIMS]

        n_edges = graph.number_of_edges() if graph else 0
        n_sources = (
            sum(1 for n, d in graph.nodes(data=True) if d.get("type") == "source")
            if graph
            else 0
        )
        edges_norm = min(1.0, n_edges / MAX_EDGES)
        sources_norm = min(1.0, n_sources / MAX_SOURCES)

        return np.array(confs + [edges_norm, sources_norm], dtype=np.float32)


# Alias for the minimal custom RL environment
re_env = VerifierTunerEnv
