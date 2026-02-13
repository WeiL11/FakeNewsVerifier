"""Tests for RL tuner (env only; no PPO to avoid heavy deps in CI)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_tuner_env_reset_step() -> None:
    from src.rl_env import re_env, THRESHOLDS

    # One sample: fake verification_results + graph + ground_truth
    v_res = [{"claim": "test", "confidence": 0.4, "status": "Low"}]
    import networkx as nx
    G = nx.DiGraph()
    G.add_node("c1", type="claim", text="test", confidence=0.4)
    G.add_node("s1", type="source")
    G.add_edge("c1", "s1", weight=0.7)
    gt = {"is_fake": True, "expected_conf": 0.2}
    dataset = [(v_res, G, gt)]

    env = re_env(dataset=dataset, seed=42)
    obs, info = env.reset(seed=42)
    # State: [conf1..conf10, num_edges_norm, num_sources_norm] = 12
    assert obs.shape == (12,)
    assert obs.dtype.kind == "f"

    obs2, reward, terminated, truncated, info = env.step(0)  # threshold 0.4
    assert obs2.shape == (12,)
    assert isinstance(reward, float)
    assert terminated is True  # one sample -> done after one step
    assert 0 <= env.action_space.n == len(THRESHOLDS)