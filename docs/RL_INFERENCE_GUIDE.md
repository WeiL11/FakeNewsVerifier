# RLlib Inference Fix Guide (2025/2026)

## Problem

- PPO trains successfully (~20 iterations, mean reward ~33)
- Inference fails: `'SingleAgentEnvRunner' object has attribute 'get_policy'` (deprecated)
- Learned params are never applied → before/after confidences identical

## Solution: Use New API Stack

**Keep the new API stack.** Do not disable it. Use `get_module()` + `forward_inference()`.

### 1. Recommended Pattern

```python
# After training
module = algo.get_module()
obs_batch = torch.from_numpy(np.array([obs], dtype=np.float32))
fwd = module.forward_inference({"obs": obs_batch})
logits = fwd.get("action_dist_inputs")
action = int(torch.argmax(logits, dim=-1)[0].item())
```

### 2. Critical: Use `num_env_runners=1`

With multiple env runners, `get_module()` can return `None` (module lives on remote workers). Use `num_env_runners=1` so the module is local:

```python
config = (
    PPOConfig()
    .environment(...)
    .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
)
```

### 3. Checkpoint Save / Load

**Save after training:**
```python
algo.save("/path/to/checkpoint")
```

**Load for inference:**
```python
from ray.rllib.algorithms.algorithm import Algorithm
algo = Algorithm.from_checkpoint("/path/to/checkpoint")
module = algo.get_module()
# ... run forward_inference as above
```

### 4. Pass Tuned Params to correct_results()

```python
best_params = {
    "intervention_threshold": best_threshold,  # from THRESHOLDS[action]
    "causal_bias": 0.4,
    "causal_truth": 0.85,
}
corrected, _ = correct_results(v_res, graph, params=best_params)
```

## Minimal Repro

```bash
python scripts/test_rllib_inference.py
```

Expected: `OK: inference works with get_module + forward_inference`

## Validation

```bash
python -m src.main --input data/train_claims.json --rl-tune --rl-iterations 5
```

Check logs for:
- `intervention_threshold` different from 0.55 (e.g. 0.4, 0.5, 0.6)
- `Before (default params):` vs `After (tuned params):` — confidences should differ on some claims
