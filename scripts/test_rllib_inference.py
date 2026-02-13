#!/usr/bin/env python3
"""
Minimal repro: CartPole PPO → extract tuned action via get_module + forward_inference.
Run: python scripts/test_rllib_inference.py
Validates the 2025/2026 RLlib inference pattern used in rl_tuner.py.
"""
import numpy as np
import torch

def main():
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .training(lr=1e-4, train_batch_size=128, minibatch_size=32, num_epochs=1)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
    )

    algo = config.build_algo()

    # Train a few iterations
    for i in range(5):
        result = algo.train()
        rew = (result.get("env_runners") or {}).get("episode_return_mean") or result.get("episode_reward_mean")
        print(f"Iter {i+1}: mean_reward={rew}")

    # Save checkpoint
    ckpt_path = "/tmp/rllib_cartpole_test"
    algo.save(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Inference: get_module + forward_inference (modern API)
    module = algo.get_module()
    assert module is not None, "get_module() returned None"

    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)

    obs_batch = torch.from_numpy(np.array([obs], dtype=np.float32))
    fwd = module.forward_inference({"obs": obs_batch})
    logits = fwd.get("action_dist_inputs")
    action = int(torch.argmax(logits, dim=-1)[0].item())
    print(f"Inference: obs={obs[:4]}, action={action}")

    # Load from checkpoint and run inference again
    from ray.rllib.algorithms.algorithm import Algorithm
    algo2 = Algorithm.from_checkpoint(ckpt_path)
    module2 = algo2.get_module()
    fwd2 = module2.forward_inference({"obs": obs_batch})
    action2 = int(torch.argmax(fwd2["action_dist_inputs"], dim=-1)[0].item())
    print(f"From checkpoint: action={action2}")

    algo.stop()
    algo2.stop()
    print("OK: inference works with get_module + forward_inference")

if __name__ == "__main__":
    main()
