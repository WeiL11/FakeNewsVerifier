# FakeNewsVerifier

A system for extracting, verifying, and correcting claims from news using a claim–source graph, causal checks, and optional RL tuning.

## Structure

- **src/** — Core logic: agents (claim extractor, verifier, corrector), graph (networkx), causal (sympy), RL tuner (Phase 3)
- **k8s/** — Kubernetes manifests for deployment
- **data/** — Sample and training datasets (see [Data](#data))
- **notebooks/** — Jupyter experiments
- **tests/** — Unit and integration tests

## Quick start

```bash
pip install -r requirements.txt
python src/main.py
```

## Functions

One-line summary each; parameters get a short follow-up where relevant.

| Module | Function | Summary |
|--------|----------|---------|
| **main** | `run_verification(input_text, correction_params=None)` | Runs extract → verify → correct → report; returns a report dict. *Params:* `correction_params` — optional dict from RL tuner (`intervention_threshold`, `causal_bias`, `causal_truth`). |
| **main** | `_build_rl_dataset(train_path)` | Loads `train_claims.json` and runs extract+verify per claim; returns list of `(verification_results, graph, ground_truth)` for RL. |
| **agents.claim_extractor** | `extract_claims(text)` | Splits raw text into 1–3 atomic claims (naive: periods + assertion keywords). Returns list of claim strings. |
| **agents.verifier** | `verify_claims(claims)` | Builds claim–source graph and assigns mock confidence per claim. Returns `(results, graph)`; each result has `claim`, `confidence`, `status`. |
| **agents.corrector** | `correct_results(verification_results, graph, params=None)` | Applies causal intervention when confidence &lt; threshold. *Params:* `params` — optional `intervention_threshold`, `causal_bias`, `causal_truth`. Returns `(corrected_results, graph)`. |
| **graph** | `build_claim_graph(claims, sources=None)` | Builds a directed graph: claim nodes → source nodes. *Params:* `sources` — list of source names (default: mock Web/DB/post). Returns NetworkX `DiGraph`. |
| **graph** | `graph_summary(G)` | Returns a short string: e.g. `"Graph: N claims • M sources • E edges"`. |
| **causal** | `simple_causal_intervention(claim, assumed_cause="source_bias", bias=0.4, truth=0.85)` | Toy causal model: confidence = truth − bias; returns `adjusted_confidence` and intervention text. *Params:* `bias`, `truth` — tunable (e.g. by RL). |
| **rl_tuner** | `tune_correction(dataset, num_iterations=5, seed=None)` | Runs PPO on tuner env; returns optimized `{intervention_threshold, causal_bias, causal_truth}`. *Params:* `dataset` — list of `(v_res, graph, ground_truth)`; `num_iterations` — PPO training steps. |

## Data

| File | Summary |
|------|---------|
| **data/sample_claims.json** | Sample claim records for manual/testing use; one example claim. |
| **data/train_claims.json** | Labeled claims for Phase 3 RL: each item has `claim` and `ground_truth` (`is_fake`, `expected_conf`). Used by `--rl-tune`. |

## Docker

```bash
docker build -t fake-news-verifier:phase2 .
docker run fake-news-verifier:phase2
```

## Kubernetes + Minikube (Mac)

1. Build image: `docker build -t fake-news-verifier:phase2 .`
2. Use Minikube’s Docker: `eval $(minikube docker-env)` (in each terminal where you build).
3. Start cluster: `minikube start`
4. Apply manifests: `kubectl apply -f k8s/deployment.yaml` (or `k8s/job.yaml` for one-shot).
5. View logs: `kubectl logs -f deployment/verifier-deployment`

See **[k8s/README.md](k8s/README.md)** for the full Minikube runbook.

## Phase 3: RL tuner

Closed-loop tuning of correction params (when to intervene, causal bias/truth) via PPO (RLlib). Reward: match ground-truth confidence; penalize over-correction.

```bash
python src/main.py --rl-tune --input "Your claim here."
python src/main.py --rl-tune --train-data data/train_claims.json --rl-iterations 10
```

**CLI:** `--rl-tune` (enable), `--train-data` (path to `train_claims.json`), `--rl-iterations` (default 5). Phase 3 needs PyTorch: `pip install torch` if missing.

**Risks / future work:** Overfitting to toy data; real X/Twitter data and K8s batch jobs are for later.
