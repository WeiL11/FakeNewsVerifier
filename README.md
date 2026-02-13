# FakeNewsVerifier

A system for extracting, verifying, and correcting claims from news using a claim–source graph, causal checks, and optional RL tuning.

*Ongoing: RL is not yet reliably "learning" (accuracy/reward gains vary); we are iterating on env design and inference to improve consistency.*

---

## What we built

- **Pipeline:** Raw text → **Claim Extractor** (splits into atomic claims) → **Verifier** (NetworkX claim–source graph + mock confidence) → **Corrector** (causal intervention when confidence &lt; threshold) → structured report.
- **Phase 2:** Docker image and Kubernetes manifests (Deployment, Job, Service) for Minikube; `run_minikube.py` to run steps locally on Mac.
- **Phase 3:** RL tuner (PPO/RLlib) to optimize correction threshold and causal params from labeled data (`data/train_claims.json`); optional `--rl-tune` in main.
- **Complex mode:** Multi-agent pipeline (Understand → Judge → Search Decider → [conditional search] → Comparator → Reporter) with decision gates; `--complex-mode` for X-like real-time verification.
- **Data:** `data/sample_claims.json` (sample claims), `data/train_claims.json` (labeled claims + ground truth for RL).

---

## Step-by-step: How to run

*All steps below are the ones referenced in this README; run them in order for the option you choose.*

### Option A — Local (Quick start)

1. **Install dependencies**  
   `pip install -r requirements.txt`

2. **Run verification**  
   `python src/main.py`  
   (Uses default claim; add `--input "Your claim."` to verify custom text.)

3. **Optional: save report**  
   `python src/main.py --output report.json`

4. **Optional: run tests**  
   `pytest tests/ -v`

---

### Option B — Docker

1. **Build image**  
   `docker build -t fake-news-verifier:phase2 .`

2. **Run container**  
   `docker run fake-news-verifier:phase2`  
   (Override: `docker run fake-news-verifier:phase2 python src/main.py --input "Your claim."`)

---

### Option C — Kubernetes + Minikube (Mac)

1. **Install**  
   Minikube, kubectl, Docker Desktop (see [k8s/README.md](k8s/README.md)).

2. **Build image**  
   `docker build -t fake-news-verifier:phase2 .`

3. **Use Minikube’s Docker**  
   `eval $(minikube docker-env)`  
   (Re-run in each new terminal where you build.)

4. **Start cluster**  
   `minikube start`

5. **Apply manifests**  
   `kubectl apply -f k8s/deployment.yaml`  
   (Or one-shot: `kubectl apply -f k8s/job.yaml`.)

6. **View logs**  
   `kubectl logs -f deployment/verifier-deployment`  
   (Or for job: `kubectl logs job/verifier-job-example`.)

7. **Clean up**  
   `kubectl delete -f k8s/deployment.yaml`  
   (Or `kubectl delete job verifier-job-example` if you used the job.)

**Alternative:** Run the script that does steps 2–6:  
`python run_minikube.py`  
(Requires minikube, kubectl, Docker on PATH; see [k8s/README.md](k8s/README.md).)

---

### Option D — Complex multi-agent pipeline

1. **Run complex mode**  
   `python src/main.py --complex-mode --input "Vaccines contain microchips to track people."`  
   Flow: Understand (parse claim) → Judge (fake_conf, common_sense_conflict) → Search Decider (search_need) → if gate triggered, web search + recursive Judge on reports → Comparator (final is_fake) → Reporter (score + matrix).

2. **Decision gate**  
   Search triggers when fake_conf &gt; 0.6 AND conflict &gt; 0.6 AND search_need &gt; 0.7. Tools: `web_search`, `x_keyword_search` (mocked; replace with real APIs).

3. **Tuning**  
   Use `train_claims.json` for eval; extend Phase 3 RL to optimize gate thresholds and judge heuristics later.

---

### Option E — Phase 3 (RL tuner)

1. **Install PyTorch (for PPO)**  
   `pip install torch`  
   (Skip if you only want default params when `--rl-tune` is used.)

2. **Run verification with RL-tuned params**  
   `python src/main.py --rl-tune --input "Your claim here."`  
   (Uses `data/train_claims.json` by default; trains PPO for 5 iterations then verifies with tuned threshold.)

3. **Optional: custom train data, test holdout, and iterations**  
   `python src/main.py --rl-tune --train-data data/train_claims.json --test-data data/test_claims.json --rl-iterations 10 --input "Your claim."`

**During runs**

- **Local run** (`python src/main.py --rl-tune ...`): Console logs show RL progress per iteration (e.g. “Iteration 1: Mean reward = 0.45 | Policy loss = -0.12”). After tuning, a before/after report is printed for the test claim (e.g. “Before: mean final confidence = 0.6 → After: 0.2” and per-claim lines). No plots by default (use the notebook for matplotlib).
- **K8s run** (Job with `--rl-tune`): Pod logs (`kubectl logs job/rl-tune-job`) mirror local output—full tuning cycle, then checkpoint/report.json inside the container; use a volume mount if you need to persist them. Set replicas=1 initially; multi-pod rollouts can be added later via RLlib config.
- **Expectations**: Initial rewards are typically low (~0.2–0.5); with a well-designed env they can reach 0.7+. If rewards fluctuate or don’t improve, try different seeds or review env actions/states.
- **Failures**: Overfitting (good on train, poor on new claims) or no improvement—debug env design; re-run with different seeds if convergence is poor.

**RL is Getting Better at Fake News Detection**

When you run with `--rl-tune`, the pipeline computes quantitative metrics and prints a summary so you can see that the tuned corrector distinguishes fake/real better than the baseline.

- **Core metrics** (computed in `rl_tuner.py` on a holdout):
  - **Mean reward over iterations**: Shown as start → end (e.g. 0.3 → 0.75); upward trend indicates learning.
  - **Accuracy (holdout)**: Pre-tune vs post-tune (e.g. ~60–70% → 80–90%+). Classifies by confidence threshold (conf &lt; 0.5 = fake).
  - **Confidence calibration**: Brier score and ECE (Expected Calibration Error)—lower is better (e.g. pre 0.25 → post 0.1).
  - **Intervention efficiency**: Avg interventions per episode; post-tune breakdown on real vs fake claims (fewer on real, more targeted on fakes).
- **Train/test split**: By default an 80/20 split of the train data is used for evaluation. Optionally pass `--test-data data/test_claims.json` to evaluate on a dedicated holdout (e.g. unseen claims).
- **Success threshold**: If mean reward &gt; 0.7 and test accuracy improves by ≥15% vs baseline, the run is marked as success; otherwise the summary suggests iterating on the env (e.g. richer states, graph embeddings) or more iterations/seeds.

---

## Structure

- **src/** — Agents (claim_extractor, verifier, corrector, understand, judge, search_decider, comparator, reporter), graph, causal, rl_tuner, complex_pipeline, tools, main.
- **k8s/** — deployment.yaml, service.yaml, job.yaml, configmap.yaml; [k8s/README.md](k8s/README.md) for Minikube.
- **data/** — sample_claims.json, train_claims.json, test_claims.json (holdout for RL evaluation).
- **tests/** — test_main.py, test_rl_tuner.py.

## Functions & data

See the **Functions** and **Data** tables in the repo (or in-code docstrings) for one-line summaries and parameters.
