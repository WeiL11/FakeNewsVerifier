# FakeNewsVerifier

A system for extracting, verifying, and correcting claims from news using a claim–source graph, causal checks, and optional RL tuning.

---

## What we built

- **Pipeline:** Raw text → **Claim Extractor** (splits into atomic claims) → **Verifier** (NetworkX claim–source graph + mock confidence) → **Corrector** (causal intervention when confidence &lt; threshold) → structured report.
- **Phase 2:** Docker image and Kubernetes manifests (Deployment, Job, Service) for Minikube; `run_minikube.py` to run steps locally on Mac.
- **Phase 3:** RL tuner (PPO/RLlib) to optimize correction threshold and causal params from labeled data (`data/train_claims.json`); optional `--rl-tune` in main.
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

### Option D — Phase 3 (RL tuner)

1. **Install PyTorch (for PPO)**  
   `pip install torch`  
   (Skip if you only want default params when `--rl-tune` is used.)

2. **Run verification with RL-tuned params**  
   `python src/main.py --rl-tune --input "Your claim here."`  
   (Uses `data/train_claims.json` by default; trains PPO for 5 iterations then verifies with tuned threshold.)

3. **Optional: custom train data and iterations**  
   `python src/main.py --rl-tune --train-data data/train_claims.json --rl-iterations 10 --input "Your claim."`

---

## Structure

- **src/** — Agents (claim_extractor, verifier, corrector), graph, causal, rl_tuner, main.
- **k8s/** — deployment.yaml, service.yaml, job.yaml, configmap.yaml; [k8s/README.md](k8s/README.md) for Minikube.
- **data/** — sample_claims.json, train_claims.json.
- **tests/** — test_main.py, test_rl_tuner.py.

## Functions & data

See the **Functions** and **Data** tables in the repo (or in-code docstrings) for one-line summaries and parameters.
