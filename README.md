# FakeNewsVerifier

A system for extracting, verifying, and correcting claims from news using a claim–source graph, causal checks, and optional RL tuning.

## Structure

- **src/** — Core logic: agents (claim extractor, verifier, corrector), graph (networkx), causal (sympy), RL tuner stub
- **k8s/** — Kubernetes manifests for deployment
- **data/** — Sample datasets (large files gitignored)
- **notebooks/** — Jupyter experiments
- **tests/** — Unit and integration tests

## Quick start

```bash
pip install -r requirements.txt
python src/main.py
```

## Docker

```bash
docker build -t fake-news-verifier .
docker run fake-news-verifier
```

## Kubernetes

```bash
kubectl apply -f k8s/
```
