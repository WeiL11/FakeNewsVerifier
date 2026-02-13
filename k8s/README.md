# Kubernetes + Minikube (Phase 2)

Run the FakeNewsVerifier simulator in Minikube on your Mac.

## Prerequisites (run once)

- **Minikube**: [Install guide](https://minikube.sigs.k8s.io/docs/start/)  
  ```bash
  brew install minikube   # or download from the link
  ```
- **kubectl**: Usually comes with Minikube, or `brew install kubectl`
- **Docker**: Minikube uses it as the default driver. [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)

## Step-by-step

### 1. Build the Docker image locally

From the **repo root** (next to `README.md`):

```bash
docker build -t fake-news-verifier:phase2 .
```

### 2. Point Minikube at your local Docker

So Minikube can see the image you just built:

```bash
eval $(minikube docker-env)
```

**Re-run this in every new terminal** where you build images.

### 3. Start Minikube (if not running)

```bash
minikube start
```

### 4. Apply manifests

**Option A — Deployment** (long-running; good for repeated runs):

```bash
kubectl apply -f k8s/deployment.yaml
```

**Option B — Job** (one-shot, then exit):

```bash
kubectl apply -f k8s/job.yaml
```

**Option C — Job with RL tune** (batch tuning on train data, then verify and save report):

```bash
kubectl apply -f k8s/job-rl-tune.yaml
```

Logs: `kubectl logs job/verifier-job-rl-tune`. Cleanup: `kubectl delete job verifier-job-rl-tune`.

Optional service (for when you add an HTTP API later):

```bash
kubectl apply -f k8s/service.yaml
```

### 5. Check status

```bash
kubectl get pods
kubectl get deployments
# For job: kubectl get jobs
```

### 6. View logs (verification output)

**Deployment:**

```bash
kubectl logs -f deployment/verifier-deployment
```

**Job:**

```bash
kubectl logs job/verifier-job-example
```

You should see the same console output as locally: extraction → verification → correction → report.

### 7. Clean up

**Deployment:**

```bash
kubectl delete -f k8s/deployment.yaml
```

**Job:**

```bash
kubectl delete job verifier-job-example
```

**All k8s resources:**

```bash
kubectl delete -f k8s/
```

## Manifests overview

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `deployment.yaml` | Runs the simulator as a Deployment (1 replica).      |
| `job.yaml`        | One-shot run; writes report to `/app/report.json` in the container. |
| `service.yaml`    | ClusterIP service (port 80 → 8000); for future API.  |
| `configmap.yaml`  | Optional config (e.g. `LOG_LEVEL`, `DATA_PATH`).     |

## Overriding the claim (Deployment)

Edit `deployment.yaml` and change the `args` under the container:

```yaml
args:
  - "--input"
  - "Your custom claim here."
```

Then:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout restart deployment/verifier-deployment
kubectl logs -f deployment/verifier-deployment
```
