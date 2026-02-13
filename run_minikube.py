#!/usr/bin/env python3
"""
Run Minikube + FakeNewsVerifier deployment steps sequentially.

Prerequisites (install once):
  - Minikube:  brew install minikube
  - kubectl:   brew install kubectl   (or use minikube kubectl)
  - Docker:    Docker Desktop for Mac  (https://docs.docker.com/desktop/install/mac-install/)

Usage (from repo root):
  python run_minikube.py              # deployment, build + apply + show logs
  python run_minikube.py --job        # one-shot job instead of deployment
  python run_minikube.py --skip-build # reuse existing image, only apply
  python run_minikube.py --no-logs   # don't stream logs at the end
  python run_minikube.py --cleanup   # delete k8s resources and exit
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Use same shell as user (zsh on Mac) so PATH from .zshrc/.zprofile is used
USER_SHELL = os.environ.get("SHELL", "/bin/zsh")


def check_prereqs() -> None:
    """Exit with a clear message if minikube/kubectl/docker are missing or Docker daemon not running."""
    result = subprocess.run(
        [USER_SHELL, "-l", "-c", "command -v minikube && command -v kubectl && command -v docker"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Prerequisites missing. This script needs minikube, kubectl, and docker on your PATH.\n")
        print("Install on macOS (Homebrew):")
        print("  brew install minikube kubectl")
        print("  # Docker: install Docker Desktop for Mac from https://docs.docker.com/desktop/install/mac-install/\n")
        print("Then open a new terminal (or run 'source ~/.zshrc') and run this script again.")
        sys.exit(1)

    # Check that Docker daemon is running (Minikube needs it)
    check = subprocess.run(
        [USER_SHELL, "-l", "-c", "docker info"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print("Docker is installed but the Docker daemon is not running.\n")
        print("Step-by-step:")
        print("  1. Open the Docker Desktop app (Applications or Spotlight).")
        print("  2. Wait until the menu bar shows 'Docker Desktop is running'.")
        print("  3. Run this script again: python run_minikube.py\n")
        print("You do not need to reinstall Docker or run this on a server.")
        sys.exit(1)


def run(cmd: list[str] | str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a command; raise on failure. Uses login shell so PATH (e.g. Homebrew) is set."""
    cwd = cwd or REPO_ROOT
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    print("  $", cmd_str)
    return subprocess.run(
        [USER_SHELL, "-l", "-c", cmd_str],
        cwd=cwd,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Minikube deployment steps")
    parser.add_argument("--job", action="store_true", help="Use Job (one-shot) instead of Deployment")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build; use existing image")
    parser.add_argument("--no-logs", action="store_true", help="Do not stream logs at the end")
    parser.add_argument("--cleanup", action="store_true", help="Delete k8s resources and exit")
    args = parser.parse_args()

    check_prereqs()

    if args.cleanup:
        if args.job:
            run(["kubectl", "delete", "job", "verifier-job-example", "--ignore-not-found=true"])
        else:
            run(["kubectl", "delete", "-f", "k8s/deployment.yaml", "--ignore-not-found=true"])
        run(["kubectl", "delete", "-f", "k8s/service.yaml", "--ignore-not-found=true"])
        print("Cleanup done.")
        return

    # 1. Start Minikube
    print("Step 1: minikube start")
    run(["minikube", "start"])

    # 2. Build image in Minikube's Docker (so it sees the image)
    if not args.skip_build:
        print("Step 2: docker build (using Minikube docker-env)")
        run("eval $(minikube docker-env) && docker build -t fake-news-verifier:phase2 .")
    else:
        print("Step 2: skip-build (using existing image)")

    # 3. Apply manifests
    print("Step 3: kubectl apply")
    if args.job:
        run(["kubectl", "apply", "-f", "k8s/job.yaml"])
    else:
        run(["kubectl", "apply", "-f", "k8s/deployment.yaml"])
    run(["kubectl", "apply", "-f", "k8s/service.yaml"])

    # 4. Wait for pod and show status
    print("Step 4: wait for pod and status")
    if args.job:
        run(["kubectl", "wait", "--for=condition=complete", "job/verifier-job-example", "--timeout=120s"])
    else:
        run(["kubectl", "rollout", "status", "deployment/verifier-deployment", "--timeout=120s"])
    run(["kubectl", "get", "pods"])
    run(["kubectl", "get", "deployments"])

    # 5. Stream logs
    if not args.no_logs:
        print("Step 5: logs (Ctrl+C to stop)")
        if args.job:
            run(["kubectl", "logs", "job/verifier-job-example"])
        else:
            try:
                run(["kubectl", "logs", "-f", "deployment/verifier-deployment"])
            except KeyboardInterrupt:
                pass
    else:
        print("Step 5: skipped (--no-logs)")
        if args.job:
            run(["kubectl", "logs", "job/verifier-job-example"])
        else:
            run(["kubectl", "logs", "deployment/verifier-deployment"])

    print("Done.")


if __name__ == "__main__":
    main()
