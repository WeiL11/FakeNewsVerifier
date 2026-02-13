"""
FakeNewsVerifier — Entry point / simulator runner.
"""
import sys
from pathlib import Path

# Ensure project root on path when run as script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    # TODO: wire claim extractor -> verifier -> corrector, graph, causal
    print("FakeNewsVerifier runner (stub)")


if __name__ == "__main__":
    main()
