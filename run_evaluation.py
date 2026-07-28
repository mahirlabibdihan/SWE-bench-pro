"""Backward-compatible entry point for the SWE-bench Pro evaluator."""

from swebenchpro.harness.run_evaluation import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
