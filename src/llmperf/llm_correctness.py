"""Compatibility command for python -m llmperf.llm_correctness."""

import runpy

if __name__ == "__main__":
    runpy.run_module("llmperf.benchmark.correctness", run_name="__main__")
