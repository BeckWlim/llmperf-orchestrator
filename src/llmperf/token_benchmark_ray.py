"""Compatibility command for python -m llmperf.token_benchmark_ray."""

import runpy

if __name__ == "__main__":
    runpy.run_module("llmperf.benchmark.throughput", run_name="__main__")
