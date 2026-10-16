"""Compatibility entry point for installed llmperf-cache-links scripts."""

from llmperf_tools.huggingface.cache_links import main

if __name__ == "__main__":
    raise SystemExit(main())
