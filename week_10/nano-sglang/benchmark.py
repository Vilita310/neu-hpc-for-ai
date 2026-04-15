"""Part 4 benchmark: scheduler batching vs sequential generation.

Usage:
    python benchmark.py
"""

import time
from dataclasses import dataclass

from nano_sglang.engine import Engine
from nano_sglang.scheduler import Scheduler
from nano_sglang.sampling import SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"


@dataclass
class BenchResult:
    mode: str
    num_requests: int
    elapsed_sec: float
    total_tokens: int

    @property
    def tok_per_sec(self) -> float:
        return self.total_tokens / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


def build_prompts(n: int) -> list[str]:
    return [f"Write 2 short facts about topic {i}." for i in range(n)]


def run_sequential(prompts: list[str], params: SamplingParams) -> BenchResult:
    engine = Engine(MODEL_PATH)
    start = time.time()
    outputs = [engine.generate(p, params) for p in prompts]
    elapsed = time.time() - start
    total_tokens = sum(len(engine.tokenizer.encode(t)) for t in outputs)
    return BenchResult("sequential", len(prompts), elapsed, total_tokens)


def run_scheduler(prompts: list[str], params: SamplingParams) -> BenchResult:
    scheduler = Scheduler(MODEL_PATH, max_batch_size=max(1, len(prompts)))
    for p in prompts:
        scheduler.add_request(p, params)

    start = time.time()
    outputs = scheduler.run_to_completion(params)
    elapsed = time.time() - start
    total_tokens = sum(len(scheduler.tokenizer.encode(t)) for t in outputs)
    return BenchResult("scheduler", len(prompts), elapsed, total_tokens)


def main():
    params = SamplingParams(temperature=0, max_tokens=30)
    concurrencies = [1, 2, 4, 8]

    print("num_requests | mode       | elapsed(s) | total_tokens | tok/s")
    print("-" * 64)
    for n in concurrencies:
        prompts = build_prompts(n)
        seq_r = run_sequential(prompts, params)
        sch_r = run_scheduler(prompts, params)
        for r in (seq_r, sch_r):
            print(
                f"{r.num_requests:12d} | {r.mode:10s} | {r.elapsed_sec:9.2f} |"
                f" {r.total_tokens:12d} | {r.tok_per_sec:6.1f}"
            )


if __name__ == "__main__":
    main()
