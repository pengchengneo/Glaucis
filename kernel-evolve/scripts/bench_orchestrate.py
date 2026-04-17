#!/usr/bin/env python3
"""Orchestrate Glaucis optimization for pallas-kernel-bench L1 problems.

Spawns one Claude Code session per problem, tracks progress, and supports resume.
Each session's stdout/stderr streams in real-time to bench_logs/bench_NNN.log.

Usage:
    python kernel-evolve/scripts/bench_orchestrate.py --parallel 3 --problems 1-100
    python kernel-evolve/scripts/bench_orchestrate.py --problems 1,19,40 --resume
    # Monitor live:  tail -f bench_logs/bench_019.log
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROGRESS_FILE = REPO_ROOT / "bench_progress.json"
LOG_DIR = REPO_ROOT / "bench_logs"

SYSTEM_PROMPT_ADDENDUM = """\
You are running an automated benchmark evaluation. Key rules:
- ALWAYS choose "Autonomous" mode when the start skill asks.
- Do NOT run `gh issue create` or `gh issue comment`.
- Do NOT ask user questions. Make all decisions autonomously.
- If a round produces only COMPILE_ERROR results, try a fundamentally different approach next round.
- Stop after the configured max_iterations or when all lineages stagnate for 2 rounds.
- Save all artifacts. Every iteration's kernel variants must be persisted.
"""


def build_prompt(problem_id: int, max_iterations: int) -> str:
    return f"""\
Run a Glaucis kernel optimization session for pallas-kernel-bench L1 problem {problem_id}.

Config file: kernel-evolve/examples/bench_{problem_id:03d}.yaml

IMPORTANT INSTRUCTIONS:
1. First, read AGENT.md for accumulated optimization knowledge.
2. Read the config file and the template/reference kernel files to understand the problem.
3. The template's EVOLVE-BLOCK currently passes through to the original JAX Model class.
   Your goal: evolve it into an efficient Pallas TPU kernel using pallas_call.
4. Run /pallas-evolve:start bench_{problem_id:03d}.yaml
   - Choose AUTONOMOUS mode (do not ask user)
   - Maximum {max_iterations} iterations
5. Do NOT create GitHub Issues or post GitHub comments.
6. Save all artifacts. Every iteration's kernel variants must be persisted to disk.
7. When done, report your final results as JSON on the last line:
   BENCH_RESULT:{{"problem_id": {problem_id}, "iterations": N, "best_speedup": X.XX, "correct_variants": N, "total_variants": N}}
"""


def parse_problem_range(spec: str) -> list[int]:
    result = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"meta": {}, "problems": {}}


def save_progress(progress: dict) -> None:
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.rename(PROGRESS_FILE)


def run_one_problem(
    problem_id: int,
    max_iterations: int = 5,
    max_budget: float = 5.0,
    claude_bin: str = "claude",
    model: str = "sonnet",
    dry_run: bool = False,
) -> dict:
    """Run a single problem through Glaucis optimization.

    Streams stdout/stderr to bench_logs/bench_NNN.log in real-time.
    """
    prompt = build_prompt(problem_id, max_iterations)
    start_time = datetime.now(timezone.utc).isoformat()
    tag = f"bench_{problem_id:03d}"

    status = {
        "problem_id": problem_id,
        "state": "running",
        "iterations": 0,
        "best_speedup": 0.0,
        "correct_variants": 0,
        "total_variants": 0,
        "start_time": start_time,
        "end_time": None,
        "error": None,
    }

    if dry_run:
        print(f"  [DRY-RUN] Problem {problem_id}: would run claude -p ...")
        status["state"] = "skipped"
        return status

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{tag}.log"

    cmd = [
        claude_bin, "-p", prompt,
        "--output-format", "stream-json",
        "--max-budget-usd", str(max_budget),
        "--append-system-prompt", SYSTEM_PROMPT_ADDENDUM,
        "--permission-mode", "bypassPermissions",
        "--model", model,
    ]

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"=== Problem {problem_id} started at {start_time} ===\n")
            log_file.write(f"=== cmd: {' '.join(cmd[:6])} ... ===\n\n")
            log_file.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_ROOT),
            )

            returncode = proc.wait(timeout=7200)

        # Parse log for BENCH_RESULT
        log_content = log_path.read_text()
        for line in log_content.splitlines():
            if line.startswith("BENCH_RESULT:"):
                try:
                    bench_data = json.loads(line[len("BENCH_RESULT:"):])
                    status["iterations"] = bench_data.get("iterations", 0)
                    status["best_speedup"] = bench_data.get("best_speedup", 0.0)
                    status["correct_variants"] = bench_data.get("correct_variants", 0)
                    status["total_variants"] = bench_data.get("total_variants", 0)
                except json.JSONDecodeError:
                    pass

        if returncode == 0:
            status["state"] = "completed"
        else:
            status["state"] = "failed"
            status["error"] = f"Exit code {returncode}"

    except subprocess.TimeoutExpired:
        proc.kill()
        status["state"] = "failed"
        status["error"] = "Timeout (2 hours)"
    except Exception as e:
        status["state"] = "failed"
        status["error"] = str(e)[:500]

    status["end_time"] = datetime.now(timezone.utc).isoformat()

    # Append summary to log
    try:
        with open(log_path, "a") as f:
            f.write(f"\n=== Problem {problem_id} finished: {status['state']} ===\n")
            f.write(f"=== speedup={status['best_speedup']}, iters={status['iterations']} ===\n")
    except Exception:
        pass

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate Glaucis optimization for pallas-kernel-bench L1 problems"
    )
    parser.add_argument(
        "--problems", type=str, default="1-100",
        help="Problem IDs: '1-100', '1,19,40', or '1-10,50-60' (default: 1-100)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Max concurrent Claude sessions (default: 1)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max optimization iterations per problem (default: 5)",
    )
    parser.add_argument(
        "--max-budget", type=float, default=5.0,
        help="Max USD budget per problem session (default: 5.0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run, skipping completed problems",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running",
    )
    parser.add_argument(
        "--claude-bin", type=str, default="claude",
        help="Path to claude CLI binary (default: claude)",
    )
    parser.add_argument(
        "--model", type=str, default="sonnet",
        help="Claude model to use (default: sonnet)",
    )
    args = parser.parse_args()

    target_ids = parse_problem_range(args.problems)
    progress = load_progress() if args.resume else {"meta": {}, "problems": {}}

    progress["meta"] = {
        "started": datetime.now(timezone.utc).isoformat(),
        "parallel": args.parallel,
        "max_iterations": args.max_iterations,
        "max_budget": args.max_budget,
        "model": args.model,
        "target_problems": len(target_ids),
    }

    if args.resume:
        completed = {
            int(pid)
            for pid, s in progress["problems"].items()
            if s.get("state") == "completed"
        }
        pending = [pid for pid in target_ids if pid not in completed]
        print(f"Resuming: {len(completed)} completed, {len(pending)} pending")
    else:
        pending = target_ids

    if not pending:
        print("No problems to process.")
        return

    print(f"Processing {len(pending)} problems with parallelism={args.parallel}")
    print(f"Max iterations: {args.max_iterations}, Max budget: ${args.max_budget}/problem, Model: {args.model}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Logs: {LOG_DIR}/bench_NNN.log  (tail -f to monitor)")
    print(f"{'='*70}")

    completed_count = len(target_ids) - len(pending)
    total_count = len(target_ids)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as pool:
        future_to_pid = {}
        for pid in pending:
            future = pool.submit(
                run_one_problem,
                pid,
                max_iterations=args.max_iterations,
                max_budget=args.max_budget,
                claude_bin=args.claude_bin,
                model=args.model,
                dry_run=args.dry_run,
            )
            future_to_pid[future] = pid

        for future in concurrent.futures.as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "problem_id": pid,
                    "state": "failed",
                    "error": f"Executor error: {e}",
                    "end_time": datetime.now(timezone.utc).isoformat(),
                }

            progress["problems"][str(pid)] = result
            save_progress(progress)

            completed_count += 1
            state_icon = {
                "completed": "OK",
                "failed": "FAIL",
                "skipped": "SKIP",
            }.get(result.get("state", "?"), "?")

            speedup_str = f"{result.get('best_speedup', 0):.2f}x" if result.get("best_speedup") else "N/A"
            print(
                f"  [{completed_count:3d}/{total_count}] Problem {pid:3d}: "
                f"{state_icon:4s}  speedup={speedup_str}  "
                f"iters={result.get('iterations', 0)}"
            )

    # Print summary
    print(f"\n{'='*70}")
    states = {}
    for s in progress["problems"].values():
        st = s.get("state", "unknown")
        states[st] = states.get(st, 0) + 1
    print(f"Summary: {states}")

    speedups = [
        s["best_speedup"]
        for s in progress["problems"].values()
        if s.get("state") == "completed" and s.get("best_speedup", 0) > 0
    ]
    if speedups:
        from math import exp, log
        geo_mean = exp(sum(log(s) for s in speedups) / len(speedups))
        print(f"Geometric mean speedup (correct only): {geo_mean:.3f}x")

    print(f"\nFull results saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
