#!/usr/bin/env python3
"""Collect and aggregate results from Glaucis pallas-kernel-bench runs.

Scans worktree run directories, extracts eval results and lineage data,
and generates summary statistics and a report.

Usage:
    python kernel-evolve/scripts/bench_collect.py --output bench_results/
    python kernel-evolve/scripts/bench_collect.py --progress bench_progress.json --output bench_results/
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from math import exp, log
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Problem category mapping
CATEGORIES = {}
for pid in range(1, 19):
    CATEGORIES[pid] = "matmul"
for pid in range(19, 33):
    CATEGORIES[pid] = "activation"
for pid in range(33, 41):
    CATEGORIES[pid] = "normalization"
for pid in range(41, 47):
    CATEGORIES[pid] = "pooling"
for pid in [47, 48, 49, 51, 52, 53]:
    CATEGORIES[pid] = "reduction"
for pid in [50] + list(range(54, 88)):
    CATEGORIES[pid] = "convolution"
CATEGORIES[88] = "composite"
for pid in range(89, 94):
    CATEGORIES[pid] = "scan"
for pid in [94, 95, 96, 98, 99, 100]:
    CATEGORIES[pid] = "loss"
CATEGORIES[97] = "attention"


def find_run_dir(problem_id: int, worktree_name: str = None) -> Path | None:
    """Find the run directory for a problem, checking worktree and main repo."""
    candidates = []

    # Check worktree location
    if worktree_name:
        wt_path = REPO_ROOT / ".claude" / "worktrees" / worktree_name
        candidates.append(wt_path / "kernel-evolve" / "runs")

    # Check main repo
    candidates.append(REPO_ROOT / "kernel-evolve" / "runs")

    for base in candidates:
        # Look for run directories matching bench_NNN pattern
        pattern = f"bench_{problem_id:03d}*"
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[-1]  # latest run

    return None


def extract_results(run_dir: Path) -> dict:
    """Extract results from a run directory."""
    result = {
        "iterations": 0,
        "variants": [],
        "best_speedup": 0.0,
        "best_kernel_path": None,
        "correct_count": 0,
        "compile_error_count": 0,
        "incorrect_count": 0,
        "total_count": 0,
    }

    # Read lineages.json if it exists
    lineages_file = run_dir / "lineages.json"
    if lineages_file.exists():
        lineages = json.loads(lineages_file.read_text())
        result["iterations"] = lineages.get("round", 0)

        for lin in lineages.get("lineages", []):
            if lin.get("best_speedup", 0) > result["best_speedup"]:
                result["best_speedup"] = lin["best_speedup"]
                result["best_kernel_path"] = lin.get("best_kernel")

    # Scan iteration directories for all variant results
    for iter_dir in sorted(run_dir.glob("iteration_*")):
        variants_dir = iter_dir / "variants"
        if not variants_dir.exists():
            continue

        for variant_dir in sorted(variants_dir.iterdir()):
            eval_result_file = variant_dir / "eval_result.json"
            if not eval_result_file.exists():
                continue

            result["total_count"] += 1
            try:
                eval_data = json.loads(eval_result_file.read_text())
                status = eval_data.get("status", "UNKNOWN")
                variant_info = {
                    "iteration": iter_dir.name,
                    "variant": variant_dir.name,
                    "status": status,
                    "speedup": eval_data.get("speedup", 0),
                    "latency_ms": eval_data.get("latency_ms", 0),
                }
                result["variants"].append(variant_info)

                if status == "SUCCESS":
                    result["correct_count"] += 1
                    speedup = eval_data.get("speedup", 0)
                    if speedup > result["best_speedup"]:
                        result["best_speedup"] = speedup
                        kernel_file = variant_dir / "kernel.py"
                        if kernel_file.exists():
                            result["best_kernel_path"] = str(kernel_file)
                elif status == "COMPILE_ERROR":
                    result["compile_error_count"] += 1
                elif status == "INCORRECT":
                    result["incorrect_count"] += 1
            except (json.JSONDecodeError, KeyError):
                pass

    return result


def compute_metrics(all_results: dict[int, dict]) -> dict:
    """Compute aggregate metrics from all problem results."""
    total = len(all_results)
    if total == 0:
        return {}

    correct = [pid for pid, r in all_results.items() if r["correct_count"] > 0]
    faster = [pid for pid, r in all_results.items() if r["best_speedup"] > 1.0]
    much_faster = [pid for pid, r in all_results.items() if r["best_speedup"] > 2.0]

    speedups = [r["best_speedup"] for r in all_results.values() if r["correct_count"] > 0 and r["best_speedup"] > 0]
    geo_mean = exp(sum(log(s) for s in speedups) / len(speedups)) if speedups else 0

    metrics = {
        "total_problems": total,
        "fast_0": len(correct) / total,
        "fast_0_count": len(correct),
        "fast_1": len(faster) / total,
        "fast_1_count": len(faster),
        "fast_2": len(much_faster) / total,
        "fast_2_count": len(much_faster),
        "geo_mean_speedup": geo_mean,
        "total_variants": sum(r["total_count"] for r in all_results.values()),
        "total_correct_variants": sum(r["correct_count"] for r in all_results.values()),
        "total_compile_errors": sum(r["compile_error_count"] for r in all_results.values()),
        "total_incorrect": sum(r["incorrect_count"] for r in all_results.values()),
    }

    # Per-category breakdown
    cat_results = defaultdict(list)
    for pid, r in all_results.items():
        cat = CATEGORIES.get(pid, "other")
        cat_results[cat].append((pid, r))

    metrics["per_category"] = {}
    for cat, items in sorted(cat_results.items()):
        cat_total = len(items)
        cat_correct = sum(1 for _, r in items if r["correct_count"] > 0)
        cat_faster = sum(1 for _, r in items if r["best_speedup"] > 1.0)
        cat_speedups = [r["best_speedup"] for _, r in items if r["correct_count"] > 0 and r["best_speedup"] > 0]
        cat_geo = exp(sum(log(s) for s in cat_speedups) / len(cat_speedups)) if cat_speedups else 0

        metrics["per_category"][cat] = {
            "total": cat_total,
            "fast_0": cat_correct / cat_total if cat_total else 0,
            "fast_1": cat_faster / cat_total if cat_total else 0,
            "geo_mean_speedup": cat_geo,
        }

    return metrics


def generate_report(all_results: dict[int, dict], metrics: dict, output_dir: Path) -> None:
    """Generate a Markdown summary report."""
    lines = [
        "# Glaucis pallas-kernel-bench L1 Results",
        "",
        f"Generated: {__import__('datetime').datetime.now().isoformat()[:19]}",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total problems | {metrics['total_problems']} |",
        f"| fast_0 (any correct) | {metrics['fast_0']:.1%} ({metrics['fast_0_count']}) |",
        f"| fast_1 (speedup > 1x) | {metrics['fast_1']:.1%} ({metrics['fast_1_count']}) |",
        f"| fast_2 (speedup > 2x) | {metrics['fast_2']:.1%} ({metrics['fast_2_count']}) |",
        f"| Geometric mean speedup | {metrics['geo_mean_speedup']:.3f}x |",
        f"| Total variants evaluated | {metrics['total_variants']} |",
        f"| Total correct variants | {metrics['total_correct_variants']} |",
        f"| Total compile errors | {metrics['total_compile_errors']} |",
        f"| Total incorrect | {metrics['total_incorrect']} |",
        "",
        "## Per-Category Breakdown",
        "",
        "| Category | Count | fast_0 | fast_1 | Geo Mean |",
        "|----------|-------|--------|--------|----------|",
    ]

    for cat, cm in sorted(metrics.get("per_category", {}).items()):
        lines.append(
            f"| {cat} | {cm['total']} | {cm['fast_0']:.0%} | {cm['fast_1']:.0%} | {cm['geo_mean_speedup']:.3f}x |"
        )

    # Top 10 best speedups
    ranked = sorted(
        ((pid, r) for pid, r in all_results.items() if r["best_speedup"] > 0),
        key=lambda x: x[1]["best_speedup"],
        reverse=True,
    )

    lines.extend([
        "",
        "## Top 10 Best Speedups",
        "",
        "| Rank | Problem | Category | Speedup | Iterations | Correct/Total |",
        "|------|---------|----------|---------|------------|---------------|",
    ])

    for i, (pid, r) in enumerate(ranked[:10], 1):
        cat = CATEGORIES.get(pid, "other")
        lines.append(
            f"| {i} | {pid:03d} | {cat} | {r['best_speedup']:.3f}x | {r['iterations']} | {r['correct_count']}/{r['total_count']} |"
        )

    # Failed problems (no correct variants)
    failed = [(pid, r) for pid, r in sorted(all_results.items()) if r["correct_count"] == 0]
    if failed:
        lines.extend([
            "",
            f"## Problems with No Correct Variants ({len(failed)})",
            "",
            "| Problem | Category | Compile Errors | Incorrect | Total |",
            "|---------|----------|----------------|-----------|-------|",
        ])
        for pid, r in failed:
            cat = CATEGORIES.get(pid, "other")
            lines.append(
                f"| {pid:03d} | {cat} | {r['compile_error_count']} | {r['incorrect_count']} | {r['total_count']} |"
            )

    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze pallas-kernel-bench results")
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "bench_results",
        help="Output directory for results and report",
    )
    parser.add_argument(
        "--progress", type=Path, default=REPO_ROOT / "bench_progress.json",
        help="Progress file from bench_orchestrate.py",
    )
    parser.add_argument(
        "--copy-kernels", action="store_true",
        help="Copy best kernel files to output directory",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress to get worktree names
    progress = {}
    if args.progress.exists():
        progress = json.loads(args.progress.read_text()).get("problems", {})
        print(f"Loaded progress for {len(progress)} problems")

    # Collect results from all problems
    all_results = {}

    # If we have progress data, use worktree names from it
    if progress:
        for pid_str, p in sorted(progress.items(), key=lambda x: int(x[0])):
            pid = int(pid_str)
            worktree = p.get("worktree")
            run_dir = find_run_dir(pid, worktree)
            if run_dir and run_dir.exists():
                results = extract_results(run_dir)
                all_results[pid] = results
                print(f"  [{pid:3d}] {results['correct_count']}/{results['total_count']} correct, "
                      f"best={results['best_speedup']:.3f}x, iters={results['iterations']}")
            else:
                print(f"  [{pid:3d}] No run directory found")
    else:
        # Scan runs directory directly
        runs_dir = REPO_ROOT / "kernel-evolve" / "runs"
        for run_dir in sorted(runs_dir.glob("bench_*")):
            try:
                pid = int(run_dir.name.split("_")[1][:3])
            except (IndexError, ValueError):
                continue
            results = extract_results(run_dir)
            all_results[pid] = results
            print(f"  [{pid:3d}] {results['correct_count']}/{results['total_count']} correct, "
                  f"best={results['best_speedup']:.3f}x")

    if not all_results:
        print("No results found.")
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(all_results)

    # Save raw results
    results_file = output_dir / "results.json"
    # Strip non-serializable data
    clean_results = {}
    for pid, r in all_results.items():
        clean_results[pid] = {k: v for k, v in r.items() if k != "best_kernel_path"}
        clean_results[pid]["best_kernel_path"] = r.get("best_kernel_path")

    results_file.write_text(json.dumps({
        "metrics": metrics,
        "problems": clean_results,
    }, indent=2, default=str))
    print(f"\nRaw results saved to: {results_file}")

    # Generate report
    generate_report(all_results, metrics, output_dir)

    # Copy best kernels
    if args.copy_kernels:
        kernels_dir = output_dir / "kernels"
        kernels_dir.mkdir(exist_ok=True)
        copied = 0
        for pid, r in all_results.items():
            kernel_path = r.get("best_kernel_path")
            if kernel_path and Path(kernel_path).exists():
                dest = kernels_dir / f"bench_{pid:03d}_best.py"
                shutil.copy2(kernel_path, dest)
                copied += 1
        print(f"Copied {copied} best kernels to: {kernels_dir}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  fast_0 (any correct):     {metrics['fast_0']:.1%} ({metrics['fast_0_count']}/{metrics['total_problems']})")
    print(f"  fast_1 (speedup > 1x):    {metrics['fast_1']:.1%} ({metrics['fast_1_count']}/{metrics['total_problems']})")
    print(f"  fast_2 (speedup > 2x):    {metrics['fast_2']:.1%} ({metrics['fast_2_count']}/{metrics['total_problems']})")
    print(f"  Geometric mean speedup:   {metrics['geo_mean_speedup']:.3f}x")


if __name__ == "__main__":
    main()
