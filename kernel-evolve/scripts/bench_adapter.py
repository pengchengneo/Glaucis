#!/usr/bin/env python3
"""Generate Glaucis ref/template/yaml for all pallas-kernel-bench L1 problems.

Reads each problem file from PallasKernelBench/level1/, parses it with AST,
and generates three Glaucis-compatible files:
  - kernels/bench_NNN_ref.py   (reference implementation)
  - kernels/bench_NNN.py       (template with EVOLVE-BLOCK)
  - bench_NNN.yaml             (config)

Usage:
    python kernel-evolve/scripts/bench_adapter.py [--problems-dir PATH] [--output-dir PATH] [--validate]
"""

import argparse
import ast
import inspect
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProblemInfo:
    problem_id: int
    name: str  # e.g. "ReLU", "Square_matrix_multiplication_"
    source: str  # full file content
    imports: list[str] = field(default_factory=list)
    constants_source: str = ""  # module-level assignments
    model_source: str = ""  # Model class source
    is_flax: bool = False
    has_setup: bool = False  # Flax uses setup() instead of @nn.compact
    get_inputs_source: str = ""
    get_init_inputs_source: str = ""
    extra_source: str = ""  # any other module-level code (helper functions, etc.)


def parse_problem(filepath: Path) -> ProblemInfo:
    """Parse a pallas-kernel-bench problem file using AST."""
    source = filepath.read_text()
    filename = filepath.stem  # e.g. "19_ReLU"
    parts = filename.split("_", 1)
    problem_id = int(parts[0])
    name = parts[1] if len(parts) > 1 else f"problem_{problem_id}"

    info = ProblemInfo(problem_id=problem_id, name=name, source=source)
    tree = ast.parse(source)

    imports = []
    constants = []
    model_lines = None
    get_inputs_lines = None
    get_init_inputs_lines = None
    extra_parts = []

    source_lines = source.splitlines(keepends=True)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno
            line_text = "".join(source_lines[start:end])
            imports.append(line_text.rstrip())
            # Check for flax
            if isinstance(node, ast.ImportFrom) and node.module and "flax" in node.module:
                info.is_flax = True  # will refine later based on class bases

        elif isinstance(node, ast.ClassDef) and node.name == "Model":
            start = node.lineno - 1
            end = node.end_lineno
            model_lines = (start, end)
            info.model_source = "".join(source_lines[start:end])
            # Check if it extends nn.Module
            for base in node.bases:
                base_src = ast.get_source_segment(source, base) or ""
                if "Module" in base_src:
                    info.is_flax = True
            # Check for setup() method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "setup":
                    info.has_setup = True

        elif isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = node.end_lineno
            fn_source = "".join(source_lines[start:end])
            if node.name == "get_inputs":
                get_inputs_lines = (start, end)
                info.get_inputs_source = fn_source
            elif node.name == "get_init_inputs":
                get_init_inputs_lines = (start, end)
                info.get_init_inputs_source = fn_source
            else:
                extra_parts.append(fn_source)

        elif isinstance(node, ast.Assign):
            start = node.lineno - 1
            end = node.end_lineno
            line_text = "".join(source_lines[start:end]).rstrip()
            constants.append(line_text)

        elif isinstance(node, ast.AnnAssign):
            # Annotated assignments at module level
            start = node.lineno - 1
            end = node.end_lineno
            line_text = "".join(source_lines[start:end]).rstrip()
            constants.append(line_text)

    info.imports = imports
    info.constants_source = "\n".join(constants)
    info.extra_source = "\n\n".join(extra_parts) if extra_parts else ""

    return info


def _make_import_block(info: ProblemInfo) -> str:
    """Build the import block, ensuring jax and jnp are present."""
    lines = []
    has_jax = False
    has_jnp = False
    for imp in info.imports:
        lines.append(imp)
        if "import jax" in imp and "jax.numpy" not in imp:
            has_jax = True
        if "jax.numpy" in imp:
            has_jnp = True
    if not has_jax:
        lines.insert(0, "import jax")
    if not has_jnp:
        lines.insert(1, "import jax.numpy as jnp")
    return "\n".join(lines)


def _make_compute_body(info: ProblemInfo) -> str:
    """Generate the body of simple_compute/optimized_compute."""
    if info.is_flax:
        return textwrap.dedent("""\
            model = Model(*get_init_inputs())
            inputs = get_inputs()
            params = model.init(jax.random.PRNGKey(42), *inputs)
            return model.apply(params, *inputs)""")
    else:
        return textwrap.dedent("""\
            model = Model(*get_init_inputs())
            inputs = get_inputs()
            return model(*inputs)""")


def generate_ref(info: ProblemInfo, out_dir: Path) -> Path:
    """Generate the reference kernel file."""
    filename = f"bench_{info.problem_id:03d}_ref.py"
    filepath = out_dir / filename

    imports = _make_import_block(info)
    compute_body = _make_compute_body(info)
    # Indent compute body for inside function
    indented_body = textwrap.indent(compute_body, "    ")

    parts = [
        f'"""Auto-generated Glaucis reference for L1 problem {info.problem_id}: {info.name}."""',
        "",
        imports,
        "",
    ]

    if info.constants_source:
        parts.extend(["", info.constants_source, ""])

    parts.extend(["", info.model_source, ""])

    if info.extra_source:
        parts.extend(["", info.extra_source, ""])

    parts.extend([
        "",
        info.get_inputs_source,
        "",
        info.get_init_inputs_source,
        "",
        "",
        f"def simple_compute(_id={info.problem_id}):",
        indented_body,
        "",
        "",
        "def reference_fn(**kwargs):",
        "    return simple_compute(**kwargs)",
        "",
    ])

    content = "\n".join(parts)
    # Clean up excessive blank lines
    while "\n\n\n\n" in content:
        content = content.replace("\n\n\n\n", "\n\n\n")

    filepath.write_text(content)
    return filepath


def generate_template(info: ProblemInfo, out_dir: Path) -> Path:
    """Generate the template kernel file with EVOLVE-BLOCK."""
    filename = f"bench_{info.problem_id:03d}.py"
    filepath = out_dir / filename

    imports = _make_import_block(info)
    # Add pallas imports for template
    pallas_imports = [
        "from jax.experimental import pallas as pl",
        "from jax.experimental.pallas import tpu as pltpu",
    ]
    for pi in pallas_imports:
        if pi not in imports:
            imports += "\n" + pi

    compute_body = _make_compute_body(info)
    indented_body = textwrap.indent(compute_body, "    ")

    parts = [
        f'"""Auto-generated Glaucis template for L1 problem {info.problem_id}: {info.name}."""',
        "",
        imports,
        "",
    ]

    if info.constants_source:
        parts.extend(["", info.constants_source, ""])

    parts.extend(["", info.model_source, ""])

    if info.extra_source:
        parts.extend(["", info.extra_source, ""])

    parts.extend([
        "",
        info.get_inputs_source,
        "",
        info.get_init_inputs_source,
        "",
        "",
        "# EVOLVE-BLOCK-START",
        f"def optimized_compute(_id={info.problem_id}):",
        f'    """Replace this with a Pallas kernel implementation.',
        f"",
        f"    The original Model class and get_inputs/get_init_inputs are available",
        f"    at module scope above for reference. Your optimized version should use",
        f"    jax.experimental.pallas.pallas_call for the core computation.",
        f'    """',
        indented_body,
        "# EVOLVE-BLOCK-END",
        "",
    ])

    content = "\n".join(parts)
    while "\n\n\n\n" in content:
        content = content.replace("\n\n\n\n", "\n\n\n")

    filepath.write_text(content)
    return filepath


def generate_yaml(info: ProblemInfo, yaml_dir: Path,
                  evaluator_branch: str = "bench-l1",
                  evaluator_repo: str = "sii-xinglong/Glaucis") -> Path:
    """Generate the YAML config file."""
    filename = f"bench_{info.problem_id:03d}.yaml"
    filepath = yaml_dir / filename

    # Determine atol based on problem type
    if info.problem_id in (51, 52):
        # Argmax/argmin return integer indices — need exact match
        atol = "0"
        rtol = "0"
    elif info.is_flax:
        atol = "1e-2"
        rtol = "1e-2"
    else:
        atol = "1e-4"
        rtol = "1e-4"

    content = f"""# Auto-generated Glaucis config for L1 problem {info.problem_id}: {info.name}
kernel:
  name: "bench_{info.problem_id:03d}"
  template: "kernels/bench_{info.problem_id:03d}.py"
  reference: "kernels/bench_{info.problem_id:03d}_ref.py"
  evolve_markers:
    start: "# EVOLVE-BLOCK-START"
    end: "# EVOLVE-BLOCK-END"

shapes:
  - {{ _id: {info.problem_id} }}

correctness:
  method: "allclose"
  rtol: {rtol}
  atol: {atol}

evaluator:
  namespace: "default"
  job_template: ".github/ci/kernel-eval-job.yaml"
  repo: "{evaluator_repo}"
  branch: "{evaluator_branch}"
  poll_interval: 15
  timeout: 600

tpu:
  cluster: "tpu7x-cluster"
  zone: "us-central1"

batch:
  variants_per_round: 3
  top_k: 1
  max_active_lineages: 2

session:
  max_iterations: 5
  output_dir: "runs/bench_{info.problem_id:03d}"
"""
    filepath.write_text(content)
    return filepath


def validate_file(filepath: Path, func_names: list[str]) -> tuple[bool, str]:
    """Validate a generated file by exec'ing it and checking for expected functions."""
    try:
        source = filepath.read_text()
        compile(source, str(filepath), "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check that expected function names exist in source
    for fn in func_names:
        if f"def {fn}(" not in source:
            return False, f"Missing function: {fn}"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Generate Glaucis files for pallas-kernel-bench L1 problems")
    parser.add_argument(
        "--problems-dir",
        type=Path,
        default=Path("pallas-kernel-bench/PallasKernelBench/level1"),
        help="Directory containing L1 problem files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kernel-evolve/examples"),
        help="Output directory for generated files (yaml goes here, kernels/ subdir for .py)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate generated files (syntax check)")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated problem IDs to process (default: all)")
    parser.add_argument("--branch", type=str, default="bench-l1", help="Git branch for evaluator config")
    parser.add_argument("--repo", type=str, default="sii-xinglong/Glaucis", help="GitHub repo for evaluator config")
    args = parser.parse_args()

    problems_dir = args.problems_dir
    kernels_dir = args.output_dir / "kernels"
    yaml_dir = args.output_dir

    # Ensure output dirs exist
    kernels_dir.mkdir(parents=True, exist_ok=True)

    # Collect problem files
    problem_files = sorted(problems_dir.glob("*.py"))
    if not problem_files:
        print(f"ERROR: No .py files found in {problems_dir}")
        sys.exit(1)

    # Filter by specific problem IDs if requested
    if args.problems:
        target_ids = set(int(x) for x in args.problems.split(","))
    else:
        target_ids = None

    total = 0
    success = 0
    errors = []

    for pf in problem_files:
        try:
            info = parse_problem(pf)
        except Exception as e:
            errors.append((pf.name, f"Parse error: {e}"))
            continue

        if target_ids and info.problem_id not in target_ids:
            continue

        total += 1

        try:
            ref_path = generate_ref(info, kernels_dir)
            tmpl_path = generate_template(info, kernels_dir)
            yaml_path = generate_yaml(info, yaml_dir, evaluator_branch=args.branch, evaluator_repo=args.repo)
        except Exception as e:
            errors.append((pf.name, f"Generation error: {e}"))
            continue

        if args.validate:
            ok_ref, msg_ref = validate_file(ref_path, ["simple_compute", "reference_fn"])
            ok_tmpl, msg_tmpl = validate_file(tmpl_path, ["optimized_compute"])

            if not ok_ref:
                errors.append((pf.name, f"Ref validation failed: {msg_ref}"))
                continue
            if not ok_tmpl:
                errors.append((pf.name, f"Template validation failed: {msg_tmpl}"))
                continue

        success += 1
        print(f"  [{info.problem_id:3d}] {info.name[:50]:<50} "
              f"{'flax' if info.is_flax else 'plain':>5}  OK")

    print(f"\n{'='*70}")
    print(f"Total: {total}  Success: {success}  Errors: {len(errors)}")

    if errors:
        print(f"\nErrors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        sys.exit(1)

    print(f"\nGenerated files in:")
    print(f"  Kernels: {kernels_dir}/bench_*.py")
    print(f"  Configs: {yaml_dir}/bench_*.yaml")


if __name__ == "__main__":
    main()
