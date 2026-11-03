from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

from contract.agent.contract import ContractAgent
from contract.eval.private import evaluate_private_cases
from contract.llm.client import DEFAULT_MODELS, LLMClient, load_llm_config
from contract.utils.dataset import (
    DatasetTask,
    build_dataset_task,
    get_tests,
    iter_dataset_rows,
)
from contract.utils.logger import logger
from contract.utils.reporting import show_pipeline_result
from contract.utils.prompt_renderer import normalize_prompt_variant
from contract.utils.results import (
    completed_task_keys,
    persist_exception,
    persist_result,
    pipeline_passed,
    sanitize_path_part,
    utc_now,
    write_json,
    write_summary,
)

MODELS = {
    "gpt54mini": "openai/gpt-5.4-mini",
    "haiku45": "anthropic/claude-haiku-4.5",
    "llama4_maverick": "meta-llama/llama-4-maverick",
}
BENCHMARKS = {
    "apps": "apps_specine_update.jsonl",
    "code_contests": "code_contests_update.jsonl",
    "xcodeeval": "xCodeEval_update.jsonl",
}
FOOTBALL_INPUT = Path("motivating_example/xCodeEval_update.json")
EXAMPLE_OUTPUTS = {
    "specification.json": "nlc.json",
    "executable_contract.py": "ec.py",
    "generated.py": "code.py",
    "contracted.py": "ec_code.py",
    "interface.py": "interface.py",
    "contract_probes.json": "contract_probes.json",
    "contract_lowering_coverage.json": "contract_lowering_coverage.json",
}


def _export_example_outputs(run_dir: Path, record: dict[str, Any]) -> None:
    for original, name in EXAMPLE_OUTPUTS.items():
        source = (record.get("artifacts") or {}).get(original)
        if source:
            shutil.copy2(source, run_dir / name)


def _choose(label: str, choices: list[str]) -> str:
    print(f"\n{label}")
    for number, choice in enumerate(choices, 1):
        print(f"  {number}. {choice}")
    while True:
        try:
            value = input("Select a number: ").strip()
        except EOFError:
            raise SystemExit("Selection cancelled.") from None
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value) - 1]
        if value in choices:
            return value
        print("Choose one of the listed options.")


def _flatten(values: list[list[str]]) -> list[str]:
    return [item for group in values for item in group]


def _parse_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("limit must be non-negative or 'all'")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_descriptors(paths: list[Path]) -> list[dict[str, Any]]:
    descriptors = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        descriptors.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    return descriptors


def _load_tasks(
    paths: list[Path], *, start: int, limit: int | None, witness_split: str
) -> list[DatasetTask]:
    tasks: list[DatasetTask] = []
    task_keys: set[str] = set()
    display_ids: set[str] = set()
    for path in paths:
        rows = iter_dataset_rows(path)
        stop = None if limit is None else start + limit
        selected = islice(rows, start, stop)
        for row_index, row in enumerate(selected, start=start):
            task = build_dataset_task(
                row,
                source_path=path,
                row_index=row_index,
                test_split=witness_split,
            )
            if task.task_key in task_keys:
                raise ValueError(f"Duplicate task key {task.task_key!r}")
            if task.display_id in display_ids:
                raise ValueError(
                    f"Duplicate display ID {task.display_id!r}; artifact directories "
                    "must be unique within a run."
                )
            task_keys.add(task.task_key)
            display_ids.add(task.display_id)
            tasks.append(task)
    return tasks


def _prepare_run(
    *,
    args: argparse.Namespace,
    datasets: list[dict[str, Any]],
    provider: str,
    model: str,
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run-name.")
    run_name = (
        args.run_name
        or f"batch__{sanitize_path_part(provider)}__{sanitize_path_part(model)}__{timestamp}"
    )
    run_dir = Path(args.output_root) / sanitize_path_part(run_name)
    config_path = run_dir / "run_config.json"
    requested = {
        "schema_version": "2026-06-contract-batch-v2",
        "run_name": sanitize_path_part(run_name),
        "created_at": utc_now(),
        "provider": provider,
        "model": model,
        "example": getattr(args, "example", None),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "exec_timeout": args.exec_timeout,
        "private_timeout": (
            args.private_timeout
            if args.private_timeout is not None
            else args.exec_timeout
        ),
        "contract_private_timeout": 2
        * float(
            args.private_timeout
            if args.private_timeout is not None
            else args.exec_timeout
        ),
        "llm_retries": args.llm_retries,
        "max_attempts": args.max_attempts,
        "code_refinement_attempts": args.code_refinement_attempts,
        "witness_split": args.witness_split,
        "eval_split": args.eval_split,
        "eval_private": not args.skip_private_eval,
        "output_structure": args.output_structure,
        "prompt_variant": normalize_prompt_variant(args.prompt_variant),
        "sleep_between_tasks": args.sleep_between_tasks,
        "datasets": datasets,
        "output_root": str(Path(args.output_root).resolve()),
    }
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {config_path}")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        return run_dir, existing
    if config_path.exists() or (run_dir / "results.jsonl").exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}. Use --resume --run-name {sanitize_path_part(run_name)}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    write_json(config_path, requested)
    return run_dir, requested


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contract",
        description="Generate the Football motivating example by default, or select benchmarks and custom JSON/JSONL datasets.",
        epilog="For offline mutation testing: contract mutate --help",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--dataset",
        "--dataset-path",
        dest="datasets",
        action="append",
        nargs="+",
    )
    inputs.add_argument("--benchmark", nargs="+", choices=[*BENCHMARKS, "all"])
    inputs.add_argument(
        "--example",
        choices=["football"],
        help="Generate the bundled Football motivating example (the default).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the offline reviewer readiness tests without model calls.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Select model, example or benchmark, and task limit interactively.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the configuration without model calls or output files.",
    )
    parser.add_argument(
        "--provider",
        "--llm-provider",
        dest="provider",
        choices=sorted(DEFAULT_MODELS),
        default="openrouter",
    )
    parser.add_argument(
        "--model",
        "--code-model",
        dest="model",
        help="Model alias (gpt54mini, haiku45, llama4_maverick) or full provider model ID.",
    )
    parser.add_argument("--output-root", "--output-dir", dest="output_root")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, rerun exceptions and pipeline-failed tasks.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=_parse_limit, default=None)
    parser.add_argument(
        "--witness-split",
        choices=["auto", "public", "private", "all"],
        default="public",
        help="Cases visible to the pipeline as execution witnesses. Keep public for clean evaluation.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["private", "all"],
        default="private",
        help="Cases used only after generation for outside evaluation.",
    )
    parser.add_argument("--skip-private-eval", action="store_true")
    parser.add_argument(
        "--private-timeout",
        type=float,
        default=None,
        help="Final private eval execution timeout. Defaults to --exec-timeout.",
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=5.0,
        help="Execution timeout default for generated-code execution and private eval.",
    )
    parser.add_argument(
        "--output-structure",
        "--output_structure",
        dest="output_structure",
        choices=["raw", "structured"],
        default="structured",
        help="LLM output protocol. Use raw for local/smaller models; structured for models with reliable structured output.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Static-validation attempts for each initial or refined code candidate (default: 3).",
    )
    parser.add_argument(
        "--code-refinement-attempts",
        type=int,
        default=3,
        help="Public-test-driven code refinement rounds (default: 3).",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=[
            "v1",
            "stripped",
            "v2",
            "fewshot",
            "examples",
            "v3",
            "cot",
            "v4",
            "contract",
            "current",
            "selfcheck",
        ],
        default="contract",
        help=(
            "Prompt ablation variant: stripped/rules only; fewshot adds examples; "
            "cot adds private reasoning; contract/current adds self-check and matches the full ContrAct prompt."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--probe-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--llm-retries", type=int, default=6)
    parser.add_argument("--print-artifacts", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--sleep-between-tasks",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each processed task.",
    )
    parser.add_argument("--fail-on-task-error", action="store_true")
    supplied = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(supplied)
    if args.check:
        if len(supplied) != 1:
            parser.error("Run --check by itself; it does not start a generation run.")
        return args
    if args.interactive:
        if not sys.stdin.isatty():
            parser.error(
                "Interactive selection requires a terminal. Supply --model and --benchmark, or --dataset."
            )
        try:
            if args.model is None:
                args.model = _choose("Model", list(MODELS))
            if not args.datasets and not args.benchmark and not args.example:
                selected = _choose(
                    "Example or benchmark", ["football", *BENCHMARKS, "all"]
                )
                if selected == "football":
                    args.example = selected
                else:
                    args.benchmark = [selected]
            if not any(
                value == "--limit" or value.startswith("--limit=") for value in supplied
            ):
                while True:
                    try:
                        args.limit = _parse_limit(
                            input("Tasks per benchmark [1, or all]: ").strip() or "1"
                        )
                        break
                    except (ValueError, argparse.ArgumentTypeError):
                        print("Enter a non-negative number or all.")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Selection cancelled.") from None
    if not args.datasets and not args.benchmark:
        args.example = "football"
    if args.example:
        if args.start != 0:
            parser.error("The Football example contains one task; --start must be 0.")
        args.model = args.model or "haiku45"
        if args.limit is None:
            args.limit = 1
        options = {
            value.split("=", 1)[0] for value in supplied if value.startswith("--")
        }
        for name, flags, value in [
            ("max_tokens", {"--max-tokens"}, 2048),
            ("probe_max_tokens", {"--probe-max-tokens"}, 2048),
            ("timeout", {"--timeout"}, 60),
            ("output_structure", {"--output-structure", "--output_structure"}, "raw"),
        ]:
            if not options.intersection(flags):
                setattr(args, name, value)
    if not args.model:
        parser.error(
            "Choose --model gpt54mini, haiku45, llama4_maverick, or a full model ID."
        )
    if args.resume and not args.run_name:
        parser.error("--resume requires --run-name.")
    if args.start < 0 or args.max_attempts < 1 or args.code_refinement_attempts < 0:
        parser.error(
            "Require --start >= 0, --max-attempts >= 1, and --code-refinement-attempts >= 0."
        )
    if (
        args.max_tokens < 1
        or args.probe_max_tokens < 1
        or args.timeout <= 0
        or args.exec_timeout <= 0
    ):
        parser.error("Token budgets and timeouts must be positive.")
    model_label = args.model
    if model_label in MODELS:
        if args.provider != "openrouter":
            parser.error(
                "Named model aliases use OpenRouter. For another provider, supply its full model ID."
            )
        args.model = MODELS[model_label]
    if args.example:
        args.datasets = [[str(FOOTBALL_INPUT)]]
        if args.output_root is None:
            args.output_root = str(
                Path("motivating_example/runs/generated")
                / sanitize_path_part(model_label)
            )
    elif args.benchmark:
        benchmarks = (
            list(BENCHMARKS)
            if "all" in args.benchmark
            else list(dict.fromkeys(args.benchmark))
        )
        args.datasets = [[str(args.data_dir / BENCHMARKS[name]) for name in benchmarks]]
        benchmark_label = "_".join(benchmarks)
    else:
        benchmark_label = "custom"
    if args.output_root is None:
        args.output_root = str(
            Path("results/batch") / sanitize_path_part(model_label) / benchmark_label
        )
    return args


def main(argv: list[str] | None = None) -> None:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied and supplied[0] == "mutate":
        from contract.eval.mutation import main as mutation_main

        mutation_main(supplied[1:])
        return
    args = parse_args(supplied)
    if args.check:
        from contract.readiness import run_checks

        raise SystemExit(run_checks())
    effective_private_timeout = (
        args.private_timeout if args.private_timeout is not None else args.exec_timeout
    )
    contract_private_timeout = effective_private_timeout * 2
    args.prompt_variant = normalize_prompt_variant(args.prompt_variant)

    dataset_paths = [Path(value) for value in _flatten(args.datasets)]
    descriptors = _dataset_descriptors(dataset_paths)
    llm_config = load_llm_config(
        provider=args.provider, model=args.model, max_tokens=args.max_tokens
    )
    if args.temperature is not None:
        llm_config.temperature = args.temperature
    if args.timeout is not None:
        llm_config.timeout = args.timeout
    if args.llm_retries is not None:
        llm_config.max_retries = max(3, args.llm_retries)
    os.environ["LLM_RETRY_BACKOFF_MAX_ATTEMPTS"] = str(llm_config.max_retries + 1)

    args.temperature = llm_config.temperature
    args.max_tokens = llm_config.max_tokens
    args.timeout = llm_config.timeout
    args.llm_retries = llm_config.max_retries

    tasks = _load_tasks(
        dataset_paths,
        start=args.start,
        limit=args.limit,
        witness_split=args.witness_split,
    )
    if args.example and (
        len(tasks) != 1
        or tasks[0].problem_id != 782
        or tasks[0].source_uid != "ed9a763362abc6ed40356731f1036b38"
    ):
        raise ValueError(
            "The Football input must contain exactly the original xCodeEval 782 task."
        )
    selected_by_dataset: dict[str, int] = {}
    for task in tasks:
        selected_by_dataset[task.dataset] = selected_by_dataset.get(task.dataset, 0) + 1
    if args.dry_run:
        print(
            json.dumps(
                {
                    "provider": llm_config.provider,
                    "example": args.example,
                    "model": llm_config.model,
                    "datasets": descriptors,
                    "tasks_per_benchmark": selected_by_dataset,
                    "output_root": args.output_root,
                    "run_name": args.run_name,
                    "resume": args.resume,
                    "max_tokens": args.max_tokens,
                    "probe_max_tokens": args.probe_max_tokens,
                    "temperature": args.temperature,
                    "timeout": args.timeout,
                    "witness_split": args.witness_split,
                    "eval_split": args.eval_split,
                    "output_structure": args.output_structure,
                },
                indent=2,
            )
        )
        return
    os.environ["LLM_PROBE_MAX_OUTPUT_TOKENS"] = str(args.probe_max_tokens)
    run_dir, run_config = _prepare_run(
        args=args,
        datasets=descriptors,
        provider=llm_config.provider,
        model=llm_config.model or DEFAULT_MODELS[llm_config.provider],
    )
    result_path = run_dir / "results.jsonl"
    done = (
        completed_task_keys(result_path, retry_failed=args.retry_failed)
        if args.resume
        else set()
    )

    logger.rule("Contract batch run")
    logger.kv(
        "Batch configuration",
        {
            "Run name": run_config["run_name"],
            "Provider": run_config["provider"],
            "Model": run_config["model"],
            "Datasets": ", ".join(str(path) for path in dataset_paths),
            "Selected tasks": len(tasks),
            "Already recorded": len(done),
            "Resume": args.resume,
            "Retry failed": args.retry_failed,
            "Witness split": args.witness_split,
            "Private eval": not args.skip_private_eval,
            "Eval split": args.eval_split,
            "Output structure": args.output_structure,
            "Prompt variant": args.prompt_variant,
            "LLM timeout": args.timeout,
            "Exec timeout": args.exec_timeout,
            "Code private timeout": effective_private_timeout,
            "Contract private timeout": contract_private_timeout,
            "Sleep between tasks": args.sleep_between_tasks,
            "Results": result_path,
        },
    )

    agent: ContractAgent | None = None
    failed_this_invocation = 0
    processed_this_invocation = 0

    for position, task in enumerate(tasks, start=1):
        if task.task_key in done:
            logger.warning(
                f"Skipping recorded task {position}/{len(tasks)}: {task.task_key}"
            )
            continue

        logger.rule(f"Batch task {position}/{len(tasks)} | {task.task_key}")
        started_at = utc_now()
        started = time.perf_counter()

        try:
            if agent is None:
                agent = ContractAgent(
                    llm=LLMClient(llm_config),
                    max_attempts=args.max_attempts,
                    code_refinement_attempts=args.code_refinement_attempts,
                    execution_timeout_sec=args.exec_timeout,
                    output_structure=args.output_structure,
                    prompt_variant=args.prompt_variant,
                )
            result = agent.run(
                task_id=task.task_id,
                prompt=task.prompt,
                entry_point=task.entry_point,
                public_witnesses=task.witnesses,
                metadata={
                    **task.metadata,
                    "Batch position": f"{position}/{len(tasks)}",
                    "Run name": run_config["run_name"],
                    "Prompt variant": args.prompt_variant,
                },
            )
            code_private_eval = None
            contract_private_eval = None
            if not args.skip_private_eval:
                private_tests = get_tests(task.source_row, args.eval_split)
                generated_code = (
                    result.code.output.code
                    if result.code and result.code.output
                    else None
                )
                merged_code = (
                    result.merged_contract.output.code
                    if result.merged_contract and result.merged_contract.output
                    else None
                )
                code_private_eval = evaluate_private_cases(
                    code=generated_code,
                    private_tests=private_tests,
                    timeout_sec=effective_private_timeout,
                    artifact="code_only",
                )
                contract_private_eval = evaluate_private_cases(
                    code=merged_code,
                    private_tests=private_tests,
                    timeout_sec=contract_private_timeout,
                    artifact="contracted_code",
                )
            record = persist_result(
                run_dir=run_dir,
                run_config=run_config,
                task=task,
                result=result,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started,
                code_private_eval=code_private_eval,
                contract_private_eval=contract_private_eval,
            )
            if args.example:
                _export_example_outputs(run_dir, record)
            if args.print_artifacts:
                show_pipeline_result(result)
        except Exception as exc:
            code_private_eval = None
            contract_private_eval = None
            if not args.skip_private_eval:
                private_tests = get_tests(task.source_row, args.eval_split)
                code_private_eval = evaluate_private_cases(
                    code=None,
                    private_tests=private_tests,
                    timeout_sec=effective_private_timeout,
                    artifact="code_only",
                )
                contract_private_eval = evaluate_private_cases(
                    code=None,
                    private_tests=private_tests,
                    timeout_sec=contract_private_timeout,
                    artifact="contracted_code",
                )
            record = persist_exception(
                run_dir=run_dir,
                run_config=run_config,
                task=task,
                started_at=started_at,
                elapsed_seconds=time.perf_counter() - started,
                error=exc,
                traceback_text=traceback.format_exc(),
                code_private_eval=code_private_eval,
                contract_private_eval=contract_private_eval,
            )
            logger.error(f"{task.task_key}: {record['error']}")

        processed_this_invocation += 1
        if record["status"] == "completed":
            logger.success(
                f"{task.task_key}: saved completed record | "
                f"pipeline_passed={record.get('pipeline_passed')} | "
                f"elapsed={record['elapsed_seconds']:.3f}s"
            )
        else:
            failed_this_invocation += 1
            logger.error(
                f"{task.task_key}: saved {record['status']} record | elapsed={record['elapsed_seconds']:.3f}s"
            )

        summary = write_summary(
            run_dir, run_config=run_config, selected_tasks=len(tasks)
        )
        dataset_summary = (summary.get("datasets") or {}).get(task.dataset, {})
        code_private = dataset_summary.get("code_private_eval", {})
        contract_private = dataset_summary.get("contract_private_eval", {})
        logger.info(
            f"Progress[{task.dataset}]: "
            f"recorded={dataset_summary.get('recorded_tasks', 0)}/"
            f"{selected_by_dataset.get(task.dataset, 0)} | "
            f"pipeline_passed={dataset_summary.get('pipeline_passed', 0)} | "
            f"code Pass@1={code_private.get('pass_at_1_rate')} | "
            f"code APR={code_private.get('apr_micro')} | "
            f"contract private Pass@1={contract_private.get('pass_at_1_rate')} | "
            f"contract APR={contract_private.get('apr_micro')} | "
            f"contract completeness={contract_private.get('completeness_micro')}"
        )
        if args.sleep_between_tasks > 0:
            logger.info(f"Sleeping {args.sleep_between_tasks:.3f}s before next task")
            time.sleep(args.sleep_between_tasks)
        if args.fail_fast and record["status"] != "completed":
            break

    summary = write_summary(run_dir, run_config=run_config, selected_tasks=len(tasks))
    logger.rule("Batch result")
    code_private = summary.get("code_private_eval", {})
    contract_private = summary.get("contract_private_eval", {})
    comparison = summary.get("private_eval_comparison", {})
    logger.kv(
        "Batch summary",
        {
            "Selected tasks": len(tasks),
            "Processed now": processed_this_invocation,
            "Recorded tasks": summary["recorded_tasks"],
            "Completed tasks": summary["completed_tasks"],
            "Exceptions": summary["failed_tasks"],
            "Pipeline passed": summary["pipeline_passed"],
            "Pipeline pass rate": summary["pipeline_pass_rate"],
            "Code private evaluated": code_private.get("evaluated_instances"),
            "Code private Pass@1 rate": code_private.get("pass_at_1_rate"),
            "Code private APR micro": code_private.get("apr_micro"),
            "Code private completeness": code_private.get("completeness_micro"),
            "Contract private evaluated": contract_private.get("evaluated_instances"),
            "Contract private Pass@1 rate": contract_private.get("pass_at_1_rate"),
            "Contract private APR micro": contract_private.get("apr_micro"),
            "Contract private APR macro": contract_private.get("apr_macro"),
            "Contract private completeness": contract_private.get("completeness_micro"),
            "Correctness delta": comparison.get("contract_correctness_delta"),
            "Completeness delta": comparison.get("contract_completeness_delta"),
            "Contract violations": contract_private.get("contract_violations"),
            "Runtime errors": contract_private.get("runtime_errors"),
            "Timeouts": contract_private.get("timeouts"),
            "Wrong outputs": contract_private.get("wrong_outputs"),
            "Results": summary["results_path"],
            "Summary": run_dir / "summary.json",
        },
    )
    dataset_rows = []
    for dataset, dataset_summary in (summary.get("datasets") or {}).items():
        dataset_code = dataset_summary.get("code_private_eval", {})
        dataset_contract = dataset_summary.get("contract_private_eval", {})
        dataset_rows.append(
            {
                "dataset": dataset,
                "recorded": dataset_summary.get("recorded_tasks"),
                "pipeline": dataset_summary.get("pipeline_pass_rate"),
                "code_pass_at_1": dataset_code.get("pass_at_1_rate"),
                "code_apr": dataset_code.get("apr_micro"),
                "contract_pass_at_1": dataset_contract.get("pass_at_1_rate"),
                "contract_apr": dataset_contract.get("apr_micro"),
            }
        )
    logger.records(
        "Dataset-wise private evaluation",
        dataset_rows,
        {
            "dataset": "Dataset",
            "recorded": "Recorded",
            "pipeline": "Pipeline rate",
            "code_pass_at_1": "Code Pass@1",
            "code_apr": "Code APR",
            "contract_pass_at_1": "Contract Pass@1",
            "contract_apr": "Contract APR",
        },
    )
    if args.fail_on_task_error and (failed_this_invocation or summary["failed_tasks"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
