import contextlib
import io
import json
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contract.cli import main, parse_args
from tests.support import OfflineTestCase


class CliTests(OfflineTestCase):
    def test_all_nine_combinations_keep_model_dataset_and_output_separate(self):
        models = {
            "gpt54mini": "openai/gpt-5.4-mini",
            "haiku45": "anthropic/claude-haiku-4.5",
            "llama4_maverick": "meta-llama/llama-4-maverick",
        }
        benchmarks = {
            "apps": "apps_specine_update.jsonl",
            "code_contests": "code_contests_update.jsonl",
            "xcodeeval": "xCodeEval_update.jsonl",
        }
        for model, model_id in models.items():
            for benchmark, filename in benchmarks.items():
                with self.subTest(model=model, benchmark=benchmark):
                    args = parse_args(["--model", model, "--benchmark", benchmark])
                    self.assertEqual(args.provider, "openrouter")
                    self.assertEqual(args.model, model_id)
                    self.assertEqual(args.datasets, [[str(Path("data") / filename)]])
                    self.assertEqual(
                        args.output_root, f"results/batch/{model}/{benchmark}"
                    )

    def test_benchmark_selection_removes_duplicates_and_supports_all(self):
        args = parse_args(["--model", "haiku45", "--benchmark", "apps", "apps"])
        self.assertEqual(len(args.datasets[0]), 1)
        args = parse_args(["--model", "haiku45", "--benchmark", "all"])
        self.assertEqual(len(args.datasets[0]), 3)

    def test_readiness_option_never_starts_generation(self):
        with (
            patch("contract.readiness.run_checks", return_value=0) as checks,
            patch(
                "contract.cli.LLMClient",
                side_effect=AssertionError("Unexpected model client"),
            ),
        ):
            with self.assertRaises(SystemExit) as result:
                main(["--check"])
        self.assertEqual(result.exception.code, 0)
        checks.assert_called_once_with()

    def test_interactive_selection_reprompts_and_defaults_to_one_task(self):
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=["invalid", "2", "4", ""]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            args = parse_args(["--interactive"])
        self.assertEqual(args.model, "anthropic/claude-haiku-4.5")
        self.assertEqual(args.benchmark, ["xcodeeval"])
        self.assertEqual(args.limit, 1)

    def test_invalid_requests_fail_before_execution(self):
        requests = [
            ["--benchmark", "apps"],
            ["--model", "haiku45", "--benchmark", "apps", "--resume"],
            ["--model", "haiku45", "--benchmark", "apps", "--start", "-1"],
            ["--model", "haiku45", "--benchmark", "apps", "--limit", "-1"],
            ["--model", "haiku45", "--benchmark", "apps", "--dataset", "a.json"],
            ["--model", "haiku45", "--benchmark", "apps", "--provider", "ollama"],
            ["--example", "football", "--benchmark", "apps"],
            ["--example", "football", "--start", "1"],
        ]
        with (
            patch("sys.stdin.isatty", return_value=False),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for request in requests:
                with (
                    self.subTest(request=request),
                    self.assertRaises(SystemExit) as error,
                ):
                    parse_args(request)
                self.assertEqual(error.exception.code, 2)

    def test_default_is_the_single_football_case_with_saved_model_settings(self):
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            args = parse_args([])
        self.assertEqual(args.example, "football")
        self.assertEqual(args.model, "anthropic/claude-haiku-4.5")
        self.assertEqual(args.datasets, [["motivating_example/xCodeEval_update.json"]])
        self.assertEqual(args.limit, 1)
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.timeout, 60)
        self.assertEqual(args.output_structure, "raw")
        self.assertEqual(args.output_root, "motivating_example/runs/generated/haiku45")

    def test_example_settings_can_be_overridden(self):
        args = parse_args(
            [
                "--example",
                "football",
                "--model",
                "gpt54mini",
                "--max-tokens=4096",
                "--output_structure",
                "structured",
                "--timeout",
                "90",
            ]
        )
        self.assertEqual(args.model, "openai/gpt-5.4-mini")
        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.output_structure, "structured")
        self.assertEqual(args.timeout, 90)

    def test_football_run_exports_all_stages_without_overwriting_saved_artifacts(self):
        from contract.agent.contract import ContractAgentResult

        root = Path("motivating_example")
        snapshot = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ["nlc.json", "ec.py", "code.py", "ec_code.py"]
        }
        saved = json.loads((root / "runs/haiku_fianl/record.json").read_text())
        stages = {}
        definitions = [
            (
                "specification",
                "specification",
                "AgentStageResult",
                "specification",
                "Specification",
                "specification",
            ),
            (
                "interface",
                "interface",
                "InterfaceStageResult",
                "interface",
                "ProgramInterface",
                "interface",
            ),
            (
                "code",
                "code_generation",
                "CodeStageResult",
                "code",
                "CodeArtifact",
                "code",
            ),
            (
                "executable_contract",
                "executable_contract",
                "ExecutableContractStageResult",
                "executable_contract",
                "ExecutableContractArtifact",
                "executable_contract",
            ),
            (
                "merged_contract",
                "merge_contract",
                "MergeContractStageResult",
                "merged_contract",
                "MergedContractArtifact",
                "merged_contract",
            ),
            (
                "contract_probe",
                "contract_probe",
                "ContractProbeStageResult",
                "contract_probe",
                "ContractProbePlan",
                "contract_probe",
            ),
        ]
        for (
            attribute,
            stage_module,
            stage_type,
            schema_module,
            schema_type,
            key,
        ) in definitions:
            result_type = getattr(
                importlib.import_module(f"contract.agent.stages.{stage_module}"),
                stage_type,
            )
            output_type = getattr(
                importlib.import_module(f"contract.agent.schema.{schema_module}"),
                schema_type,
            )
            stages[attribute] = result_type(
                name=attribute, output=output_type.model_validate(saved["outputs"][key])
            )
        result = ContractAgentResult(task_id=saved["task_id"], **stages)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("contract.cli.LLMClient"),
                patch("contract.cli.ContractAgent") as agent,
                patch("contract.cli.logger"),
            ):
                agent.return_value.run.return_value = result
                main(
                    [
                        "--example",
                        "football",
                        "--output-root",
                        tmp,
                        "--run-name",
                        "mock_generation",
                        "--skip-private-eval",
                    ]
                )
            output = Path(tmp) / "mock_generation"
            self.assertEqual(
                agent.return_value.run.call_args.kwargs["task_id"], saved["task_id"]
            )
            for name in [
                "nlc.json",
                "ec.py",
                "code.py",
                "ec_code.py",
                "interface.py",
                "contract_probes.json",
            ]:
                self.assertTrue((output / name).is_file(), name)
            record = json.loads((output / "results.jsonl").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["problem_id"], 782)
            self.assertEqual(
                json.loads((output / "run_config.json").read_text())["example"],
                "football",
            )
        for name, digest in snapshot.items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), digest
            )

    def test_custom_provider_and_dataset(self):
        args = parse_args(
            [
                "--model",
                "local-model",
                "--provider",
                "ollama",
                "--dataset",
                "input.json",
            ]
        )
        self.assertEqual(args.model, "local-model")
        self.assertEqual(args.datasets, [["input.json"]])

    def test_dry_run_reads_inputs_without_clients_or_output_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "fixture.jsonl"
            dataset.write_text(
                json.dumps({"task_id": "echo", "prompt": "Echo stdin."}) + "\n"
            )
            output = root / "new-run"
            captured = io.StringIO()
            with (
                patch(
                    "contract.cli.LLMClient",
                    side_effect=AssertionError("Model client initialized"),
                ),
                patch(
                    "contract.cli.ContractAgent",
                    side_effect=AssertionError("Agent initialized"),
                ),
                contextlib.redirect_stdout(captured),
            ):
                main(
                    [
                        "--model",
                        "gpt54mini",
                        "--dataset",
                        str(dataset),
                        "--output-root",
                        str(output),
                        "--dry-run",
                    ]
                )
            config = json.loads(captured.getvalue())
            self.assertEqual(config["model"], "openai/gpt-5.4-mini")
            self.assertEqual(sum(config["tasks_per_benchmark"].values()), 1)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [dataset])


if __name__ == "__main__":
    unittest.main()
