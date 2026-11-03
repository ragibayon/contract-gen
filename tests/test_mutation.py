import ast
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from contract.cli import main
from contract.eval.mutation import (
    candidate_pool,
    merge_contract,
    output_outcome,
    parse_args,
    registered_operators,
    run_program,
    run_study,
    select_mutations,
)
from tests.support import OfflineTestCase

SOURCE = 'def solve(n):\n    return n + 1\n\ndef main():\n    print(solve(int(input())))\n\nif __name__ == "__main__":\n    main()\n'
CONTRACT = "import icontract\n@icontract.require(lambda n: n >= 0)\n@icontract.ensure(lambda n, result: result == n + 1)\n"


class MutationTests(OfflineTestCase):
    def fixture(self, root, *, source=SOURCE, contract=CONTRACT, expected="3"):
        code = root / "code.py"
        ec = root / "ec.py"
        dataset = root / "task.json"
        code.write_text(source)
        ec.write_text(contract)
        dataset.write_text(
            json.dumps(
                {
                    "id": "arithmetic",
                    "prompt": "Return the input plus one.",
                    "private_test_cases": {"inputs": ["2\n"], "outputs": [expected]},
                }
            )
        )
        args = parse_args(
            [
                "--code",
                str(code),
                "--contract",
                str(ec),
                "--dataset",
                str(dataset),
                "--output",
                str(root / "study"),
                "--pool",
                "arithmetic",
                "--scope",
                "target",
                "--oracle",
                "exact",
                "--max-mutants",
                "2",
            ]
        )
        return args

    def test_full_pool_selection_is_unique_reproducible_and_bounded(self):
        candidates, counts = candidate_pool(SOURCE)
        self.assertEqual(len(registered_operators()), 15)
        self.assertGreater(counts["eligible"], 0)
        first = select_mutations(candidates, limit=3, seed=42)
        self.assertEqual(first, select_mutations(candidates, limit=3, seed=42))
        self.assertEqual(len(first), min(3, len(candidates)))
        self.assertEqual(
            len({value["ast_sha256"] for value in candidates}), len(candidates)
        )
        for candidate in candidates:
            compile(ast.parse(candidate["source"]), "<candidate>", "exec")
            self.assertNotEqual(
                ast.dump(ast.parse(candidate["source"])), ast.dump(ast.parse(SOURCE))
            )

    def test_operator_and_function_body_selection(self):
        candidates, _ = candidate_pool(SOURCE, pool="arithmetic", scope="target")
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate["line"], 2)
            self.assertIn("operator_swap_op", candidate["operators"])
            self.assertEqual(candidate["node_type"], "BinaryOperation")
        numbers, _ = candidate_pool(
            SOURCE, operators=["operator_number"], scope="target"
        )
        self.assertTrue(numbers)
        self.assertTrue(
            all("operator_number" in value["operators"] for value in numbers)
        )
        with self.assertRaises(ValueError):
            candidate_pool(SOURCE, operators=["not-an-operator"])
        with self.assertRaises(ValueError):
            candidate_pool(merge_contract(SOURCE, CONTRACT))

    def test_preparation_writes_frozen_selection_without_running_programs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.prepare_only = True
            with patch(
                "contract.eval.mutation.run_program",
                side_effect=AssertionError("Preparation executed a program"),
            ):
                summary = run_study(args)
            self.assertEqual(summary["status"], "prepared")
            self.assertGreater(summary["selected_mutants"], 0)
            self.assertEqual(summary["mutants_executed"], 0)
            selection = json.loads((args.output / "selection.json").read_text())
            for mutation in selection["mutants"]:
                content = (
                    args.output / "mutants" / f"{mutation['id']}.py"
                ).read_bytes()
                self.assertEqual(
                    hashlib.sha256(content).hexdigest(), mutation["source_sha256"]
                )
            with self.assertRaises(FileExistsError):
                run_study(args)

    def test_actual_mutants_and_unchanged_contracts_discriminate_wrong_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            original_code = args.code.read_bytes()
            original_contract = args.contract.read_bytes()
            summary = run_study(args)
            self.assertEqual(summary["status"], "completed")
            self.assertGreater(
                summary["reference_controlled_postcondition_violations"], 0
            )
            self.assertEqual(
                summary["reference_eligible_wrong_output_executions"],
                summary["reference_controlled_postcondition_violations"],
            )
            self.assertEqual(args.code.read_bytes(), original_code)
            self.assertEqual(args.contract.read_bytes(), original_contract)
            self.assertEqual((args.output / "ec.py").read_bytes(), original_contract)

    def test_bad_reference_and_overstrong_contract_receive_no_detection_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), expected="999")
            with patch(
                "contract.eval.mutation.candidate_pool",
                side_effect=AssertionError("Reference gate was bypassed"),
            ):
                result = run_study(args)
            self.assertEqual(result["status"], "reference_gate_failed")
            self.assertEqual(result["mutants_executed"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(
                Path(tmp),
                contract="import icontract\n@icontract.ensure(lambda result: result == 999)\n",
            )
            result = run_study(args)
            self.assertGreater(
                result["raw_outcomes"].get("confirmed_wrong_output", 0), 0
            )
            self.assertEqual(result["reference_eligible_wrong_output_executions"], 0)
            self.assertEqual(result["reference_controlled_postcondition_violations"], 0)

    def test_survivors_unresolved_outputs_preconditions_and_failures_stay_separate(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(
                Path(tmp),
                source=SOURCE.replace("n + 1", "n + 0"),
                contract="import icontract\n@icontract.ensure(lambda n, result: result == n)\n",
                expected="2",
            )
            result = run_study(args)
            self.assertGreater(result["private_test_survivors"], 0)
            self.assertEqual(result["reference_controlled_postcondition_violations"], 0)
        self.assertEqual(
            output_outcome(
                {"state": "normal", "stdout": "other"}, "expected", "unverified"
            ),
            "unresolved_output_difference",
        )
        bad = run_program("raise RuntimeError('fixture failure')\n", "")
        self.assertEqual(output_outcome(bad, "expected", "exact"), "program_failure")
        timed_out = run_program("while True:\n    pass\n", "", timeout=0.2)
        self.assertEqual(timed_out["state"], "timeout")
        rejected = run_program(merge_contract(SOURCE, CONTRACT), "-1\n", trace=True)
        self.assertEqual(rejected["state"], "precondition_violation")
        self.assertEqual(rejected["predicate_calls"]["ensure"], 0)

    def test_cli_dispatch_keeps_mutation_work_separate_from_generation(self):
        with (
            patch("contract.eval.mutation.main") as mutation,
            patch(
                "contract.cli.LLMClient",
                side_effect=AssertionError("Unexpected model call"),
            ),
        ):
            main(["mutate", "--prepare-only"])
        mutation.assert_called_once_with(["--prepare-only"])

    def test_saved_result_selection_uses_embedded_artifacts_not_old_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self.fixture(root)
            results = root / "saved.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "task_key": "task/arithmetic",
                        "outputs": {
                            "code": {"code": SOURCE},
                            "executable_contract": {
                                "imports": "import icontract",
                                "decorators": "@icontract.ensure(lambda n, result: result == n + 1)",
                            },
                        },
                        "artifacts": {"generated.py": "/nonexistent/old/artifact.py"},
                        "run": {"model": "saved-model"},
                    }
                )
                + "\n"
            )
            args = parse_args(
                [
                    "--results",
                    str(results),
                    "--task-key",
                    "task/arithmetic",
                    "--dataset",
                    str(fixture.dataset),
                    "--output",
                    str(root / "selected"),
                    "--prepare-only",
                ]
            )
            result = run_study(args)
            self.assertEqual(result["status"], "prepared")
            self.assertEqual((args.output / "seed.py").read_text(), SOURCE)
            config = json.loads((args.output / "config.json").read_text())
            self.assertEqual(config["saved_record"]["task_key"], "task/arithmetic")
            self.assertEqual(
                config["saved_record"]["recorded_run"]["model"], "saved-model"
            )
