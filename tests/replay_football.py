import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from contract.agent.contract import ContractAgent
from contract.cli import _export_example_outputs
from contract.eval.execution import run_python_program_detailed
from contract.eval.private import evaluate_private_cases
from contract.utils.dataset import build_dataset_task, get_tests
from contract.utils.results import persist_result
from tests.support import ROOT


class SavedResponses:
    def __init__(self):
        self.stages = [
            "specification",
            "interface",
            "code",
            "executable_contract",
            "contract_probe",
        ]
        self.used = []

    def bind(self, **kwargs):
        return self

    def invoke(self, payload, **kwargs):
        if not self.stages:
            raise AssertionError(
                "Pipeline requested an unexpected extra model response"
            )
        stage = self.stages.pop(0)
        self.used.append(stage)
        return (
            ROOT / "motivating_example/runs/haiku_fianl/raw_outputs" / f"{stage}.txt"
        ).read_text()


def run():
    root = ROOT / "motivating_example"
    row = json.loads((root / "xCodeEval_update.json").read_text())
    task = build_dataset_task(
        row,
        source_path=root / "xCodeEval_update.json",
        row_index=0,
        test_split="public",
    )
    saved = SavedResponses()
    client = SimpleNamespace(
        llm=saved,
        config=SimpleNamespace(max_tokens=2048, provider="offline-saved-response"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        agent = ContractAgent(
            llm=client,
            max_attempts=1,
            output_structure="raw",
            code_refinement_attempts=0,
            execution_timeout_sec=5,
        )
        result = agent.run(
            task_id=task.task_id,
            prompt=task.prompt,
            entry_point=task.entry_point,
            public_witnesses=task.witnesses,
            metadata=task.metadata,
        )
    assert saved.stages == [], f"Not all stages completed: {saved.stages}"
    assert result.specification.output is not None
    assert result.interface.output is not None
    assert result.code.static_validation_passed and result.code.execution_witness_passed
    assert result.executable_contract.static_validation_passed
    assert (
        result.merged_contract.static_validation_passed
        and result.merged_contract.execution_witness_passed
    )
    assert (
        result.contract_probe.output is not None and result.contract_probe.output.probes
    )
    evaluations = {}
    for name, code in [
        ("code", result.code.output.code),
        ("ec_code", result.merged_contract.output.code),
    ]:
        report = evaluate_private_cases(
            code=code,
            private_tests=get_tests(row, "private"),
            timeout_sec=5,
            artifact=name,
        )
        assert report["passed"] == report["total"] == 43, (name, report)
        evaluations[name] = {"passed": report["passed"], "total": report["total"]}
    run_dir = Path.cwd() / "offline-replay"
    record = persist_result(
        result,
        run_dir=run_dir,
        run_config={
            "run_name": "offline-replay",
            "provider": "offline-saved-response",
            "model": "recorded-responses",
        },
        task=task,
    )
    _export_example_outputs(run_dir, record)
    for name in [
        "nlc.json",
        "interface.py",
        "code.py",
        "ec.py",
        "ec_code.py",
        "contract_probes.json",
        "contract_lowering_coverage.json",
    ]:
        assert (run_dir / name).is_file(), name
    for filename in ["code.py", "ec_code.py"]:
        for stdin, expected in [
            ("001001\n", "NO"),
            ("1000000001\n", "YES"),
            ("0000000\n", None),
        ]:
            observed = run_python_program_detailed(
                (root / filename).read_text(), stdin, timeout_sec=5
            )
            if expected is not None:
                assert observed.returncode == 0 and observed.stdout == expected, (
                    filename,
                    observed,
                )
            elif filename == "ec_code.py":
                assert (
                    observed.returncode != 0
                    and "icontract.errors.ViolationError" in observed.stderr
                )
                assert "'0' in s and '1' in s" in observed.stderr
            else:
                assert observed.returncode == 0 and observed.stdout == "YES"
    print(
        json.dumps(
            {
                "response_source": "bundled historical text, not new model calls",
                "stages": saved.used,
                "private_evaluations": evaluations,
                "saved_program_replays": 6,
            }
        )
    )
