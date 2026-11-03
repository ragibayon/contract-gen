import json

from tests.support import OfflineTestCase, ROOT, isolated_python


class FootballReadinessTests(OfflineTestCase):
    def test_football_matches_its_original_benchmark_row_and_saved_private_results(
        self,
    ):
        root = ROOT / "motivating_example"
        row = json.loads((root / "xCodeEval_update.json").read_text())
        provenance = json.loads(
            (root / "runs/haiku_fianl/dataset_row.json").read_text()
        )
        self.assertEqual(row, provenance["row"])
        with (ROOT / "data/xCodeEval_update.jsonl").open() as handle:
            for index, line in enumerate(handle):
                if index == provenance["zero_based_row"]:
                    self.assertEqual(row, json.loads(line))
                    break
            else:
                self.fail("Football row is missing from the full xCodeEval dataset")
        self.assertEqual(row["problem_id"], 782)
        self.assertEqual(row["src_uid"], "ed9a763362abc6ed40356731f1036b38")
        for filename in ["code_private_eval.json", "contract_private_eval.json"]:
            report = json.loads((root / "runs/haiku_fianl" / filename).read_text())
            self.assertEqual(report["passed"], 43)
            self.assertEqual(report["total"], 43)

    def test_full_pipeline_with_saved_responses_and_actual_program_execution(self):
        completed = isolated_python(
            "from tests.replay_football import run\nrun()\n", timeout=120
        )
        self.assertEqual(
            completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-4000:]
        )
        result = json.loads(completed.stdout)
        self.assertEqual(
            result["stages"],
            [
                "specification",
                "interface",
                "code",
                "executable_contract",
                "contract_probe",
            ],
        )
        self.assertEqual(
            result["private_evaluations"],
            {
                "code": {"passed": 43, "total": 43},
                "ec_code": {"passed": 43, "total": 43},
            },
        )
        self.assertEqual(result["saved_program_replays"], 6)
