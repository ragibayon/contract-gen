from collections import Counter
import hashlib
import json
from pathlib import Path

from contract.utils.dataset import build_dataset_task
from tests.support import OfflineTestCase, ROOT


class ArtifactReadinessTests(OfflineTestCase):
    def test_bundled_inputs_results_prompts_and_football_evidence(self):
        manifest = json.loads((ROOT / "tests/artifact_manifest.json").read_text())
        entries = manifest["files"]
        self.assertEqual(len({entry["path"] for entry in entries}), len(entries))
        dataset_keys = {}
        result_keys = []
        subset_keys = []
        for entry in entries:
            path = ROOT / entry["path"]
            with self.subTest(file=entry["path"]):
                self.assertTrue(
                    path.is_file(), f"Missing submitted artifact: {entry['path']}"
                )
                self.assertEqual(
                    path.stat().st_size,
                    entry["size_bytes"],
                    f"File size changed: {entry['path']}",
                )
                digest = hashlib.sha256()
                count = 0
                keys = set()
                model_sets = Counter()
                with path.open("rb") as handle:
                    if path.suffix == ".jsonl":
                        for line_no, line in enumerate(handle, 1):
                            digest.update(line)
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            self.assertIsInstance(
                                row,
                                dict,
                                f"Expected a JSON object: {entry['path']}:{line_no}",
                            )
                            if entry["kind"] == "result":
                                self.assertEqual(row["dataset"], entry["dataset"])
                                key = row["task_key"]
                                models = {
                                    attempt["model_name"]
                                    for stage in (
                                        (row.get("llm_metrics") or {}).get("by_stage")
                                        or {}
                                    ).values()
                                    for attempt in (stage or {}).get("attempts") or []
                                    if attempt.get("model_name")
                                }
                                model_sets[
                                    " | ".join(sorted(models)) or "unavailable"
                                ] += 1
                            else:
                                task = build_dataset_task(
                                    row,
                                    source_path=path,
                                    row_index=count,
                                    test_split="public",
                                )
                                self.assertEqual(task.dataset, entry["dataset"])
                                key = task.task_key
                            self.assertNotIn(
                                key, keys, f"Duplicate task: {entry['path']}:{line_no}"
                            )
                            keys.add(key)
                            count += 1
                    else:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                self.assertEqual(
                    digest.hexdigest(),
                    entry["sha256"],
                    f"Artifact checksum mismatch: {entry['path']}",
                )
                if path.suffix == ".json":
                    json.loads(path.read_text())
                if "records" in entry:
                    self.assertEqual(count, entry["records"], entry["path"])
                if entry["kind"] == "dataset":
                    dataset_keys[entry["dataset"]] = keys
                elif entry["kind"] == "subset":
                    subset_keys.append((entry, keys))
                elif entry["kind"] == "result":
                    self.assertEqual(
                        dict(model_sets),
                        entry["response_model_sets"],
                        "Historical model evidence changed; do not assume all records match their filename's model.",
                    )
                    result_keys.append((entry, keys))
        self.assertEqual(len(dataset_keys), 3)
        self.assertEqual(sum(len(keys) for keys in dataset_keys.values()), 765)
        self.assertEqual(len(result_keys), 9)
        self.assertEqual(sum(len(keys) for _, keys in result_keys), 2295)
        for entry, keys in result_keys:
            with self.subTest(population=entry["path"]):
                self.assertEqual(keys, dataset_keys[entry["dataset"]])
        for entry, keys in subset_keys:
            with self.subTest(subset=entry["path"]):
                self.assertTrue(keys.issubset(dataset_keys[entry["dataset"]]))
