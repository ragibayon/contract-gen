from contextlib import chdir
from pathlib import Path
import sys
import unittest


def run_checks() -> int:
    root = Path(__file__).resolve().parents[2]
    if not (root / "tests" / "artifact_manifest.json").is_file():
        print(
            "NOT READY: tests require the complete repository checkout, including tests/ and artifact files.",
            file=sys.stderr,
        )
        return 2
    print(
        "Running offline readiness checks. This reads the bundled datasets and saved results; no API key is needed.",
        flush=True,
    )
    with chdir(root):
        suite = unittest.TestLoader().discover(
            str(root / "tests"), top_level_dir=str(root)
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        print(
            "NOT READY: resolve the failed or skipped checks above. Historical inputs and results were not modified.",
            file=sys.stderr,
        )
        return 1
    print(
        "READY: offline artifact and pipeline checks passed. Live provider access and fresh LLM output quality have not been verified."
    )
    return 0
