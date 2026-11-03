import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
import tomllib

from packaging.requirements import Requirement

import contract
from contract.cli import main
from contract.utils.dataset import get_tests
from tests.support import OfflineTestCase, ROOT, isolated_python


class EnvironmentReadinessTests(OfflineTestCase):
    def test_python_dependencies_and_current_repository_package(self):
        self.assertGreaterEqual(sys.version_info[:2], (3, 12))
        self.assertEqual(
            Path(contract.__file__).resolve().parent, ROOT / "src/contract"
        )
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        for requirement in config["project"]["dependencies"]:
            dependency = Requirement(requirement)
            with self.subTest(dependency=dependency.name):
                version = metadata.version(dependency.name)
                self.assertTrue(
                    dependency.specifier.contains(version),
                    f"Installed {dependency.name}=={version} does not satisfy {dependency.specifier}; run uv sync --frozen.",
                )
        self.assertEqual(
            config["project"]["scripts"]["contract"],
            f"{main.__module__}:{main.__name__}",
        )
        for module in [
            "icontract",
            "jinja2",
            "pydantic",
            "rich",
            "dotenv",
            "langchain_openai",
            "langchain_anthropic",
            "langchain_ollama",
        ]:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_module_cli_help_and_default_dry_run_need_no_api_key(self):
        for arguments in [["--help"], ["--dry-run"]]:
            with self.subTest(arguments=arguments):
                script = "import os, runpy, sys\nfrom tests.support import ROOT\nos.chdir(ROOT)\nfor key in list(os.environ):\n    if key.endswith('API_KEY'): del os.environ[key]\nos.environ['PYTHON_DOTENV_DISABLED'] = '1'\n"
                script += f"sys.argv = ['contract', *{arguments!r}]\nrunpy.run_module('contract', run_name='__main__')\n"
                result = isolated_python(script, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                if arguments == ["--dry-run"]:
                    config = json.loads(result.stdout)
                    self.assertEqual(config["example"], "football")
                    self.assertEqual(config["tasks_per_benchmark"], {"XcodeEval": 1})

    def test_public_private_and_generic_tests_remain_separate(self):
        row = {
            "public_test_cases": {"inputs": ["public"], "outputs": ["P"]},
            "private_test_cases": {"inputs": ["private"], "outputs": ["S"]},
        }
        self.assertEqual(get_tests(row, "public")["inputs"], ["public\n"])
        self.assertEqual(get_tests(row, "private")["inputs"], ["private\n"])
        self.assertEqual(get_tests(row, "auto")["inputs"], ["public\n"])
        generic = {"tests": [{"input": "hidden", "expected": "H"}]}
        self.assertEqual(get_tests(generic, "public")["inputs"], [])
        self.assertEqual(get_tests(generic, "auto")["inputs"], [])
        self.assertEqual(get_tests(generic, "all")["inputs"], ["hidden\n"])

    def test_lockfile_matches_project_runtime_requirements(self):
        from dotenv import dotenv_values
        from packaging.utils import canonicalize_name

        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        locked = next(
            package for package in lock["package"] if package["name"] == project["name"]
        )
        self.assertEqual(locked["version"], project["version"])
        self.assertEqual(lock["requires-python"], project["requires-python"])
        expected = {
            canonicalize_name(Requirement(value).name): str(
                Requirement(value).specifier
            )
            for value in project["dependencies"]
        }
        actual = {
            canonicalize_name(value["name"]): value.get("specifier", "")
            for value in locked["metadata"]["requires-dist"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual((ROOT / ".python-version").read_text().strip(), "3.12")
        self.assertIn("OPENROUTER_API_KEY", dotenv_values(ROOT / ".env.example"))
