import ast
import fnmatch
import json
import tomllib

from contract.agent.schema.specification import RawSpecification
from contract.utils.prompt_renderer import PROMPT_ROOT, render_prompt
from tests.support import OfflineTestCase, ROOT


class PromptReadinessTests(OfflineTestCase):
    def test_all_packaged_templates_render_for_each_supported_variant(self):
        root = ROOT / "motivating_example"
        spec = json.loads((root / "nlc.json").read_text())
        context = {
            "task_id": "football",
            "task_prompt": (root / "problem_statement.txt").read_text(),
            "entry_point_json": json.dumps("solve"),
            "schema_json": json.dumps(RawSpecification.model_json_schema()),
            "contracts_json": json.dumps(spec["contracts"]),
            "skeleton": (root / "interface.py").read_text(),
            "interface_json": json.dumps({"entry_point": "solve", "parameters": ["s"]}),
            "executable_coverage_json": (
                root / "contract_lowering_coverage.json"
            ).read_text(),
            "public_witnesses_json": (root / "public_test_cases.json").read_text(),
            "feedback": "Offline repair-template rendering check.",
            "error": "Offline schema-validation example.",
            "attempt": 1,
            "max_attempts": 3,
            "previous_error": "Offline schema-validation example.",
            "previous_raw_output": "{}",
            "repair_feedback": "Return a complete probe plan.",
        }
        templates = sorted(PROMPT_ROOT.rglob("*.j2"))
        self.assertEqual(len(templates), 15)
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertIn(
            "agent/prompts/*/*.j2",
            config["tool"]["setuptools"]["package-data"]["contract"],
        )
        for template in templates:
            for variant in ["stripped", "fewshot", "cot", "contract"]:
                for structure in ["raw", "structured"]:
                    with self.subTest(
                        template=template.name, variant=variant, structure=structure
                    ):
                        rendered = render_prompt(
                            template.relative_to(PROMPT_ROOT).as_posix(),
                            prompt_variant=variant,
                            output_structure=structure,
                            **context,
                        )
                        self.assertTrue(rendered.strip())

    def test_every_stage_reference_resolves_and_every_prompt_is_used_and_packaged(self):
        templates = {
            path.relative_to(PROMPT_ROOT).as_posix()
            for path in PROMPT_ROOT.rglob("*.j2")
        }
        cases = {
            "specification",
            "interface",
            "code_generation",
            "executable_contract",
            "contract_probe",
        }
        self.assertEqual(
            templates,
            {
                f"{case}/{role}.j2"
                for case in cases
                for role in ["system", "user", "repair"]
            },
        )
        references = set()
        for stage in (ROOT / "src/contract/agent/stages").glob("*.py"):
            for node in ast.walk(ast.parse(stage.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "render_prompt"
                ):
                    self.assertTrue(
                        node.args and isinstance(node.args[0], ast.Constant), str(stage)
                    )
                    name = node.args[0].value
                    with self.subTest(stage=stage.name, prompt=name):
                        self.assertIn(
                            name, templates, f"Unreachable prompt in {stage.name}"
                        )
                    references.add(name)
        self.assertEqual(
            references,
            templates,
            "A prompt is orphaned or a pipeline reference is missing",
        )
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        patterns = config["tool"]["setuptools"]["package-data"]["contract"]
        for template in templates:
            with self.subTest(packaged=template):
                self.assertTrue(
                    any(
                        fnmatch.fnmatchcase(f"agent/prompts/{template}", pattern)
                        for pattern in patterns
                    )
                )
