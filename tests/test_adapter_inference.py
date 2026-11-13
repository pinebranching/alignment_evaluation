from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import yaml

from inference.huggingface_runtime import load_causal_lm, render_prompt
from orchestrator.config import load_master_config, validate_master_config
from orchestrator.schemas import StagePlan
from orchestrator.stage_builders import build_inference_config, resolve_inference_model_payload


CONFIG_PATH = Path(__file__).resolve().parents[1] / "exemplary_scenario_runs" / (
    "orchestrator_config.qwen3_8b_adapter_sft_safe_refusal.yaml"
)
DPO_KTO_CONFIG_PATH = CONFIG_PATH.with_name(
    "orchestrator_config.qwen3_8b_adapter_dpo_kto_safe_refusal.yaml"
)
QWEN14_CONFIG_PATH = CONFIG_PATH.with_name(
    "orchestrator_config.qwen3_14b_adapter_all_safe_refusal.yaml"
)
OPEN_MODEL_CONFIG_PATHS = (
    CONFIG_PATH.with_name("orchestrator_config.mistral7b_v03_adapter_all_safe_refusal.yaml"),
    CONFIG_PATH.with_name("orchestrator_config.olmo2_7b_adapter_all_safe_refusal.yaml"),
    CONFIG_PATH.with_name("orchestrator_config.granite33_8b_adapter_all_safe_refusal.yaml"),
)


class AdapterInferenceTests(TestCase):
    def test_qwen_config_validates_when_datasets_are_present(self):
        for config_path in (
            CONFIG_PATH,
            DPO_KTO_CONFIG_PATH,
            QWEN14_CONFIG_PATH,
            *OPEN_MODEL_CONFIG_PATHS,
        ):
            with self.subTest(config=config_path.name), TemporaryDirectory() as temp:
                root = Path(temp)
                copied_config = root / "configs" / "github_repo" / config_path.name
                copied_config.parent.mkdir(parents=True)
                copied_config.write_bytes(config_path.read_bytes())
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                for dataset in raw["catalog"]["datasets"].values():
                    path = (copied_config.parent / dataset["path"]).resolve()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                config = load_master_config(copied_config)
                validate_master_config(config)
                self.assertEqual(config.output_root, root / "runs")

    def test_open_model_configs_have_all_adapter_branches(self):
        expected_models = {
            "orchestrator_config.mistral7b_v03_adapter_all_safe_refusal.yaml": (
                "mistral7b_v03",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistral",
            ),
            "orchestrator_config.olmo2_7b_adapter_all_safe_refusal.yaml": (
                "olmo2_7b",
                "allenai/OLMo-2-1124-7B-Instruct",
                "olmo",
            ),
            "orchestrator_config.granite33_8b_adapter_all_safe_refusal.yaml": (
                "granite33_8b",
                "ibm-granite/granite-3.3-8b-instruct",
                "granite3",
            ),
        }
        for config_path in OPEN_MODEL_CONFIG_PATHS:
            with self.subTest(config=config_path.name):
                slug, model_id, template = expected_models[config_path.name]
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                model = raw["catalog"]["models"][f"{slug}_base"]
                self.assertEqual(model["model_name_or_path"], model_id)
                self.assertEqual(model["template"], template)
                self.assertIsNone(raw["defaults"]["alignment"]["tuning"]["quantization_bit"])
                self.assertIs(raw["defaults"]["alignment"]["output"]["export_merged_model"], False)
                selected = {
                    candidate["candidate_id"]
                    for candidate in raw["alignment"]["candidates"]
                }
                self.assertEqual(selected, {
                    f"{slug}_base",
                    f"{slug}_sft_safe_refusal",
                    f"{slug}_dpo_safe_refusal",
                    f"{slug}_kto_safe_refusal",
                })
                for section in ("inference", "evaluation"):
                    for benchmark in raw[section]["benchmarks"]:
                        self.assertEqual(set(benchmark["candidate_refs"]), selected)

    def test_qwen14_config_runs_all_four_candidates_with_adapters(self):
        raw = yaml.safe_load(QWEN14_CONFIG_PATH.read_text(encoding="utf-8"))
        selected = {candidate["candidate_id"] for candidate in raw["alignment"]["candidates"]}
        self.assertEqual(selected, {
            "qwen3_14b_base",
            "qwen3_14b_sft_safe_refusal",
            "qwen3_14b_dpo_safe_refusal",
            "qwen3_14b_kto_safe_refusal",
        })
        self.assertEqual(raw["catalog"]["models"]["qwen3_14b_base"]["model_name_or_path"], "Qwen/Qwen3-14B")
        self.assertIsNone(raw["defaults"]["alignment"]["tuning"]["quantization_bit"])
        self.assertIs(raw["defaults"]["alignment"]["output"]["export_merged_model"], False)
        self.assertIs(raw["defaults"]["inference"]["generation"]["enable_thinking"], False)
        for section in ("inference", "evaluation"):
            for benchmark in raw[section]["benchmarks"]:
                self.assertEqual(set(benchmark["candidate_refs"]), selected)

    def test_dpo_kto_config_contains_only_requested_candidates(self):
        raw = yaml.safe_load(DPO_KTO_CONFIG_PATH.read_text(encoding="utf-8"))
        selected = {candidate["candidate_id"] for candidate in raw["alignment"]["candidates"]}
        self.assertEqual(selected, {
            "qwen3_8b_dpo_safe_refusal",
            "qwen3_8b_kto_safe_refusal",
        })
        for section in ("inference", "evaluation"):
            for benchmark in raw[section]["benchmarks"]:
                self.assertEqual(set(benchmark["candidate_refs"]), selected)

    def test_orchestrator_passes_base_model_and_adapter_separately(self):
        config = load_master_config(CONFIG_PATH)
        plan = StagePlan(
            branch_id="sft-test",
            candidate_id="qwen3_8b_sft_safe_refusal",
            candidate_kind="aligned",
            method="sft",
            run_name="sft-test",
            inference_benchmark_id="harmful_refusal_action_main",
        )
        benchmark = config.raw["inference"]["benchmarks"][0]
        result = {
            "primary_artifact_path": "/run/adapter",
            "artifacts": {
                "adapter_path": "/run/adapter",
                "exported_model_path": None,
            },
        }

        model = resolve_inference_model_payload(config, plan, benchmark, result)
        self.assertEqual(model["model_name_or_path"], "Qwen/Qwen3-8B")
        self.assertEqual(model["tokenizer_name_or_path"], "Qwen/Qwen3-8B")
        self.assertEqual(model["adapter_name_or_path"], "/run/adapter")

        generated = build_inference_config(config, plan, Path("/run/branch"), result)
        self.assertEqual(generated["inference"]["model"]["adapter_name_or_path"], "/run/adapter")
        self.assertIs(generated["inference"]["generation"]["enable_thinking"], False)

        result["artifacts"]["exported_model_path"] = "/run/merged"
        merged = resolve_inference_model_payload(config, plan, benchmark, result)
        self.assertEqual(merged["model_name_or_path"], "/run/merged")
        self.assertIsNone(merged["adapter_name_or_path"])

    def test_runtime_attaches_adapter_to_base_model(self):
        torch = ModuleType("torch")
        torch.float16 = object()
        torch.bfloat16 = object()
        torch.float32 = object()
        tokenizer = MagicMock(pad_token_id=0, eos_token_id=1)
        tokenizer.get_chat_template.return_value = "template"
        base = MagicMock()
        adapted = MagicMock()
        adapted.eval.return_value = adapted
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=MagicMock(return_value=base))
        transformers_utils = ModuleType("transformers.utils")
        transformers_utils.logging = MagicMock()
        peft = ModuleType("peft")
        peft.PeftModel = SimpleNamespace(from_pretrained=MagicMock(return_value=adapted))

        with patch.dict("sys.modules", {
            "torch": torch,
            "transformers": transformers,
            "transformers.utils": transformers_utils,
            "peft": peft,
        }):
            loaded = load_causal_lm(
                model_name_or_path="Qwen/Qwen3-8B",
                adapter_name_or_path="/run/adapter",
                tokenizer_name_or_path="Qwen/Qwen3-8B",
                torch_dtype="bfloat16",
                device="cuda",
            )

        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        base.to.assert_called_once_with("cuda")
        peft.PeftModel.from_pretrained.assert_called_once_with(base, "/run/adapter")
        self.assertIs(loaded.model, adapted)

    def test_qwen_thinking_switch_is_forwarded_to_chat_template(self):
        tokenizer = MagicMock()
        tokenizer.get_chat_template.return_value = "template"
        render_prompt(tokenizer, "prompt", enable_thinking=False)
        self.assertIs(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"], False)
