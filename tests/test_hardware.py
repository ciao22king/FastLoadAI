import argparse
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastloadai.chat import trim_history
from fastloadai.cli import (
    _doctor_command,
    _models_command,
    _plan_command,
    _quant_label,
    _run_command,
    _select_gguf,
    build_parser,
)
from fastloadai.hardware import (
    HardwareSpecs,
    recommend_settings,
    recommended_context_size,
    usable_model_memory_gb,
)


class HardwareRecommendationTests(unittest.TestCase):
    def test_cpu_only_machine_keeps_gpu_offload_disabled(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=8)

        settings = recommend_settings(specs)

        self.assertEqual(settings["n_gpu_layers"], 0)
        self.assertEqual(settings["context_size"], 2048)
        self.assertTrue(settings["mmap"])

    def test_gpu_offload_only_when_model_fits_in_vram_budget(self):
        specs = HardwareSpecs(cpu_threads=16, ram_gb=32, gpu_name="GPU", gpu_vram_gb=16)

        self.assertEqual(recommend_settings(specs)["n_gpu_layers"], 0)
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "small.gguf"
            model_path.write_bytes(b"gguf")
            self.assertEqual(recommend_settings(specs, model_path)["n_gpu_layers"], 99)

    def test_usable_memory_reserves_system_headroom(self):
        specs = HardwareSpecs(cpu_threads=8, ram_gb=16)

        self.assertGreater(usable_model_memory_gb(specs), 0)
        self.assertLess(usable_model_memory_gb(specs), specs.ram_gb)

    def test_gguf_selection_uses_best_quant_that_fits(self):
        class Sibling:
            def __init__(self, filename, size):
                self.rfilename = filename
                self.size = size

        siblings = [
            Sibling("model-Q4_K_M.gguf", 4 * 1024**3),
            Sibling("model-Q5_K_M.gguf", 5 * 1024**3),
            Sibling("model-F16.gguf", 12 * 1024**3),
        ]

        selected = _select_gguf(siblings, 6, None, prefer="quality")

        self.assertEqual(selected["name"], "model-Q5_K_M.gguf")

    def test_gguf_selection_balanced_prefers_smaller_high_quality_quant(self):
        class Sibling:
            def __init__(self, filename, size):
                self.rfilename = filename
                self.size = size

        siblings = [
            Sibling("model-Q4_K_M.gguf", 4 * 1024**3),
            Sibling("model-Q5_K_M.gguf", 5 * 1024**3),
            Sibling("model-Q8_0.gguf", 8 * 1024**3),
        ]

        selected = _select_gguf(siblings, 16, None)

        self.assertEqual(selected["name"], "model-Q4_K_M.gguf")

    def test_gguf_selection_speed_prefers_q4(self):
        class Sibling:
            def __init__(self, filename, size):
                self.rfilename = filename
                self.size = size

        siblings = [
            Sibling("model-Q2_K.gguf", 2 * 1024**3),
            Sibling("model-Q4_K_M.gguf", 4 * 1024**3),
            Sibling("model-Q8_0.gguf", 8 * 1024**3),
        ]

        selected = _select_gguf(siblings, 16, None, prefer="speed")

        self.assertEqual(selected["name"], "model-Q4_K_M.gguf")

    def test_gguf_selection_prefers_k_quant_over_legacy(self):
        class Sibling:
            def __init__(self, filename, size):
                self.rfilename = filename
                self.size = size

        siblings = [
            Sibling("model-Q4_0.gguf", 400 * 1024**2),
            Sibling("model-Q4_K_M.gguf", 460 * 1024**2),
        ]

        selected = _select_gguf(siblings, 16, None)

        self.assertEqual(selected["name"], "model-Q4_K_M.gguf")

    def test_context_grows_when_model_leaves_memory_headroom(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=16)

        self.assertEqual(recommended_context_size(specs), 4096)
        self.assertEqual(recommended_context_size(specs, 0.6), 8192)

    def test_gguf_selection_uses_smallest_file_when_none_fit(self):
        class Sibling:
            def __init__(self, filename, size):
                self.rfilename = filename
                self.size = size

        siblings = [
            Sibling("model-Q4_K_M.gguf", 5 * 1024**3),
            Sibling("model-Q5_K_M.gguf", 6 * 1024**3),
        ]

        selected = _select_gguf(siblings, 3, None)

        self.assertEqual(selected["name"], "model-Q4_K_M.gguf")

    def test_run_resolves_downloaded_model_and_starts_interactive_conversation(self):
        specs = HardwareSpecs(cpu_threads=6, ram_gb=16)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models_dir = root / "models"
            models_dir.mkdir()
            model_path = models_dir / "test-model-Q4_K_M.gguf"
            model_path.write_bytes(b"gguf")
            config_path = root / "fastloadai-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": str(model_path),
                        "model_repo": "test-org/test-model-GGUF",
                        "model_size_bytes": 4,
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser().parse_args(
                [
                    "run",
                    "test-model",
                    "--config",
                    str(config_path),
                    "--models-dir",
                    str(models_dir),
                ]
            )

            with (
                patch("fastloadai.cli.detect_hardware", return_value=specs),
                patch("fastloadai.cli.run_chat", return_value=0) as run_chat,
            ):
                self.assertEqual(_run_command(args), 0)

            self.assertEqual(run_chat.call_args.args[0], model_path.resolve())
            self.assertEqual(run_chat.call_args.args[2], None)
            updated_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_config["context_size"], 4096)
            self.assertEqual(updated_config["model_repo"], "test-org/test-model-GGUF")

    def test_chat_streams_openai_compatible_response(self):
        from fastloadai.chat import stream_chat

        class FakeResponse:
            lines = [
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"!"}}]}\n',
                b"data: [DONE]\n",
            ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                return iter(self.lines)

        with patch("fastloadai.chat.urlopen", return_value=FakeResponse()):
            chunks = list(stream_chat("http://127.0.0.1:8080", [{"role": "user", "content": "Hello"}]))

        self.assertEqual(chunks, ["Hello", "!"])

    def test_plan_command_generates_config_for_model(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=8)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"gguf")
            output_path = root / "config.json"
            args = argparse.Namespace(model=str(model_path), output=output_path)

            with patch("fastloadai.cli.detect_hardware", return_value=specs):
                self.assertEqual(_plan_command(args), 0)

            config = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(config["model_path"], str(model_path.resolve()))


    def test_quant_label_extracts_short_name(self):
        self.assertEqual(_quant_label("qwen2.5-0.5b-instruct-q8_0.gguf"), "Q8_0")
        self.assertEqual(_quant_label("model-IQ4_XS.gguf"), "IQ4_XS")

    def test_models_command_groups_shards_and_reports_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "small-Q4_K_M.gguf").write_bytes(b"x" * 2048)
            (root / "big-00001-of-00002.gguf").write_bytes(b"y" * 1024)
            (root / "big-00002-of-00002.gguf").write_bytes(b"y" * 1024)
            args = argparse.Namespace(models_dir=root, json=True)

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(_models_command(args), 0)

            payload = json.loads(output.getvalue())
            entries = {entry["name"]: entry for entry in payload}
            self.assertEqual(entries["big.gguf"]["size_bytes"], 2048)
            self.assertEqual(entries["small-Q4_K_M.gguf"]["quantization"], "Q4_K_M")

    def test_doctor_reports_missing_server(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"model_path": str(config)}), encoding="utf-8")
            args = argparse.Namespace(config=config)

            with patch("fastloadai.cli.shutil.which", return_value=None):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(_doctor_command(args), 1)

            self.assertIn("missing", output.getvalue())

    def test_trim_history_drops_old_pairs_but_keeps_recent(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
            {"role": "user", "content": "recent"},
        ]

        trim_history(messages, 50)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["content"], "recent")


if __name__ == "__main__":
    unittest.main()