import argparse
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastloadai.cli import _plan_command, _run_command, _select_gguf, build_parser
from fastloadai.hardware import HardwareSpecs, recommend_settings, usable_model_memory_gb


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

        selected = _select_gguf(siblings, 6, None)

        self.assertEqual(selected["name"], "model-Q5_K_M.gguf")

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
                b'data: {"choices":[{"delta":{"content":"Ciao"}}]}\n',
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
            chunks = list(stream_chat("http://127.0.0.1:8080", [{"role": "user", "content": "Ciao"}]))

        self.assertEqual(chunks, ["Ciao", "!"])

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


if __name__ == "__main__":
    unittest.main()