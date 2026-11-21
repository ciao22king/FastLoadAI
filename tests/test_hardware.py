import argparse
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from fastloadai.catalog import catalog_match, catalog_rows
from fastloadai.chat import trim_history
from fastloadai.cli import (
    _doctor_command,
    _download_command,
    _hf_token,
    _is_repo_reference,
    _models_command,
    _plan_command,
    _quant_label,
    _run_command,
    _select_gguf,
    _suggest_command,
    build_parser,
)
from fastloadai.hardware import (
    HardwareSpecs,
    _cgroup_memory_limit_gb,
    _cpu_count_from_quota,
    _macos_available_ram_gb,
    model_fit,
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

    def test_run_without_a_model_explains_how_to_start(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                [
                    "run",
                    "--config",
                    str(root / "fastloadai-config.json"),
                    "--models-dir",
                    str(root / "models"),
                ]
            )
            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(_run_command(args), 2)

        self.assertIn("Pass a model name", errors.getvalue())

    def test_run_with_catalog_name_auto_downloads_that_model(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models_dir = root / "models"
            config_path = root / "fastloadai-config.json"
            model_path = models_dir / "qwen-Q4_K_M.gguf"
            args = build_parser().parse_args(
                [
                    "run",
                    "qwen2.5-0.5b",
                    "--config",
                    str(config_path),
                    "--models-dir",
                    str(models_dir),
                    "--once",
                    "--prompt",
                    "hi",
                ]
            )

            def fake_download(run_args, repo):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"gguf")
                config_path.write_text(
                    json.dumps(
                        {
                            "model_path": str(model_path),
                            "model_repo": repo,
                            "model_size_bytes": 4,
                        }
                    ),
                    encoding="utf-8",
                )
                return 0

            with (
                patch("fastloadai.cli.detect_hardware", return_value=specs),
                patch("fastloadai.cli._download_for_run", side_effect=fake_download) as download,
                patch("fastloadai.cli.shutil.which", return_value="/usr/bin/llama-server"),
                patch("fastloadai.cli.run_chat", return_value=0) as run_chat,
            ):
                self.assertEqual(_run_command(args), 0)

            self.assertEqual(
                download.call_args.args[1], "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
            )
            self.assertEqual(run_chat.call_args.args[0], model_path.resolve())

    def test_catalog_name_matching_is_fuzzy_but_unambiguous(self):
        self.assertEqual(
            catalog_match("qwen2.5-0.5b").repo,
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        )
        self.assertEqual(
            catalog_match("SmolLM2 135M").repo,
            "unsloth/SmolLM2-135M-Instruct-GGUF",
        )
        self.assertIsNone(catalog_match("qwen"))

    def test_run_with_repo_id_auto_downloads_that_repository(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=8, available_ram_gb=6)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models_dir = root / "models"
            config_path = root / "fastloadai-config.json"
            model_path = models_dir / "repo-Q4_K_M.gguf"
            args = build_parser().parse_args(
                [
                    "run",
                    "org/model-GGUF",
                    "--config",
                    str(config_path),
                    "--models-dir",
                    str(models_dir),
                    "--once",
                    "--prompt",
                    "hi",
                ]
            )

            def fake_download(run_args, repo):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(b"gguf")
                config_path.write_text(
                    json.dumps(
                        {
                            "model_path": str(model_path),
                            "model_repo": repo,
                            "model_size_bytes": 4,
                        }
                    ),
                    encoding="utf-8",
                )
                return 0

            with (
                patch("fastloadai.cli.detect_hardware", return_value=specs),
                patch("fastloadai.cli._download_for_run", side_effect=fake_download) as download,
                patch("fastloadai.cli.shutil.which", return_value="/usr/bin/llama-server"),
                patch("fastloadai.cli.run_chat", return_value=0),
            ):
                self.assertEqual(_run_command(args), 0)

            self.assertEqual(download.call_args.args[1], "org/model-GGUF")

    def test_repo_references_are_distinguished_from_local_models(self):
        self.assertTrue(_is_repo_reference("org/model-GGUF"))
        self.assertFalse(_is_repo_reference("model-Q4_K_M.gguf"))
        self.assertFalse(_is_repo_reference("test-model"))

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

    def test_model_fit_uses_available_memory(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=8, available_ram_gb=3.0)

        self.assertEqual(model_fit(specs, 0.5), "comfortable")
        self.assertEqual(model_fit(specs, 1.1), "tight")
        self.assertEqual(model_fit(specs, 2.0), "too_large")
        self.assertEqual(model_fit(specs, 0.0), "unknown")

    def test_cgroup_memory_limit_is_parsed(self):
        def fake_read(path):
            return "2147483648" if str(path).endswith("memory.max") else ""

        with patch("fastloadai.hardware._read_text", side_effect=fake_read):
            self.assertAlmostEqual(_cgroup_memory_limit_gb(), 2.0)

    def test_cgroup_cpu_quota_is_rounded_to_threads(self):
        self.assertEqual(_cpu_count_from_quota(200000, 100000), 2)
        self.assertEqual(_cpu_count_from_quota(50000, 100000), 1)
        self.assertEqual(_cpu_count_from_quota(-1, 100000), 0)

    def test_macos_available_memory_uses_vm_stat(self):
        def fake_run(command, **kwargs):
            if command[0] == "sysctl":
                return SimpleNamespace(stdout="17179869184\n")
            return SimpleNamespace(
                stdout=(
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                    "Pages free: 100000.\n"
                    "Pages inactive: 50000.\n"
                    "Pages speculative: 25000.\n"
                )
            )

        with patch("fastloadai.hardware.subprocess.run", side_effect=fake_run):
            available = _macos_available_ram_gb()

        self.assertAlmostEqual(available, 175000 * 16384 / (1024**3))

    def test_apple_unified_memory_offloads_only_when_model_fits(self):
        specs = HardwareSpecs(
            cpu_threads=8,
            ram_gb=16,
            gpu_name="Apple Silicon (Metal)",
            gpu_vram_gb=11.2,
            available_ram_gb=8.0,
            unified_memory=True,
        )

        self.assertEqual(
            recommend_settings(specs, model_size_bytes=1024**3)["n_gpu_layers"], 99
        )
        self.assertEqual(
            recommend_settings(specs, model_size_bytes=20 * 1024**3)["n_gpu_layers"], 0
        )

    def test_low_memory_mode_reduces_context_and_batches(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        settings = recommend_settings(specs, model_size_bytes=2 * 1024**3)

        self.assertTrue(settings["low_memory"])
        self.assertEqual(settings["model_fit"], "too_large")
        self.assertEqual(settings["context_size"], 512)

    def test_catalog_recommends_a_model_that_fits(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        rows = catalog_rows(specs)

        recommended = [row for row in rows if row["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["fit"], "comfortable")
        self.assertLessEqual(float(recommended[0]["size_gb"]), 1.0)

    def test_two_gb_laptop_gets_a_small_recommendation(self):
        specs = HardwareSpecs(cpu_threads=2, ram_gb=2, available_ram_gb=1.8)
        rows = catalog_rows(specs)

        recommended = [row for row in rows if row["recommended"]]
        self.assertEqual(len(recommended), 1)
        self.assertEqual(recommended[0]["repo"], "Qwen/Qwen2.5-0.5B-Instruct-GGUF")

    def test_hf_token_falls_back_to_environment(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=True):
            self.assertEqual(_hf_token(argparse.Namespace(token=None)), "hf_test")

    def test_parser_accepts_suggest_and_download_controls(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["suggest"]).command, "suggest")
        download = parser.parse_args(
            ["download", "org/model-GGUF", "--dry-run", "--force", "--revision", "main"]
        )
        self.assertTrue(download.dry_run)
        self.assertTrue(download.force)
        self.assertEqual(download.revision, "main")
        self.assertTrue(parser.parse_args(["plan", "--low-memory"]).low_memory)

    def test_download_dry_run_reports_fit_without_downloading(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        siblings = [SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=100 * 1024**2)]
        api = SimpleNamespace(
            model_info=lambda repo, files_metadata=True, revision=None: SimpleNamespace(
                siblings=siblings
            )
        )
        args = argparse.Namespace(
            repo="org/model-GGUF",
            file=None,
            directory=Path("/tmp/unused"),
            output=Path("/tmp/unused/config.json"),
            token=None,
            revision=None,
            dry_run=True,
            force=False,
            prefer="balanced",
        )

        output = io.StringIO()
        with patch("fastloadai.cli.detect_hardware", return_value=specs), patch(
            "huggingface_hub.HfApi", return_value=api
        ), patch("huggingface_hub.hf_hub_download") as download, redirect_stdout(output):
            self.assertEqual(_download_command(args), 0)

        download.assert_not_called()
        self.assertIn("comfortable", output.getvalue())

    def test_download_refuses_too_large_model_without_force(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        siblings = [SimpleNamespace(rfilename="model-Q4_K_M.gguf", size=20 * 1024**3)]
        api = SimpleNamespace(
            model_info=lambda repo, files_metadata=True, revision=None: SimpleNamespace(
                siblings=siblings
            )
        )
        args = argparse.Namespace(
            repo="org/model-GGUF",
            file=None,
            directory=Path("/tmp/unused"),
            output=Path("/tmp/unused/config.json"),
            token=None,
            revision=None,
            dry_run=False,
            force=False,
            prefer="balanced",
        )

        output = io.StringIO()
        errors = io.StringIO()
        with patch("fastloadai.cli.detect_hardware", return_value=specs), patch(
            "huggingface_hub.HfApi", return_value=api
        ), patch("huggingface_hub.hf_hub_download") as download, redirect_stdout(
            output
        ), redirect_stderr(errors):
            self.assertEqual(_download_command(args), 1)

        download.assert_not_called()
        self.assertIn("too_large", output.getvalue())
        self.assertIn("No quantization fits", errors.getvalue())

    def test_suggest_json_marks_one_recommendation(self):
        specs = HardwareSpecs(cpu_threads=4, ram_gb=4, available_ram_gb=3.5)
        output = io.StringIO()
        with patch("fastloadai.cli.detect_hardware", return_value=specs), redirect_stdout(
            output
        ):
            self.assertEqual(_suggest_command(argparse.Namespace(json=True)), 0)

        payload = json.loads(output.getvalue())
        self.assertEqual(sum(1 for row in payload if row["recommended"]), 1)


if __name__ == "__main__":
    unittest.main()