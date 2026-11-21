import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastloadai.chat import _is_memory_error, build_server_command
from fastloadai.hardware import HardwareSpecs
from fastloadai.tui import (
    ChatStats,
    Theme,
    color_enabled,
    conversation_tokens,
    format_duration,
    format_size,
    render_banner,
    render_context,
    render_hardware,
    render_help,
    render_kv,
    render_plan,
    render_timing,
    visible_width,
)


CONFIG = {
    "cpu_threads": 8,
    "context_size": 4096,
    "n_gpu_layers": 0,
    "mmap": True,
    "quantization": "Q4_K_M",
}


class ServerCommandTests(unittest.TestCase):
    def test_build_server_command_binds_localhost_and_passes_flags(self):
        command = build_server_command("llama-server", Path("/models/x.gguf"), CONFIG, 8123)
        self.assertEqual(command[:2], ["llama-server", "--model"])
        self.assertIn("127.0.0.1", command)
        self.assertIn("8123", command)
        self.assertIn("4096", command)
        self.assertNotIn("--no-mmap", command)

    def test_build_server_command_disables_mmap_when_requested(self):
        config = {**CONFIG, "mmap": False}
        command = build_server_command("llama-server", Path("/models/x.gguf"), config, 8123)
        self.assertIn("--no-mmap", command)

    def test_build_server_command_uses_small_batches_in_low_memory(self):
        config = {**CONFIG, "low_memory": True}
        command = build_server_command("llama-server", Path("/models/x.gguf"), config, 8123)
        self.assertIn("--batch-size", command)
        self.assertIn("128", command)
        self.assertIn("--ubatch-size", command)

    def test_memory_errors_are_recognized(self):
        self.assertTrue(_is_memory_error("llama_model_load: failed to allocate buffer"))
        self.assertTrue(_is_memory_error("std::bad_alloc"))
        self.assertFalse(_is_memory_error("invalid model file"))


class ThemeTests(unittest.TestCase):
    def test_disabled_theme_is_plain_text(self):
        self.assertEqual(Theme(False).paint("hello", "1", "32"), "hello")

    def test_enabled_theme_adds_ansi(self):
        painted = Theme(True).paint("hello", "32")
        self.assertIn("\033[32m", painted)
        self.assertTrue(painted.endswith("\033[0m"))
        self.assertEqual(visible_width(painted), 5)

    def test_no_color_env_disables_colour(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            self.assertFalse(color_enabled())

    def test_force_color_env_enables_colour(self):
        with patch.dict(os.environ, {"NO_COLOR": "", "FORCE_COLOR": "1"}, clear=False):
            self.assertTrue(color_enabled())


class RenderTests(unittest.TestCase):
    def test_banner_contains_model_and_settings(self):
        banner = render_banner("tiny.gguf", CONFIG, theme=Theme(False), width=60)
        self.assertIn("tiny.gguf", banner)
        self.assertIn("Q4_K_M", banner)
        self.assertIn("ctx 4096", banner)
        self.assertIn("CPU only", banner)

    def test_kv_aligns_labels(self):
        lines = render_kv("Title", [("a", "1"), ("long", "2")], Theme(False)).splitlines()
        self.assertEqual(lines[0], "Title")
        self.assertIn("  a     1", lines)
        self.assertIn("  long  2", lines)

    def test_help_lists_commands(self):
        text = render_help(Theme(False))
        for command in ("/help", "/clear", "/stats", "/context", "/retry", "/exit"):
            self.assertIn(command, text)

    def test_conversation_tokens_estimates_history(self):
        messages = [
            {"role": "system", "content": "x" * 8},
            {"role": "user", "content": "x" * 12},
        ]
        self.assertEqual(conversation_tokens(messages), 5)

    def test_context_bar_scales_with_usage(self):
        text = render_context(2048, 4096, Theme(False))
        self.assertIn("~2048 / 4096 tokens", text)
        self.assertIn("50%", text)
        self.assertIn("█" * 12, text)

    def test_context_handles_unknown_window(self):
        self.assertIn("unknown", render_context(10, 0, Theme(False)))

    def test_timing_colours_speed_when_enabled(self):
        text = render_timing(1.0, "x" * 40, Theme(True))
        self.assertIn("10 tok/s", text)
        self.assertIn("\033[32m", text)

    def test_hardware_marks_cpu_only(self):
        text = render_hardware(HardwareSpecs(cpu_threads=4, ram_gb=16), Theme(False))
        self.assertIn("none (CPU only)", text)

    def test_plan_shows_not_specified_without_model(self):
        text = render_plan({"context_size": 2048, "cpu_threads": 4}, "config.json", Theme(False))
        self.assertIn("not specified", text)
        self.assertIn("config.json", text)

    def test_format_helpers(self):
        self.assertEqual(format_duration(0.25), "250 ms")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_size(0), "unknown")
        self.assertEqual(format_size(0.5), "512 MB")
        self.assertEqual(format_size(2), "2.0 GB")


class StatsTests(unittest.TestCase):
    def test_stats_accumulate_turns_and_speed(self):
        stats = ChatStats()
        stats.record("x" * 40, 1.0)
        stats.record("x" * 40, 1.0)
        self.assertEqual(stats.turns, 2)
        self.assertEqual(stats.estimated_tokens, 20)
        self.assertAlmostEqual(stats.tokens_per_second, 10.0)
        self.assertIn("turns", stats.render(Theme(False)))


if __name__ == "__main__":
    unittest.main()
