package tui

import (
	"strings"
	"testing"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestVisibleWidthIgnoresANSI(t *testing.T) {
	plain := Theme{}
	painted := plain.Paint("hello", Cyan, Bold)
	if got := VisibleWidth(painted); got != 5 {
		t.Fatalf("width = %d, want 5", got)
	}
}

func TestFormatSize(t *testing.T) {
	if got := FormatSize(0.5); got != "512 MB" {
		t.Fatalf("FormatSize(0.5) = %q", got)
	}
	if got := FormatSize(2); got != "2.0 GB" {
		t.Fatalf("FormatSize(2) = %q", got)
	}
}

func TestRenderKVAndBanner(t *testing.T) {
	theme := Theme{}
	kv := RenderKV("Test", [][2]string{{"a", "1"}, {"long", "2"}}, theme, "  ")
	if !strings.Contains(kv, "a     1") || !strings.Contains(kv, "long  2") {
		t.Fatalf("unexpected kv: %q", kv)
	}
	banner := RenderBanner("model.gguf", model.Config{ContextSize: 2048, CPUThreads: 4, Mmap: true}, theme, 48)
	if !strings.Contains(banner, "FastLoadAI") || !strings.Contains(banner, "model.gguf") || !strings.Contains(banner, "ctx 2048") {
		t.Fatalf("unexpected banner: %q", banner)
	}
}

func TestRenderSuggestions(t *testing.T) {
	rows := []SuggestionRow{{Name: "Small", Repo: "org/small", SizeGB: 0.5, Note: "fast", Fit: "comfortable", Recommended: true}}
	out := RenderSuggestions(rows, 4, Theme{})
	if !strings.Contains(out, "Small") || !strings.Contains(out, "comfortable") || !strings.Contains(out, "*") {
		t.Fatalf("unexpected suggestions: %q", out)
	}
}

func TestSpinnerWithoutTerminal(t *testing.T) {
	spinner := StartSpinner("loading", Theme{})
	spinner.Stop()
	spinner.Stop()
}

func TestRenderHardwareIncludesPlatformAndCPU(t *testing.T) {
	out := RenderHardware(model.Specs{CPUName: "Test CPU", CPUThreads: 4, RAMGB: 8}, Theme{})
	if !strings.Contains(out, "Platform") || !strings.Contains(out, "Test CPU") {
		t.Fatalf("unexpected hardware: %q", out)
	}
}

func TestRenderSummary(t *testing.T) {
	stats := ChatStats{Turns: 2, PromptTokens: 10, GeneratedTokens: 20}
	stats.Elapsed = 2 * time.Second
	out := stats.RenderSummary(Theme{})
	if !strings.Contains(out, "Session summary") || !strings.Contains(out, "10 in / ~20 out") {
		t.Fatalf("unexpected summary: %q", out)
	}
	if (ChatStats{}).RenderSummary(Theme{}) != "" {
		t.Fatal("empty stats should render nothing")
	}
}
