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

func TestRenderBannerShowsNewOptimizationState(t *testing.T) {
	unified := true
	cache := 0
	swa := 8
	cfg := model.Config{
		ContextSize: 8192, CPUThreads: 4, Mmap: true,
		KVCacheTypeK: "q8_0", KVCacheTypeV: "f16", KVUnified: &unified,
		CacheRAMMiB: &cache, SlotSavePath: "/tmp/slots",
		RopeScaling: "yarn", RopeScale: 2, CPURange: "0-3",
		SWACheckpoints: &swa, NoRepack: true, NoWarmup: true,
	}
	out := RenderBanner("model.gguf", cfg, Theme{}, 180)
	for _, want := range []string{"KV q8_0/f16", "KV unified", "cache off", "slots", "rope yarn x2", "P-cores 0-3", "SWA 8", "no-repack", "no-warmup"} {
		if !strings.Contains(out, want) {
			t.Fatalf("banner missing %q: %q", want, out)
		}
	}
}

func TestRenderBannerKeepsAdvancedDetailsOnNarrowTerminal(t *testing.T) {
	unified := true
	cache := 0
	cfg := model.Config{KVCacheTypeK: "q8_0", KVUnified: &unified, CacheRAMMiB: &cache}
	out := RenderBanner("model.gguf", cfg, Theme{}, 80)
	if !strings.Contains(out, "KV unified") || !strings.Contains(out, "cache off") {
		t.Fatalf("advanced details missing: %q", out)
	}
}

func TestRenderHardwareShowsHybridCPU(t *testing.T) {
	out := RenderHardware(model.Specs{CPUThreads: 12, Hybrid: true, PerformanceThreads: 8, EfficiencyThreads: 4, PerformanceCPUs: "0-7"}, Theme{})
	if !strings.Contains(out, "Hybrid CPU") || !strings.Contains(out, "8 performance / 4 efficiency") {
		t.Fatalf("unexpected hardware: %q", out)
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

func TestFitDetailsDropsTrailing(t *testing.T) {
	parts := []string{"profile balanced", "Q4_K_M", "ctx 8192", "4 thr", "CPU", "mmap", "KV q8_0", "469 MB"}
	got := fitDetails(parts, 30)
	if VisibleWidth(got) > 30 {
		t.Fatalf("fitDetails exceeded width: %q", got)
	}
	if !strings.Contains(got, "profile balanced") {
		t.Fatalf("fitDetails dropped the leading detail: %q", got)
	}
	if strings.Contains(got, "469 MB") {
		t.Fatalf("fitDetails kept a trailing detail that did not fit: %q", got)
	}
}
