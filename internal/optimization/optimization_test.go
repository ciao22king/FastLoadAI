package optimization

import (
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func testSpecs() model.Specs {
	return model.Specs{
		CPUThreads: 16, PhysicalCores: 8, RAMGB: 32, AvailableRAMGB: 24,
	}
}

func baseConfig() model.Config {
	return model.Config{
		CPUThreads: 8, ContextSize: 8192, NGPULayers: 0, Mmap: true,
		ModelFit: "comfortable", ModelSizeGB: 4, UsableModelMemoryGB: 20,
		Hardware: testSpecs(),
	}
}

func autoOptions() Options {
	return Options{
		NCPUMoE: -1, MoECacheMiB: -1, MoECacheCPUShare: -1,
		CacheReuse: -1, Priority: -999, Poll: -1,
		DraftMax: -1, DraftMin: -1, DraftPMin: -1, DraftGPULayers: -1,
	}
}

func TestParseProfile(t *testing.T) {
	for _, value := range []string{"auto", "speed", "balanced", "memory", "quality", ""} {
		if _, err := ParseProfile(value); err != nil {
			t.Fatalf("ParseProfile(%q): %v", value, err)
		}
	}
	if _, err := ParseProfile("turbo"); err == nil {
		t.Fatal("expected an error for an unknown profile")
	}
}

func TestDetectMoE(t *testing.T) {
	tests := []struct {
		name string
		path string
		repo string
		want bool
	}{
		{"dense qwen", "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", false},
		{"mixtral", "/models/mixtral-8x7b-q4_k_m.gguf", "", true},
		{"qwen a3b", "/models/Qwen3-30B-A3B-Q4_K_M.gguf", "Qwen/Qwen3-30B-A3B-GGUF", true},
		{"llama 4 experts", "/models/Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf", "", true},
		{"deepseek v3", "/models/DeepSeek-V3-Q4_K_M.gguf", "", true},
		{"deepseek distill", "/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", "", false},
		{"gpt oss", "/models/gpt-oss-20b-Q4_K_M.gguf", "", true},
		{"olmoe", "/models/OLMoE-1B-7B-Q4_K_M.gguf", "", true},
		{"moe only in directory", "/models/moe-cache/qwen2.5-0.5b.gguf", "", false},
	}
	for _, item := range tests {
		t.Run(item.name, func(t *testing.T) {
			if got := DetectMoE(item.path, item.repo); got != item.want {
				t.Fatalf("DetectMoE = %v, want %v", got, item.want)
			}
		})
	}
}

func TestSpeedProfile(t *testing.T) {
	cfg := Apply(baseConfig(), "/models/qwen.gguf", "", autoOptions())
	if cfg.OptimizationProfile != "speed" {
		// Apply without an explicit profile auto-detects balanced.
		if cfg.OptimizationProfile != "balanced" {
			t.Fatalf("profile = %q", cfg.OptimizationProfile)
		}
	}
	opts := autoOptions()
	opts.Profile = "speed"
	cfg = Apply(baseConfig(), "/models/qwen.gguf", "", opts)
	if cfg.FlashAttention != "on" || cfg.BatchSize != 2048 || cfg.UBatchSize != 512 {
		t.Fatalf("unexpected speed config: %+v", cfg)
	}
	if cfg.ThreadsBatch != 8 || cfg.CacheReuse != 256 || !cfg.ContextShift {
		t.Fatalf("unexpected speed tuning: %+v", cfg)
	}
}

func TestMemoryProfile(t *testing.T) {
	opts := autoOptions()
	opts.Profile = "memory"
	cfg := Apply(baseConfig(), "/models/qwen.gguf", "", opts)
	if cfg.KVCacheTypeK != "q4_0" || cfg.KVCacheTypeV != "q4_0" {
		t.Fatalf("KV cache = %s/%s", cfg.KVCacheTypeK, cfg.KVCacheTypeV)
	}
	if cfg.BatchSize != 128 || cfg.UBatchSize != 128 || cfg.ThreadsBatch != 4 {
		t.Fatalf("unexpected memory tuning: %+v", cfg)
	}
	if cfg.ContextSize != 4096 {
		t.Fatalf("context = %d, want 4096", cfg.ContextSize)
	}
}

func TestBalancedProfileQuantizesLargeContext(t *testing.T) {
	opts := autoOptions()
	opts.Profile = "balanced"
	cfg := Apply(baseConfig(), "/models/qwen.gguf", "", opts)
	if cfg.KVCacheTypeK != "q8_0" || !cfg.ContextShift {
		t.Fatalf("unexpected balanced config: %+v", cfg)
	}
}

func TestMoEKeepsExpertsOnCPU(t *testing.T) {
	cfg := baseConfig()
	cfg.ModelSizeGB = 12
	cfg.ModelFit = "tight"
	cfg.UsableModelMemoryGB = 24
	cfg.Hardware.GPUName = "Test GPU"
	cfg.Hardware.GPUVramGB = 8
	opts := autoOptions()
	opts.Profile = "speed"
	cfg = Apply(cfg, "/models/Qwen3-30B-A3B-Q4_K_M.gguf", "Qwen/Qwen3-30B-A3B-GGUF", opts)
	if !cfg.MoE || !cfg.CPUMoE {
		t.Fatalf("MoE/CPU MoE not detected: %+v", cfg)
	}
	if cfg.NGPULayers != 99 {
		t.Fatalf("GPU layers = %d, want 99", cfg.NGPULayers)
	}
	if cfg.MoECacheMiB <= 0 || cfg.MoECacheCPUShare <= 0 {
		t.Fatalf("MoE cache not configured: %+v", cfg)
	}
}

func TestOverrides(t *testing.T) {
	lock := true
	contextShift := false
	opts := autoOptions()
	opts.Profile = "memory"
	opts.FlashAttention = "off"
	opts.KVCacheType = "f16"
	opts.NoMmap = true
	opts.Mlock = &lock
	opts.BatchSize = 64
	opts.UBatchSize = 32
	opts.Parallel = 2
	opts.NoContBatching = true
	opts.NCPUMoE = 4
	opts.Speculative = true
	opts.CacheReuse = 0
	opts.ContextShift = &contextShift
	cfg := Apply(baseConfig(), "/models/mixtral.gguf", "", opts)
	if cfg.FlashAttention != "off" || cfg.KVCacheTypeK != "f16" {
		t.Fatalf("flash/KV overrides failed: %+v", cfg)
	}
	if cfg.LoadMode != "none" || cfg.Mmap {
		t.Fatalf("load mode = %q mmap=%v", cfg.LoadMode, cfg.Mmap)
	}
	if cfg.BatchSize != 64 || cfg.UBatchSize != 32 || cfg.Parallel != 2 || !cfg.NoContBatching {
		t.Fatalf("batching overrides failed: %+v", cfg)
	}
	if cfg.CPUMoE || cfg.NCPUMoE != 4 {
		t.Fatalf("n-cpu-moe override failed: %+v", cfg)
	}
	if cfg.SpecType != "ngram-simple" {
		t.Fatalf("spec type = %q", cfg.SpecType)
	}
	if cfg.CacheReuse != 0 || cfg.ContextShift {
		t.Fatalf("cache/context overrides failed: %+v", cfg)
	}
}

func TestInherit(t *testing.T) {
	saved := model.Config{
		OptimizationProfile: "memory", FlashAttention: "on", KVCacheTypeK: "q4_0",
		LoadMode: "mmap", SpecType: "ngram-simple", CacheReuse: 64,
	}
	cfg := Inherit(baseConfig(), saved)
	if cfg.OptimizationProfile != "memory" || cfg.FlashAttention != "on" || cfg.KVCacheTypeK != "q4_0" {
		t.Fatalf("inherit failed: %+v", cfg)
	}
	if cfg.SpecType != "ngram-simple" || cfg.CacheReuse != 64 {
		t.Fatalf("spec/cache inherit failed: %+v", cfg)
	}
}

func TestSummaryRows(t *testing.T) {
	opts := autoOptions()
	opts.Profile = "speed"
	cfg := Apply(baseConfig(), "/models/Qwen3-30B-A3B-Q4_K_M.gguf", "", opts)
	rows := SummaryRows(cfg)
	found := false
	for _, row := range rows {
		if row[0] == "Model type" && row[1] == "Mixture of Experts (MoE)" {
			found = true
		}
	}
	if !found {
		t.Fatalf("summary missing MoE row: %#v", rows)
	}
}

func TestMoEFitsVRAMStaysOnGPU(t *testing.T) {
	cfg := baseConfig()
	cfg.ModelSizeGB = 4
	cfg.UsableModelMemoryGB = 24
	cfg.NGPULayers = 99
	cfg.Hardware.GPUName = "Test GPU"
	cfg.Hardware.GPUVramGB = 16
	opts := autoOptions()
	opts.Profile = "speed"
	cfg = Apply(cfg, "/models/Qwen3-30B-A3B-Q4_K_M.gguf", "", opts)
	if !cfg.MoE {
		t.Fatal("MoE not detected")
	}
	if cfg.CPUMoE {
		t.Fatalf("experts should stay on the GPU when the model fits VRAM: %+v", cfg)
	}
	if cfg.MoECacheMiB != 0 {
		t.Fatalf("MoE cache should be off when all experts are on the GPU: %+v", cfg)
	}
}

func TestNoMmapWinsOverNoMlock(t *testing.T) {
	opts := autoOptions()
	no := false
	opts.Mlock = &no
	opts.NoMmap = true
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.LoadMode != "none" || cfg.Mmap {
		t.Fatalf("--no-mmap should win: %+v", cfg)
	}
}

func TestNoMlockKeepsExplicitLoadMode(t *testing.T) {
	opts := autoOptions()
	no := false
	opts.Mlock = &no
	opts.LoadMode = "none"
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.LoadMode != "none" {
		t.Fatalf("--no-mlock should not override --load-mode none: %+v", cfg)
	}
}

func TestExplicitDraftMaxZero(t *testing.T) {
	opts := autoOptions()
	opts.DraftModel = "/models/draft.gguf"
	opts.DraftMax = 0
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.DraftMax != 0 {
		t.Fatalf("explicit --draft-max 0 was lost: %+v", cfg)
	}
}
