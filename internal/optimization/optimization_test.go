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
		SWACheckpoints: -1, CheckpointMinStep: -1,
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
		ContextSize: 4096, CPUThreads: 6, NGPULayers: 12,
	}
	cfg := Inherit(baseConfig(), saved)
	if cfg.OptimizationProfile != "memory" || cfg.FlashAttention != "on" || cfg.KVCacheTypeK != "q4_0" {
		t.Fatalf("inherit failed: %+v", cfg)
	}
	if cfg.SpecType != "ngram-simple" || cfg.CacheReuse != 64 {
		t.Fatalf("spec/cache inherit failed: %+v", cfg)
	}
	if cfg.ContextSize != 4096 || cfg.CPUThreads != 6 || cfg.NGPULayers != 12 {
		t.Fatalf("runtime sizing not inherited: %+v", cfg)
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

func TestSplitKVCacheAndPromptCache(t *testing.T) {
	cache := 0
	unified := true
	opts := autoOptions()
	opts.Profile = "balanced"
	opts.KVCacheType = "q8_0"
	opts.KVCacheTypeK = "q8_0"
	opts.KVCacheTypeV = "f16"
	opts.CacheRAMMiB = &cache
	opts.KVUnified = &unified
	opts.SlotSavePath = "/tmp/slots"
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.KVCacheTypeK != "q8_0" || cfg.KVCacheTypeV != "f16" {
		t.Fatalf("split KV cache = %s/%s", cfg.KVCacheTypeK, cfg.KVCacheTypeV)
	}
	if cfg.CacheRAMMiB == nil || *cfg.CacheRAMMiB != 0 {
		t.Fatalf("prompt cache RAM = %v, want 0", cfg.CacheRAMMiB)
	}
	if cfg.KVUnified == nil || !*cfg.KVUnified {
		t.Fatalf("unified KV = %v, want true", cfg.KVUnified)
	}
	if cfg.SlotSavePath != "/tmp/slots" {
		t.Fatalf("slot save path = %q", cfg.SlotSavePath)
	}
}

func TestMemoryProfileDisablesPromptCacheAndRepack(t *testing.T) {
	opts := autoOptions()
	opts.Profile = "memory"
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.CacheRAMMiB == nil || *cfg.CacheRAMMiB != 0 {
		t.Fatalf("memory profile prompt cache = %v, want 0", cfg.CacheRAMMiB)
	}
	if !cfg.NoRepack {
		t.Fatal("memory profile should disable weight repacking")
	}
}

func TestLowMemoryHintsApplyOutsideMemoryProfile(t *testing.T) {
	cfg := baseConfig()
	cfg.LowMemory = true
	opts := autoOptions()
	opts.Profile = "balanced"
	cfg = Apply(cfg, "/models/dense.gguf", "", opts)
	if cfg.CacheRAMMiB == nil || *cfg.CacheRAMMiB != 0 {
		t.Fatalf("low memory prompt cache = %v, want 0", cfg.CacheRAMMiB)
	}
	if !cfg.NoRepack {
		t.Fatal("low memory should disable weight repacking")
	}
}

func TestSpeedProfileSkipsWarmup(t *testing.T) {
	opts := autoOptions()
	opts.Profile = "speed"
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if !cfg.NoWarmup {
		t.Fatal("speed profile should skip warmup")
	}
}

func TestHybridCPUPinsPerformanceCores(t *testing.T) {
	cfg := baseConfig()
	cfg.CPUThreads = 16
	cfg.ThreadsBatch = 16
	cfg.Hardware.Hybrid = true
	cfg.Hardware.PerformanceMask = "ff"
	cfg.Hardware.PerformanceThreads = 8
	opts := autoOptions()
	opts.Profile = "balanced"
	cfg = Apply(cfg, "/models/dense.gguf", "", opts)
	if cfg.CPUMask != "ff" {
		t.Fatalf("CPU mask = %q, want ff", cfg.CPUMask)
	}
	if cfg.CPUThreads != 8 || cfg.ThreadsBatch != 8 {
		t.Fatalf("threads = %d/%d, want 8/8", cfg.CPUThreads, cfg.ThreadsBatch)
	}
}

func TestMemoryProfileDoesNotPinPerformanceCores(t *testing.T) {
	cfg := baseConfig()
	cfg.Hardware.Hybrid = true
	cfg.Hardware.PerformanceMask = "ff"
	cfg.Hardware.PerformanceThreads = 8
	opts := autoOptions()
	opts.Profile = "memory"
	cfg = Apply(cfg, "/models/dense.gguf", "", opts)
	if cfg.CPUMask != "" {
		t.Fatalf("memory profile should not pin CPUs: %+v", cfg)
	}
}

func TestRopeSWAAndAffinityOverrides(t *testing.T) {
	mask := "0f"
	noRepack := true
	noWarmup := true
	opts := autoOptions()
	opts.Profile = "balanced"
	opts.RopeScaling = "yarn"
	opts.RopeScale = 2
	opts.YarnOrigCtx = 4096
	opts.SWACheckpoints = 0
	opts.CheckpointMinStep = 0
	opts.CPUMask = mask
	opts.NoRepack = &noRepack
	opts.NoWarmup = &noWarmup
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	if cfg.RopeScaling != "yarn" || cfg.RopeScale != 2 || cfg.YarnOrigCtx != 4096 {
		t.Fatalf("rope overrides failed: %+v", cfg)
	}
	if cfg.SWACheckpoints == nil || *cfg.SWACheckpoints != 0 || cfg.CheckpointMinStep == nil || *cfg.CheckpointMinStep != 0 {
		t.Fatalf("checkpoint overrides failed: %+v", cfg)
	}
	if cfg.CPUMask != mask || !cfg.NoRepack || !cfg.NoWarmup {
		t.Fatalf("affinity/startup overrides failed: %+v", cfg)
	}
}

func TestInheritNewOptimizationFields(t *testing.T) {
	cache := 0
	unified := false
	noRepack := true
	noWarmup := true
	saved := model.Config{
		OptimizationProfile: "balanced", SlotSavePath: "/tmp/slots",
		RopeScaling: "linear", RopeScale: 4, YarnOrigCtx: 2048,
		SWACheckpoints: intPointer(0), CheckpointMinStep: intPointer(0), CPUMask: "0f",
		CacheRAMMiB: &cache, KVUnified: &unified, NoRepack: noRepack, NoWarmup: noWarmup,
	}
	cfg := Inherit(baseConfig(), saved)
	if cfg.SlotSavePath != "/tmp/slots" || cfg.RopeScaling != "linear" || cfg.RopeScale != 4 || cfg.YarnOrigCtx != 2048 {
		t.Fatalf("new string fields not inherited: %+v", cfg)
	}
	if cfg.SWACheckpoints == nil || *cfg.SWACheckpoints != 0 || cfg.CheckpointMinStep == nil || *cfg.CheckpointMinStep != 0 || cfg.CPUMask != "0f" {
		t.Fatalf("new int fields not inherited: %+v", cfg)
	}
	if cfg.CacheRAMMiB == nil || *cfg.CacheRAMMiB != 0 || cfg.KVUnified == nil || *cfg.KVUnified {
		t.Fatalf("new pointer fields not inherited: %+v", cfg)
	}
	if !cfg.NoRepack || !cfg.NoWarmup {
		t.Fatalf("new bool fields not inherited: %+v", cfg)
	}
}

func TestSummaryRowsIncludeNewControls(t *testing.T) {
	cache := 0
	unified := true
	opts := autoOptions()
	opts.Profile = "balanced"
	opts.CacheRAMMiB = &cache
	opts.KVUnified = &unified
	opts.SlotSavePath = "/tmp/slots"
	opts.SWACheckpoints = 0
	opts.CheckpointMinStep = 0
	opts.RopeScaling = "yarn"
	opts.RopeScale = 2
	opts.YarnOrigCtx = 4096
	opts.CPUMask = "0f"
	opts.NoRepack = boolPointer(true)
	opts.NoWarmup = boolPointer(true)
	cfg := Apply(baseConfig(), "/models/dense.gguf", "", opts)
	rows := SummaryRows(cfg)
	wanted := map[string]string{
		"KV buffer":          "unified",
		"Prompt cache RAM":   "off",
		"Slot save path":     "/tmp/slots",
		"SWA checkpoints":    "off",
		"Checkpoint spacing": "no minimum",
		"CPU affinity":       "mask 0f",
		"Weight repack":      "off",
		"Warmup":             "off",
	}
	found := map[string]bool{}
	for _, row := range rows {
		if want, ok := wanted[row[0]]; ok {
			found[row[0]] = true
			if row[1] != want {
				t.Fatalf("%s = %q, want %q", row[0], row[1], want)
			}
		}
	}
	for key := range wanted {
		if !found[key] {
			t.Fatalf("summary missing %q: %#v", key, rows)
		}
	}
}

func TestSummaryRowsShowsAdaptiveBudget(t *testing.T) {
	cfg := model.Config{
		Adaptive: true, Effort: "deep",
		ModelParameters: 30_000_000_000, ActiveParameters: 30_000_000_000,
		ParameterBudget: 13_600_000_000, ResidentParameters: 13_600_000_000,
		AdaptiveRisk: "streaming",
	}
	rows := SummaryRows(cfg)
	wanted := map[string]string{
		"Effort":     "deep",
		"Parameters": "30.0B total · 30.0B active · 13.6B budget",
		"Residency":  "13.6B resident · 16.4B streamed · streaming",
	}
	found := map[string]bool{}
	for _, row := range rows {
		if want, ok := wanted[row[0]]; ok {
			found[row[0]] = true
			if row[1] != want {
				t.Fatalf("%s = %q, want %q", row[0], row[1], want)
			}
		}
	}
	for key := range wanted {
		if !found[key] {
			t.Fatalf("summary missing %q: %#v", key, rows)
		}
	}
}

func boolPointer(value bool) *bool {
	return &value
}
