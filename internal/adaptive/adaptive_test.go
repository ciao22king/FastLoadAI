package adaptive

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

type ggufBuilder struct {
	bytes.Buffer
}

func (b *ggufBuilder) u32(value uint32) {
	_ = binary.Write(&b.Buffer, binary.LittleEndian, value)
}

func (b *ggufBuilder) u64(value uint64) {
	_ = binary.Write(&b.Buffer, binary.LittleEndian, value)
}

func (b *ggufBuilder) str(value string) {
	b.u64(uint64(len(value)))
	b.WriteString(value)
}

func (b *ggufBuilder) key(key string, valueType uint32) {
	b.str(key)
	b.u32(valueType)
}

func testGGUF(t *testing.T) string {
	t.Helper()
	var metadata ggufBuilder
	metadata.key("general.architecture", ggufString)
	metadata.str("qwen2")
	metadata.key("general.name", ggufString)
	metadata.str("test model")
	metadata.key("qwen2.block_count", ggufUint32)
	metadata.u32(4)
	metadata.key("qwen2.embedding_length", ggufUint32)
	metadata.u32(64)
	metadata.key("qwen2.attention.head_count", ggufUint32)
	metadata.u32(8)
	metadata.key("qwen2.attention.head_count_kv", ggufUint32)
	metadata.u32(2)
	metadata.key("qwen2.attention.key_length", ggufUint32)
	metadata.u32(8)
	metadata.key("qwen2.attention.value_length", ggufUint32)
	metadata.u32(8)
	metadata.key("qwen2.context_length", ggufUint32)
	metadata.u32(2048)
	metadata.key("qwen2.expert_count", ggufUint32)
	metadata.u32(8)
	metadata.key("qwen2.expert_used_count", ggufUint32)
	metadata.u32(2)
	metadata.key("tokenizer.ggml.tokens", ggufArray)
	metadata.u32(ggufString)
	metadata.u64(3)
	metadata.str("a")
	metadata.str("b")
	metadata.str("c")

	var tensors ggufBuilder
	tensors.str("token_embd.weight")
	tensors.u32(2)
	tensors.u64(64)
	tensors.u64(64)
	tensors.u32(0)
	tensors.u64(0)
	tensors.str("output.weight")
	tensors.u32(2)
	tensors.u64(64)
	tensors.u64(8)
	tensors.u32(0)
	tensors.u64(0)

	var file ggufBuilder
	file.WriteString("GGUF")
	file.u32(3)
	file.u64(2)
	file.u64(12)
	file.Write(metadata.Bytes())
	file.Write(tensors.Bytes())
	path := filepath.Join(t.TempDir(), "test-Q4_K_M.gguf")
	if err := os.WriteFile(path, file.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadGGUF(t *testing.T) {
	profile, err := ReadGGUF(testGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	if profile.Architecture != "qwen2" || profile.Name != "test model" {
		t.Fatalf("unexpected identity: %+v", profile)
	}
	if profile.Parameters != 4608 {
		t.Fatalf("parameters = %d, want 4608", profile.Parameters)
	}
	if profile.Layers != 4 || profile.HiddenSize != 64 {
		t.Fatalf("shape = %+v", profile)
	}
	if profile.Heads != 8 || profile.KVHeads != 2 || profile.KeyLength != 8 {
		t.Fatalf("attention = %+v", profile)
	}
	if profile.Experts != 8 || profile.ExpertsUsed != 2 {
		t.Fatalf("experts = %+v", profile)
	}
	if profile.ContextLength != 2048 || profile.Source != "gguf" {
		t.Fatalf("context/source = %+v", profile)
	}
}

func TestLoadProfileFallsBackForNonGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model-Q4_K_M.gguf")
	if err := os.WriteFile(path, bytes.Repeat([]byte{0x42}, 4096), 0o644); err != nil {
		t.Fatal(err)
	}
	profile := LoadProfile(path)
	if profile.Source != "estimate" {
		t.Fatalf("source = %q, want estimate", profile.Source)
	}
	if profile.Parameters <= 0 || profile.Layers <= 0 || profile.Quantization != "Q4_K_M" {
		t.Fatalf("unexpected estimate: %+v", profile)
	}
}

func TestClassifyPrompt(t *testing.T) {
	cases := []struct {
		prompt string
		want   Effort
	}{
		{"hello", EffortLight},
		{"what is the capital of France?", EffortLight},
		{"Explain step by step how a B-tree split works and compare it with a hash table.", EffortDeep},
		{"Write a SQL query that groups orders by customer and shows the top ten.", EffortDeep},
		{"Summarize this paragraph in one sentence.", EffortLight},
		{"", EffortStandard},
	}
	for _, test := range cases {
		if got := ClassifyPrompt(test.prompt); got != test.want {
			t.Errorf("ClassifyPrompt(%q) = %q, want %q", test.prompt, got, test.want)
		}
	}
}

func denseProfile() ModelProfile {
	return ModelProfile{
		Parameters:    7_000_000_000,
		Layers:        32,
		HiddenSize:    4096,
		Heads:         32,
		KVHeads:       8,
		KeyLength:     128,
		ValueLength:   128,
		ContextLength: 32768,
		FileBytes:     4_700_000_000,
		Quantization:  "Q4_K_M",
	}
}

func TestPlanStreamsWhenModelExceedsBudget(t *testing.T) {
	specs := model.Specs{RAMGB: 8, AvailableRAMGB: 6, CPUThreads: 4}
	plan := Plan(denseProfile(), specs, EffortStandard, 0)
	if !plan.Streaming || plan.Risk != "streaming" {
		t.Fatalf("expected streaming plan, got %+v", plan)
	}
	if plan.ResidentParameters <= 0 || plan.ResidentParameters >= plan.TotalParameters {
		t.Fatalf("resident params = %d", plan.ResidentParameters)
	}
	if plan.ContextSize < 256 || plan.BatchSize != 128 {
		t.Fatalf("streaming should shrink context/batch: %+v", plan)
	}
	if plan.LoadMode != "mmap" || plan.KVBytesPerToken <= 0 {
		t.Fatalf("unexpected plan: %+v", plan)
	}
}

func TestPlanUsesGPUWhenModelFits(t *testing.T) {
	specs := model.Specs{
		RAMGB: 32, AvailableRAMGB: 24, GPUVramGB: 12, GPUName: "test GPU",
		CPUThreads: 8,
	}
	plan := Plan(denseProfile(), specs, EffortDeep, 0)
	if plan.Streaming {
		t.Fatalf("model should fit: %+v", plan)
	}
	if plan.GPULayers == 0 {
		t.Fatalf("expected GPU layers: %+v", plan)
	}
	if plan.ContextSize > 8192 {
		t.Fatalf("context should be capped: %+v", plan)
	}
	if plan.Risk != "comfortable" && plan.Risk != "tight" {
		t.Fatalf("risk = %q", plan.Risk)
	}
}

func TestPlanMoEActiveParameters(t *testing.T) {
	profile := ModelProfile{
		Parameters:    46_700_000_000,
		Layers:        32,
		HiddenSize:    4096,
		Heads:         32,
		KVHeads:       8,
		KeyLength:     128,
		ValueLength:   128,
		Experts:       8,
		ExpertsUsed:   2,
		FileBytes:     26_000_000_000,
		Quantization:  "Q4_K_M",
		ContextLength: 32768,
	}
	specs := model.Specs{
		RAMGB: 16, AvailableRAMGB: 12, GPUVramGB: 8, GPUName: "test GPU",
		CPUThreads: 8,
	}
	plan := Plan(profile, specs, EffortStandard, 0)
	if plan.ActiveParameters >= plan.TotalParameters {
		t.Fatalf("MoE active params should be smaller: %+v", plan)
	}
	if plan.ActiveParameters <= 0 {
		t.Fatalf("active params missing: %+v", plan)
	}
	if !plan.CPUMoE && plan.NCPUMoE == 0 {
		t.Fatalf("expected expert offload: %+v", plan)
	}
}

func TestBudgetApply(t *testing.T) {
	cfg := model.Config{Hardware: model.Specs{GPUName: "test GPU"}}
	plan := Plan(denseProfile(), model.Specs{RAMGB: 8, AvailableRAMGB: 6}, EffortLight, 0)
	cfg = plan.Apply(cfg)
	if !cfg.Adaptive || cfg.Effort != "light" {
		t.Fatalf("adaptive fields missing: %+v", cfg)
	}
	if cfg.ContextSize != plan.ContextSize || cfg.NGPULayers != plan.GPULayers {
		t.Fatalf("plan not applied: %+v", cfg)
	}
	if !cfg.Streaming || !cfg.LowMemory {
		t.Fatalf("streaming should imply low memory: %+v", cfg)
	}
	if cfg.KVCacheTypeK != "q4_0" {
		t.Fatalf("light effort should use q4_0 KV: %+v", cfg)
	}
}

func TestSelectModelByEffort(t *testing.T) {
	candidates := []Candidate{
		{Name: "tiny", SizeGB: 0.11},
		{Name: "small", SizeGB: 0.46},
		{Name: "medium", SizeGB: 2.0, Local: true},
		{Name: "large", SizeGB: 4.7},
		{Name: "huge", SizeGB: 12},
	}
	light, ok := SelectModel(candidates, 9, EffortLight)
	if !ok || light.Name != "small" {
		t.Fatalf("light = %+v, %v", light, ok)
	}
	standard, ok := SelectModel(candidates, 9, EffortStandard)
	if !ok || standard.Name != "medium" {
		t.Fatalf("standard = %+v, %v", standard, ok)
	}
	deep, ok := SelectModel(candidates, 9, EffortDeep)
	if !ok || deep.Name != "large" {
		t.Fatalf("deep = %+v, %v", deep, ok)
	}
	if _, ok := SelectModel(candidates, 0.1, EffortDeep); ok {
		t.Fatal("nothing should fit a 0.1 GB budget")
	}
}

func TestApplyUsesTotalShardedSize(t *testing.T) {
	path := testGGUF(t)
	cfg := model.Config{ModelSizeBytes: 2_000_000_000, Hardware: model.Specs{RAMGB: 8, AvailableRAMGB: 6}}
	cfg = Apply(cfg, path, "", EffortStandard)
	if !cfg.Adaptive || cfg.ModelParameters < 2_000_000_000 {
		t.Fatalf("total size was not used: %+v", cfg)
	}
}

func TestClassifyPromptDoesNotMatchSubstrings(t *testing.T) {
	if got := ClassifyPrompt("This is a simple question about a planet, but please explain step by step."); got == EffortLight {
		t.Fatalf("substring markers should not force light: %q", got)
	}
}
