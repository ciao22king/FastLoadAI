package llama

import (
	"strings"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestBuildCommand(t *testing.T) {
	args := BuildCommand("llama-server", "/models/a.gguf", model.Config{
		CPUThreads: 6, ContextSize: 4096, NGPULayers: 10, Mmap: false, LowMemory: true,
	}, 8123)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--model /models/a.gguf", "--port 8123", "--threads 6", "--ctx-size 4096",
		"--n-gpu-layers 10", "--load-mode none", "--batch-size 128",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q missing %q", joined, want)
		}
	}
}

func TestAvailablePort(t *testing.T) {
	port, err := AvailablePort()
	if err != nil || port <= 0 {
		t.Fatalf("port=%d err=%v", port, err)
	}
}

func TestBuildCommandOptimizationFlags(t *testing.T) {
	args := BuildCommand("llama-server", "/models/moe.gguf", model.Config{
		CPUThreads: 6, ContextSize: 8192, NGPULayers: 99,
		FlashAttention: "on", KVCacheTypeK: "q8_0", KVCacheTypeV: "q8_0",
		LoadMode: "mmap+mlock", NoKVOffload: true, NoOpOffload: true,
		NoContBatching: true, Parallel: 2, BatchSize: 256, UBatchSize: 128,
		ThreadsBatch: 3, CacheReuse: 128, ContextShift: true, SWAFull: true,
		NumA: "distribute", SplitMode: "row", TensorSplit: "3,1",
		CPUMoE: true, MoECacheMiB: 1024, MoECacheCPUShare: 0.5,
		OverrideTensor: "\\.ffn_.*_exps\\.=CPU", SpecType: "ngram-simple",
		DraftModel: "/models/draft.gguf", DraftMax: 4, DraftMin: 1, DraftPMin: 0.2, DraftGPULayers: 10,
		Priority: 1, Poll: 20,
	}, 8123)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--flash-attn on", "--cache-type-k q8_0", "--cache-type-v q8_0",
		"--load-mode mmap+mlock", "--no-kv-offload", "--no-op-offload",
		"--no-cont-batching", "--parallel 2", "--batch-size 256", "--ubatch-size 128",
		"--threads-batch 3", "--cache-reuse 128", "--context-shift", "--swa-full",
		"--numa distribute", "--split-mode row", "--tensor-split 3,1",
		"--cpu-moe", "--moe-cache-mib 1024", "--moe-cache-cpu-share 0.5",
		"--override-tensor \\.ffn_.*_exps\\.=CPU", "--spec-type ngram-simple",
		"--model-draft /models/draft.gguf", "--gpu-layers-draft 10",
		"--spec-draft-n-max 4", "--spec-draft-n-min 1", "--spec-draft-p-min 0.2",
		"--prio 1", "--poll 20",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q missing %q", joined, want)
		}
	}
}

func TestBuildCommandExtendedOptimizationFlags(t *testing.T) {
	unified := true
	cache := 0
	swaCheckpoints := 0
	checkpointMinStep := 0
	args := BuildCommand("llama-server", "/models/dense.gguf", model.Config{
		CPUThreads: 8, ContextSize: 8192, NGPULayers: 0,
		KVCacheTypeK: "q8_0", KVCacheTypeV: "q4_0",
		KVUnified: &unified, CacheRAMMiB: &cache, SlotSavePath: "/tmp/slots",
		SWACheckpoints: &swaCheckpoints, CheckpointMinStep: &checkpointMinStep,
		RopeScaling: "yarn", RopeScale: 2, YarnOrigCtx: 4096,
		CPUMask: "0f", NoRepack: true, NoWarmup: true,
	}, 8123)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--cache-type-k q8_0", "--cache-type-v q4_0", "--kv-unified",
		"--cache-ram 0", "--no-cache-idle-slots", "--slot-save-path /tmp/slots",
		"--swa-checkpoints 0", "--checkpoint-min-step 0",
		"--rope-scaling yarn", "--rope-scale 2", "--yarn-orig-ctx 4096",
		"--cpu-mask 0f", "--no-repack", "--no-warmup",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q missing %q", joined, want)
		}
	}
}

func TestBuildCommandNoKVUnifiedAndCPURange(t *testing.T) {
	unified := false
	args := BuildCommand("llama-server", "/models/dense.gguf", model.Config{
		CPUThreads: 8, ContextSize: 4096, KVUnified: &unified, CPURange: "0-7",
		CPUMask: "ff",
	}, 8123)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--no-kv-unified") {
		t.Errorf("args %q missing --no-kv-unified", joined)
	}
	if !strings.Contains(joined, "--cpu-range 0-7") || strings.Contains(joined, "--cpu-mask") {
		t.Errorf("args %q should prefer --cpu-range", joined)
	}
}
