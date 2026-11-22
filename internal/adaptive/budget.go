package adaptive

import (
	"fmt"
	"math"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

const gib = 1024 * 1024 * 1024

// Budget is the safe residency plan for one model, machine and prompt effort.
type Budget struct {
	Effort             Effort `json:"effort"`
	TotalParameters    int64  `json:"total_parameters"`
	ModelLayers        int    `json:"model_layers"`
	ActiveParameters   int64  `json:"active_parameters"`
	ParameterBudget    int64  `json:"parameter_budget"`
	ResidentParameters int64  `json:"resident_parameters"`
	StreamedParameters int64  `json:"streamed_parameters"`
	ModelBytes         int64  `json:"model_bytes"`
	ResidentBytes      int64  `json:"resident_bytes"`
	KVBytesPerToken    int64  `json:"kv_bytes_per_token"`
	ContextSize        int    `json:"context_size"`
	GPULayers          int    `json:"gpu_layers"`
	CPUMoE             bool   `json:"cpu_moe"`
	NCPUMoE            int    `json:"n_cpu_moe"`
	MoECacheMiB        int    `json:"moe_cache_mib"`
	BatchSize          int    `json:"batch_size"`
	UBatchSize         int    `json:"ubatch_size"`
	KVCacheType        string `json:"kv_cache_type"`
	LoadMode           string `json:"load_mode"`
	Streaming          bool   `json:"streaming"`
	Risk               string `json:"risk"`
	Reason             string `json:"reason"`
}

// Plan chooses the largest safe working set for a model on this machine.
//
// The planner treats RAM and VRAM as two pools. Weights that do not fit are
// left in the mmap page cache instead of being forced into anonymous memory;
// that is what lets a large GGUF start on a small machine without an OOM kill.
func Plan(profile ModelProfile, specs model.Specs, effort Effort, requestedContext int) Budget {
	effort = ResolveEffort(effort, "")
	totalParameters := profile.Parameters
	if totalParameters < 0 {
		totalParameters = 0
	}
	modelBytes := profile.FileBytes
	if modelBytes <= 0 {
		modelBytes = int64(float64(totalParameters) * bytesPerParameter(profile.Quantization))
	}
	availableRAM := specs.AvailableRAMGB
	if availableRAM <= 0 {
		availableRAM = specs.RAMGB
	}
	reserve := math.Max(0.75, availableRAM*0.20)
	if availableRAM < 2.5 {
		reserve = 0.50
	}
	ramBudget := math.Max(0, availableRAM-reserve) * 0.85
	vramBudget := 0.0
	if specs.GPUVramGB > 0 && !specs.UnifiedMemory {
		vramBudget = math.Max(0, specs.GPUVramGB-0.35) * 0.80
	}
	totalBudget := ramBudget + vramBudget
	if totalBudget <= 0 {
		totalBudget = math.Max(0.25, availableRAM*0.50)
	}

	modelGB := float64(modelBytes) / gib
	streaming := modelGB > totalBudget*0.95
	residentGB := modelGB
	if streaming {
		residentGB = totalBudget * 0.90
	}
	residentBytes := int64(residentGB * gib)
	if residentBytes > modelBytes {
		residentBytes = modelBytes
	}
	bytesPerParam := 0.65
	if totalParameters > 0 && modelBytes > 0 {
		bytesPerParam = float64(modelBytes) / float64(totalParameters)
	}
	parameterBudget := int64(totalBudget * 0.90 * gib / bytesPerParam)
	if totalParameters > 0 && parameterBudget > totalParameters {
		parameterBudget = totalParameters
	}
	residentParameters := totalParameters
	if streaming && modelGB > 0 {
		residentParameters = int64(float64(totalParameters) * residentGB / modelGB)
		if residentParameters > parameterBudget {
			residentParameters = parameterBudget
		}
	}
	activeParameters := totalParameters
	if profile.Experts > 0 && profile.ExpertsUsed > 0 {
		ratio := float64(profile.ExpertsUsed) / float64(profile.Experts)
		if ratio > 1 {
			ratio = 1
		}
		activeParameters = int64(float64(totalParameters) * (0.05 + 0.95*ratio))
		if activeParameters > totalParameters {
			activeParameters = totalParameters
		}
	}

	kvBytesPerToken := estimateKVBytesPerToken(profile, "q8_0")
	baseContext := contextForEffort(effort)
	if requestedContext > 0 && requestedContext < baseContext {
		baseContext = requestedContext
	}
	maxContext := baseContext
	if kvBytesPerToken > 0 {
		kvBudget := math.Min(totalBudget*0.20, 4.0)
		if streaming {
			kvBudget = math.Min(totalBudget*0.12, 1.5)
		}
		safeContext := int(kvBudget * gib / float64(kvBytesPerToken))
		if safeContext < maxContext {
			maxContext = safeContext
		}
	}
	if profile.ContextLength > 0 && maxContext > profile.ContextLength {
		maxContext = profile.ContextLength
	}
	maxContext = clamp(maxContext, 256, 131072)

	kvType := "q8_0"
	if effort == EffortLight || streaming {
		kvType = "q4_0"
	}
	kvBytesPerToken = estimateKVBytesPerToken(profile, kvType)

	gpuLayers := 0
	if specs.UnifiedMemory && profile.Layers > 0 && modelBytes > 0 && modelGB <= totalBudget*0.95 {
		gpuLayers = profile.Layers
	}
	if specs.GPUVramGB > 0 && !specs.UnifiedMemory && profile.Layers > 0 && modelBytes > 0 {
		kvGB := float64(kvBytesPerToken*int64(maxContext)) / gib
		vramForWeights := vramBudget - kvGB - 0.50
		if vramForWeights > 0 {
			layerBytes := float64(modelBytes) / float64(profile.Layers)
			gpuLayers = int(vramForWeights * gib / layerBytes)
			gpuLayers = clamp(gpuLayers, 0, profile.Layers)
		}
	}

	cpuMoE := false
	ncpuMoE := 0
	if profile.Experts > 0 && profile.ExpertsUsed > 0 && specs.GPUVramGB > 0 && !specs.UnifiedMemory {
		if gpuLayers == 0 {
			cpuMoE = true
		} else if modelGB > vramBudget {
			ncpuMoE = profile.Layers - gpuLayers
			if ncpuMoE < 1 {
				ncpuMoE = 1
			}
			if ncpuMoE > profile.Layers {
				ncpuMoE = profile.Layers
			}
		}
	}
	moeCache := 0
	if ncpuMoE > 0 || cpuMoE {
		moeCache = int(math.Min(specs.GPUVramGB*1024*0.20, 2048))
		if moeCache < 128 {
			moeCache = 128
		}
	}

	batchSize, ubatchSize := batchForEffort(effort)
	if streaming || availableRAM < 4 {
		batchSize, ubatchSize = 128, 128
	}
	risk := "comfortable"
	switch {
	case streaming:
		risk = "streaming"
	case modelGB > totalBudget*0.75:
		risk = "tight"
	}

	reason := fmt.Sprintf(
		"%s effort: %.1fB total / %.1fB active params, %.1f GB model in %.1f GB budget; %d/%d GPU layers, %s KV, ctx %d.",
		effort, float64(totalParameters)/1e9, float64(activeParameters)/1e9,
		modelGB, totalBudget, gpuLayers, profile.Layers, kvType, maxContext,
	)
	if streaming {
		reason = fmt.Sprintf(
			"%s effort: %.1f GB model does not fit in %.1f GB RAM+VRAM; mmap streaming keeps %.1fB of %.1fB params resident, ctx %d, %d/%d GPU layers.",
			effort, modelGB, totalBudget, float64(residentParameters)/1e9,
			float64(totalParameters)/1e9, maxContext, gpuLayers, profile.Layers,
		)
	}

	return Budget{
		Effort:             effort,
		TotalParameters:    totalParameters,
		ModelLayers:        profile.Layers,
		ActiveParameters:   activeParameters,
		ParameterBudget:    parameterBudget,
		ResidentParameters: residentParameters,
		StreamedParameters: maxInt64(0, totalParameters-residentParameters),
		ModelBytes:         modelBytes,
		ResidentBytes:      residentBytes,
		KVBytesPerToken:    kvBytesPerToken,
		ContextSize:        maxContext,
		GPULayers:          gpuLayers,
		CPUMoE:             cpuMoE,
		NCPUMoE:            ncpuMoE,
		MoECacheMiB:        moeCache,
		BatchSize:          batchSize,
		UBatchSize:         ubatchSize,
		KVCacheType:        kvType,
		LoadMode:           "mmap",
		Streaming:          streaming,
		Risk:               risk,
		Reason:             reason,
	}
}

// Apply writes a plan into a runtime config. Explicit CLI overrides are applied
// after this function, so a user can always pin a value manually.
func (b Budget) Apply(cfg model.Config) model.Config {
	cfg.Adaptive = true
	cfg.Effort = string(b.Effort)
	cfg.ModelParameters = b.TotalParameters
	cfg.ParameterBudget = b.ParameterBudget
	cfg.ActiveParameters = b.ActiveParameters
	cfg.ResidentParameters = b.ResidentParameters
	cfg.KVBytesPerToken = b.KVBytesPerToken
	cfg.Streaming = b.Streaming
	cfg.AdaptiveRisk = b.Risk
	cfg.AdaptiveReason = b.Reason
	cfg.ContextSize = b.ContextSize
	cfg.NGPULayers = b.GPULayers
	cfg.BatchSize = b.BatchSize
	cfg.UBatchSize = b.UBatchSize
	cfg.KVCacheTypeK = b.KVCacheType
	cfg.KVCacheTypeV = b.KVCacheType
	cfg.LoadMode = b.LoadMode
	cfg.Mmap = true
	cfg.CPUMoE = b.CPUMoE
	cfg.NCPUMoE = b.NCPUMoE
	cfg.MoECacheMiB = b.MoECacheMiB
	if b.CPUMoE || b.NCPUMoE > 0 {
		cfg.MoE = true
	}
	if b.Streaming {
		cfg.LowMemory = true
		cfg.CacheRAMMiB = intPointer(0)
		cfg.NoRepack = true
		cfg.NoKVOffload = cfg.Hardware.GPUName != ""
	}
	if b.ModelLayers > 0 {
		cfg.ModelLayers = b.ModelLayers
	}
	return cfg
}

func estimateKVBytesPerToken(profile ModelProfile, cacheType string) int64 {
	layers := profile.Layers
	kvHeads := profile.KVHeads
	if kvHeads <= 0 {
		kvHeads = profile.Heads
	}
	keyLength := profile.KeyLength
	if keyLength <= 0 && profile.Heads > 0 && profile.HiddenSize > 0 {
		keyLength = profile.HiddenSize / profile.Heads
	}
	valueLength := profile.ValueLength
	if valueLength <= 0 {
		valueLength = keyLength
	}
	if layers <= 0 || kvHeads <= 0 || keyLength <= 0 || valueLength <= 0 {
		return 0
	}
	bytes := kvBytes(cacheType)
	return int64(float64(layers*kvHeads*(keyLength+valueLength)) * bytes)
}

func kvBytes(cacheType string) float64 {
	switch cacheType {
	case "q4_0":
		return 0.5625
	case "q8_0":
		return 1.0625
	case "bf16", "f16":
		return 2
	default:
		return 2
	}
}

func contextForEffort(effort Effort) int {
	switch effort {
	case EffortLight:
		return 2048
	case EffortDeep:
		return 8192
	default:
		return 4096
	}
}

func batchForEffort(effort Effort) (int, int) {
	switch effort {
	case EffortLight:
		return 512, 128
	case EffortDeep:
		return 2048, 512
	default:
		return 2048, 512
	}
}

func clamp(value, low, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func intPointer(value int) *int {
	return &value
}

// Apply loads a model profile and writes a safe plan into cfg.
func Apply(cfg model.Config, modelPath, prompt string, effort Effort) model.Config {
	if modelPath == "" {
		return cfg
	}
	profile := LoadProfile(modelPath)
	if cfg.ModelSizeBytes > profile.FileBytes {
		// A sharded model has several files; the first shard is only a fraction
		// of the total size, so use the size the config already recorded.
		profile.FileBytes = cfg.ModelSizeBytes
		if estimated := estimateParameters(cfg.ModelSizeBytes, profile.Quantization); estimated > profile.Parameters {
			profile.Parameters = estimated
		}
	}
	resolved := ResolveEffort(effort, prompt)
	return Plan(profile, cfg.Hardware, resolved, cfg.ContextSize).Apply(cfg)
}
