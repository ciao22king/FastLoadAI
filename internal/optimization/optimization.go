// Package optimization selects llama.cpp runtime flags for local inference.
package optimization

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

// Profile is a high-level optimization goal.
type Profile string

const (
	ProfileAuto     Profile = "auto"
	ProfileSpeed    Profile = "speed"
	ProfileBalanced Profile = "balanced"
	ProfileMemory   Profile = "memory"
	ProfileQuality  Profile = "quality"
)

// Profiles lists every supported profile.
func Profiles() []Profile {
	return []Profile{ProfileAuto, ProfileSpeed, ProfileBalanced, ProfileMemory, ProfileQuality}
}

// ParseProfile validates a profile name.
func ParseProfile(value string) (Profile, error) {
	switch Profile(strings.ToLower(strings.TrimSpace(value))) {
	case "", ProfileAuto:
		return ProfileAuto, nil
	case ProfileSpeed:
		return ProfileSpeed, nil
	case ProfileBalanced:
		return ProfileBalanced, nil
	case ProfileMemory:
		return ProfileMemory, nil
	case ProfileQuality:
		return ProfileQuality, nil
	default:
		return "", fmt.Errorf("unknown optimization profile %q (use auto, speed, balanced, memory or quality)", value)
	}
}

// Options overrides the automatic choices. Negative sentinels mean "auto".
type Options struct {
	Profile          string
	FlashAttention   string
	KVCacheType      string
	NoKVOffload      bool
	NoOpOffload      bool
	LoadMode         string
	LazyMode         string
	NoMmap           bool
	Mlock            *bool
	BatchSize        int
	UBatchSize       int
	ThreadsBatch     int
	NoContBatching   bool
	Parallel         int
	NumA             string
	SplitMode        string
	TensorSplit      string
	CPUMoE           *bool
	NCPUMoE          int
	MoECacheMiB      int
	MoECacheCPUShare float64
	OverrideTensor   string
	Speculative      bool
	SpecType         string
	DraftModel       string
	DraftMax         int
	DraftMin         int
	DraftPMin        float64
	DraftGPULayers   int
	CacheReuse       int
	ContextShift     *bool
	SWAFull          bool
	Priority         int
	Poll             int
}

// Apply fills a runtime config with the best-known llama.cpp optimizations.
func Apply(cfg model.Config, modelPath, repo string, opts Options) model.Config {
	profile, err := ParseProfile(opts.Profile)
	if err != nil {
		profile = ProfileBalanced
	}
	if profile == ProfileAuto && cfg.OptimizationProfile != "" {
		if saved, err := ParseProfile(cfg.OptimizationProfile); err == nil {
			profile = saved
		}
	}
	if profile == ProfileAuto {
		profile = automaticProfile(cfg)
	}

	cfg.OptimizationProfile = string(profile)
	cfg.MoE = DetectMoE(modelPath, repo, cfg.ModelRepo, cfg.QuantizationFile)
	applyDefaults(&cfg)
	applyProfile(&cfg, profile)
	applyMoE(&cfg, profile)
	applySpeculative(&cfg, opts)
	applyOverrides(&cfg, opts)
	if !cfg.MoE || cfg.Hardware.GPUName == "" {
		// llama.cpp's MoE cache needs a MoE model and a GPU backend.
		cfg.MoECacheMiB = 0
		cfg.MoECacheCPUShare = 0
	}
	cfg.Mmap = usesMmap(cfg.LoadMode)
	cfg.LlamaCPPMoe = moeDescription(cfg)
	return cfg
}

func automaticProfile(cfg model.Config) Profile {
	if cfg.LowMemory || cfg.ModelFit == "too_large" {
		return ProfileMemory
	}
	return ProfileBalanced
}

func applyDefaults(cfg *model.Config) {
	cfg.FlashAttention = "auto"
	cfg.KVCacheTypeK = "f16"
	cfg.KVCacheTypeV = "f16"
	cfg.LoadMode = "mmap"
	if !cfg.Mmap {
		cfg.LoadMode = "none"
	}
	cfg.LazyMode = "auto"
	cfg.NoKVOffload = false
	cfg.NoOpOffload = false
	cfg.NoContBatching = false
	cfg.Parallel = 1
	cfg.BatchSize = 2048
	cfg.UBatchSize = 512
	cfg.ThreadsBatch = max(1, cfg.CPUThreads)
	cfg.NumA = ""
	cfg.SplitMode = ""
	cfg.TensorSplit = ""
	cfg.CPUMoE = false
	cfg.NCPUMoE = 0
	cfg.MoECacheMiB = 0
	cfg.MoECacheCPUShare = 0
	cfg.OverrideTensor = ""
	cfg.SpecType = ""
	cfg.DraftModel = ""
	cfg.DraftMax = -1
	cfg.DraftMin = -1
	cfg.DraftPMin = -1
	cfg.DraftGPULayers = -1
	cfg.CacheReuse = 256
	cfg.ContextShift = false
	cfg.SWAFull = false
	cfg.Priority = 0
	cfg.Poll = 50
}

func applyProfile(cfg *model.Config, profile Profile) {
	switch profile {
	case ProfileSpeed:
		cfg.FlashAttention = "on"
		cfg.BatchSize = 2048
		cfg.UBatchSize = 512
		cfg.ThreadsBatch = max(1, cfg.CPUThreads)
		cfg.CacheReuse = 256
		cfg.ContextShift = cfg.ContextSize >= 8192
	case ProfileMemory:
		cfg.FlashAttention = "on"
		cfg.KVCacheTypeK = "q4_0"
		cfg.KVCacheTypeV = "q4_0"
		cfg.BatchSize = 128
		cfg.UBatchSize = 128
		cfg.ThreadsBatch = max(1, cfg.CPUThreads/2)
		cfg.CacheReuse = 0
		cfg.ContextShift = true
		if cfg.ContextSize > 4096 {
			cfg.ContextSize = 4096
		}
		if cfg.Hardware.GPUName != "" {
			cfg.NoKVOffload = true
		}
	case ProfileQuality:
		cfg.FlashAttention = "auto"
		cfg.KVCacheTypeK = "f16"
		cfg.KVCacheTypeV = "f16"
		cfg.BatchSize = 2048
		cfg.UBatchSize = 512
		cfg.ThreadsBatch = max(1, cfg.CPUThreads)
		cfg.CacheReuse = 256
		cfg.ContextShift = false
	default:
		cfg.FlashAttention = "auto"
		if cfg.ContextSize >= 8192 || cfg.LowMemory {
			cfg.KVCacheTypeK = "q8_0"
			cfg.KVCacheTypeV = "q8_0"
		}
		cfg.BatchSize = 2048
		cfg.UBatchSize = 512
		cfg.ThreadsBatch = max(1, cfg.CPUThreads)
		cfg.CacheReuse = 256
		cfg.ContextShift = cfg.ContextSize >= 8192
	}
}

func applyMoE(cfg *model.Config, profile Profile) {
	if !cfg.MoE || cfg.Hardware.GPUName == "" {
		return
	}
	fitsVRAM := cfg.ModelSizeGB > 0 && cfg.ModelSizeGB <= cfg.Hardware.GPUVramGB*0.60
	if cfg.Hardware.UnifiedMemory {
		fitsVRAM = cfg.ModelSizeGB > 0 && cfg.ModelSizeGB <= cfg.UsableModelMemoryGB
	}
	if fitsVRAM {
		return
	}
	// A MoE model can keep attention on the GPU and experts in RAM even when
	// the full file does not fit in VRAM.
	if cfg.NGPULayers == 0 && cfg.ModelSizeGB > 0 && cfg.ModelSizeGB <= cfg.UsableModelMemoryGB {
		cfg.NGPULayers = 99
	}
	if cfg.NGPULayers == 0 {
		return
	}
	cfg.CPUMoE = true
	cache := 1024
	switch profile {
	case ProfileSpeed:
		cache = int(cfg.Hardware.GPUVramGB * 1024 * 0.25)
		cache = clamp(cache, 512, 4096)
	case ProfileMemory:
		cache = 0
	case ProfileQuality:
		cache = 2048
	}
	if cache > 0 {
		maxCache := int(cfg.Hardware.GPUVramGB * 1024 * 0.25)
		if maxCache < 256 {
			maxCache = 256
		}
		cache = clamp(cache, 0, maxCache)
		cfg.MoECacheMiB = cache
		if profile == ProfileSpeed {
			cfg.MoECacheCPUShare = 0.5
		}
	}
}

func applySpeculative(cfg *model.Config, opts Options) {
	if opts.DraftModel != "" {
		cfg.DraftModel = opts.DraftModel
		cfg.SpecType = "draft-simple"
		cfg.DraftMax = 3
		cfg.DraftMin = 0
		cfg.DraftPMin = 0
		cfg.DraftGPULayers = -1
	}
	if opts.Speculative && cfg.SpecType == "" {
		cfg.SpecType = "ngram-simple"
	}
}

func applyOverrides(cfg *model.Config, opts Options) {
	if opts.FlashAttention != "" {
		cfg.FlashAttention = opts.FlashAttention
	}
	if opts.KVCacheType != "" {
		cfg.KVCacheTypeK = opts.KVCacheType
		cfg.KVCacheTypeV = opts.KVCacheType
	}
	if opts.NoKVOffload {
		cfg.NoKVOffload = true
	}
	if opts.NoOpOffload {
		cfg.NoOpOffload = true
	}
	if opts.LoadMode != "" {
		cfg.LoadMode = opts.LoadMode
	}
	if opts.Mlock != nil {
		if *opts.Mlock {
			if cfg.LoadMode != "none" {
				cfg.LoadMode = "mmap+mlock"
			}
		} else if cfg.LoadMode == "mlock" || cfg.LoadMode == "mmap+mlock" {
			cfg.LoadMode = "mmap"
		}
	}
	if opts.NoMmap {
		cfg.LoadMode = "none"
	}
	if opts.LazyMode != "" {
		cfg.LazyMode = opts.LazyMode
	}
	if opts.BatchSize > 0 {
		cfg.BatchSize = opts.BatchSize
	}
	if opts.UBatchSize > 0 {
		cfg.UBatchSize = opts.UBatchSize
	}
	if opts.ThreadsBatch > 0 {
		cfg.ThreadsBatch = opts.ThreadsBatch
	}
	if opts.NoContBatching {
		cfg.NoContBatching = true
	}
	if opts.Parallel > 0 {
		cfg.Parallel = opts.Parallel
	}
	if opts.NumA != "" {
		cfg.NumA = opts.NumA
	}
	if opts.SplitMode != "" {
		cfg.SplitMode = opts.SplitMode
	}
	if opts.TensorSplit != "" {
		cfg.TensorSplit = opts.TensorSplit
	}
	if opts.CPUMoE != nil {
		cfg.CPUMoE = *opts.CPUMoE
		if *opts.CPUMoE {
			cfg.NCPUMoE = 0
		}
	}
	if opts.NCPUMoE >= 0 {
		cfg.NCPUMoE = opts.NCPUMoE
		if opts.NCPUMoE > 0 {
			cfg.CPUMoE = false
		}
	}
	if opts.MoECacheMiB >= 0 {
		cfg.MoECacheMiB = opts.MoECacheMiB
	}
	if opts.MoECacheCPUShare >= 0 {
		cfg.MoECacheCPUShare = opts.MoECacheCPUShare
	}
	if opts.OverrideTensor != "" {
		cfg.OverrideTensor = opts.OverrideTensor
	}
	if opts.SpecType != "" {
		cfg.SpecType = opts.SpecType
	}
	if opts.DraftModel != "" {
		cfg.DraftModel = opts.DraftModel
	}
	if opts.DraftMax >= 0 {
		cfg.DraftMax = opts.DraftMax
	}
	if opts.DraftMin >= 0 {
		cfg.DraftMin = opts.DraftMin
	}
	if opts.DraftPMin >= 0 {
		cfg.DraftPMin = opts.DraftPMin
	}
	if opts.DraftGPULayers >= 0 {
		cfg.DraftGPULayers = opts.DraftGPULayers
	}
	if opts.CacheReuse >= 0 {
		cfg.CacheReuse = opts.CacheReuse
	}
	if opts.ContextShift != nil {
		cfg.ContextShift = *opts.ContextShift
	}
	if opts.SWAFull {
		cfg.SWAFull = true
	}
	if opts.Priority > -999 {
		cfg.Priority = opts.Priority
	}
	if opts.Poll >= 0 {
		cfg.Poll = opts.Poll
	}
}

// Inherit copies saved optimization settings into a freshly recommended config.
func Inherit(dst model.Config, saved model.Config) model.Config {
	if saved.OptimizationProfile != "" {
		dst.OptimizationProfile = saved.OptimizationProfile
	}
	for _, value := range []struct {
		dst *string
		src string
	}{
		{&dst.FlashAttention, saved.FlashAttention},
		{&dst.KVCacheTypeK, saved.KVCacheTypeK},
		{&dst.KVCacheTypeV, saved.KVCacheTypeV},
		{&dst.LoadMode, saved.LoadMode},
		{&dst.LazyMode, saved.LazyMode},
		{&dst.NumA, saved.NumA},
		{&dst.SplitMode, saved.SplitMode},
		{&dst.TensorSplit, saved.TensorSplit},
		{&dst.OverrideTensor, saved.OverrideTensor},
		{&dst.SpecType, saved.SpecType},
		{&dst.DraftModel, saved.DraftModel},
	} {
		if value.src != "" {
			*value.dst = value.src
		}
	}
	for _, value := range []struct {
		dst *int
		src int
	}{
		{&dst.Parallel, saved.Parallel},
		{&dst.BatchSize, saved.BatchSize},
		{&dst.UBatchSize, saved.UBatchSize},
		{&dst.ThreadsBatch, saved.ThreadsBatch},
		{&dst.NCPUMoE, saved.NCPUMoE},
		{&dst.MoECacheMiB, saved.MoECacheMiB},
		{&dst.DraftMax, saved.DraftMax},
		{&dst.DraftMin, saved.DraftMin},
		{&dst.DraftGPULayers, saved.DraftGPULayers},
		{&dst.CacheReuse, saved.CacheReuse},
		{&dst.Priority, saved.Priority},
		{&dst.Poll, saved.Poll},
	} {
		if value.src != 0 {
			*value.dst = value.src
		}
	}
	if saved.MoECacheCPUShare > 0 {
		dst.MoECacheCPUShare = saved.MoECacheCPUShare
	}
	if saved.DraftModel != "" {
		dst.DraftMax = saved.DraftMax
		dst.DraftMin = saved.DraftMin
		dst.DraftPMin = saved.DraftPMin
		dst.DraftGPULayers = saved.DraftGPULayers
	}
	dst.NoKVOffload = dst.NoKVOffload || saved.NoKVOffload
	dst.NoOpOffload = dst.NoOpOffload || saved.NoOpOffload
	dst.NoContBatching = dst.NoContBatching || saved.NoContBatching
	dst.CPUMoE = dst.CPUMoE || saved.CPUMoE
	dst.ContextShift = dst.ContextShift || saved.ContextShift
	dst.SWAFull = dst.SWAFull || saved.SWAFull
	dst.MoE = dst.MoE || saved.MoE
	if saved.LoadMode == "" && !saved.Mmap {
		dst.LoadMode = "none"
	}
	if saved.LlamaCPPMoe != "" {
		dst.LlamaCPPMoe = saved.LlamaCPPMoe
	}
	dst.Mmap = usesMmap(dst.LoadMode)
	return dst
}

var (
	aExpertsRE = regexp.MustCompile(`(^|[-_./])a[0-9]+(\.[0-9]+)?b([-_./]|$)`)
	eExpertsRE = regexp.MustCompile(`(^|[-_./])[0-9]+e([-_./]|$)`)
)

// DetectMoE guesses whether a model is a Mixture of Experts model.
func DetectMoE(modelPath, repo string, extra ...string) bool {
	names := []string{repo}
	if modelPath != "" {
		names = append(names, filepath.Base(modelPath))
	}
	names = append(names, extra...)
	for _, name := range names {
		lower := strings.ToLower(name)
		if lower == "" {
			continue
		}
		if strings.Contains(lower, "moe") ||
			strings.Contains(lower, "mixtral") ||
			strings.Contains(lower, "gpt-oss") ||
			strings.Contains(lower, "gpt_oss") ||
			strings.Contains(lower, "grok") ||
			strings.Contains(lower, "dbrx") ||
			strings.Contains(lower, "jamba") ||
			strings.Contains(lower, "arctic") ||
			strings.Contains(lower, "deepseek-v2") ||
			strings.Contains(lower, "deepseek-v3") ||
			strings.Contains(lower, "deepseek_v2") ||
			strings.Contains(lower, "deepseek_v3") ||
			strings.Contains(lower, "deepseek2") ||
			strings.Contains(lower, "deepseek3") {
			return true
		}
		if aExpertsRE.MatchString(lower) || eExpertsRE.MatchString(lower) {
			return true
		}
	}
	return false
}

// SummaryRows returns human-readable optimization details for CLI output.
func SummaryRows(cfg model.Config) [][2]string {
	profile := cfg.OptimizationProfile
	if profile == "" {
		profile = string(ProfileAuto)
	}
	moe := cfg.MoE || DetectMoE(cfg.ModelPath, cfg.ModelRepo, cfg.QuantizationFile)
	modelType := "dense"
	if moe {
		modelType = "Mixture of Experts (MoE)"
	}
	kv := cfg.KVCacheTypeK
	if cfg.KVCacheTypeV != "" && cfg.KVCacheTypeV != cfg.KVCacheTypeK {
		kv += " / " + cfg.KVCacheTypeV
	}
	batching := "auto"
	if cfg.BatchSize > 0 || cfg.UBatchSize > 0 || cfg.Parallel > 0 {
		batching = fmt.Sprintf("batch %d, ubatch %d, parallel %d", cfg.BatchSize, cfg.UBatchSize, cfg.Parallel)
	}
	threadTuning := "auto"
	if cfg.ThreadsBatch > 0 {
		threadTuning = fmt.Sprintf("decode %d, batch %d", cfg.CPUThreads, cfg.ThreadsBatch)
	}
	rows := [][2]string{
		{"Profile", profile},
		{"Model type", modelType},
		{"Flash attention", fallback(cfg.FlashAttention, "auto")},
		{"KV cache", fallback(kv, "f16")},
		{"Batching", batching},
		{"Thread tuning", threadTuning},
		{"Load mode", fallback(cfg.LoadMode, "mmap")},
	}
	if cfg.NoContBatching {
		rows = append(rows, [2]string{"Continuous batching", "off"})
	}
	if cfg.CacheReuse > 0 {
		rows = append(rows, [2]string{"KV cache reuse", fmt.Sprintf("%d tokens", cfg.CacheReuse)})
	}
	if cfg.ContextShift {
		rows = append(rows, [2]string{"Context shift", "on"})
	}
	if cfg.SWAFull {
		rows = append(rows, [2]string{"SWA cache", "full"})
	}
	if moe {
		switch {
		case cfg.CPUMoE:
			rows = append(rows, [2]string{"MoE experts", "CPU (attention on GPU)"})
		case cfg.NCPUMoE > 0:
			rows = append(rows, [2]string{"MoE experts", fmt.Sprintf("first %d layers on CPU", cfg.NCPUMoE)})
		default:
			placement := "device / default"
			if cfg.NGPULayers == 0 {
				placement = "CPU only"
			}
			rows = append(rows, [2]string{"MoE experts", placement})
		}
		if cfg.MoECacheMiB > 0 {
			rows = append(rows, [2]string{"MoE cache", fmt.Sprintf("%d MiB, CPU share %.0f%%", cfg.MoECacheMiB, cfg.MoECacheCPUShare*100)})
		}
	}
	if cfg.OverrideTensor != "" {
		rows = append(rows, [2]string{"Tensor override", cfg.OverrideTensor})
	}
	if cfg.SpecType != "" {
		detail := cfg.SpecType
		if cfg.DraftModel != "" {
			detail += " · " + cfg.DraftModel
		}
		if cfg.DraftMax > 0 {
			detail += fmt.Sprintf(" · max %d", cfg.DraftMax)
		}
		rows = append(rows, [2]string{"Speculative decoding", detail})
	}
	if cfg.NumA != "" {
		rows = append(rows, [2]string{"NUMA", cfg.NumA})
	}
	if cfg.SplitMode != "" || cfg.TensorSplit != "" {
		detail := cfg.SplitMode
		if cfg.TensorSplit != "" {
			detail += " · " + cfg.TensorSplit
		}
		rows = append(rows, [2]string{"GPU split", detail})
	}
	if cfg.NoKVOffload {
		rows = append(rows, [2]string{"KV offload", "off"})
	}
	if cfg.NoOpOffload {
		rows = append(rows, [2]string{"Op offload", "off"})
	}
	if cfg.Priority != 0 {
		rows = append(rows, [2]string{"Priority", fmt.Sprintf("%d", cfg.Priority)})
	}
	return rows
}

func moeDescription(cfg model.Config) string {
	if !cfg.MoE {
		return ""
	}
	if cfg.CPUMoE {
		return "MoE detected: experts kept on CPU, attention offloaded to GPU."
	}
	if cfg.NCPUMoE > 0 {
		return fmt.Sprintf("MoE detected: experts of the first %d layers kept on CPU.", cfg.NCPUMoE)
	}
	return "MoE detected: using llama.cpp default expert placement."
}

func usesMmap(mode string) bool {
	return mode == "" || mode == "auto" || mode == "mmap" || mode == "mmap+mlock"
}

func fallback(value, def string) string {
	if value == "" {
		return def
	}
	return value
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

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
