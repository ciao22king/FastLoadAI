package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/adaptive"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
)

type optimizationFlags struct {
	set *flag.FlagSet

	profile           *string
	flashAttention    *string
	kvCache           *string
	kvCacheK          *string
	kvCacheV          *string
	kvUnified         *bool
	noKVUnified       *bool
	cacheRAM          *int
	slotSavePath      *string
	noKVOffload       *bool
	noOpOffload       *bool
	loadMode          *string
	lazyMode          *string
	mlock             *bool
	noMlock           *bool
	noMmap            *bool
	batchSize         *int
	ubatchSize        *int
	threadsBatch      *int
	noContBatching    *bool
	parallel          *int
	numa              *string
	splitMode         *string
	tensorSplit       *string
	cpuMoE            *bool
	noCPUMoE          *bool
	nCPUMoE           *int
	moeCacheMiB       *int
	moeCacheShare     *float64
	overrideTensor    *string
	speculative       *bool
	specType          *string
	draftModel        *string
	draftMax          *int
	draftMin          *int
	draftPMin         *float64
	draftGPULayers    *int
	cacheReuse        *int
	contextShift      *bool
	noContextShift    *bool
	swaFull           *bool
	swaCheckpoints    *int
	checkpointMinStep *int
	ropeScaling       *string
	ropeScale         *float64
	yarnOrigCtx       *int
	cpuMask           *string
	cpuRange          *string
	noRepack          *bool
	repack            *bool
	noWarmup          *bool
	warmup            *bool
	priority          *int
	poll              *int
	adaptive          *bool
	effort            *string
}

func registerOptimizationFlags(set *flag.FlagSet) *optimizationFlags {
	return &optimizationFlags{
		set:               set,
		profile:           set.String("profile", "", "optimization profile: auto, speed, balanced, memory or quality"),
		flashAttention:    set.String("flash-attn", "", "Flash Attention: on, off or auto"),
		kvCache:           set.String("kv-cache", "", "KV cache type for both K and V: f16, q8_0 or q4_0"),
		kvCacheK:          set.String("kv-cache-k", "", "KV cache type for K, for example q8_0"),
		kvCacheV:          set.String("kv-cache-v", "", "KV cache type for V, for example f16"),
		kvUnified:         set.Bool("kv-unified", false, "share one unified KV buffer across slots"),
		noKVUnified:       set.Bool("no-kv-unified", false, "use one KV buffer per slot"),
		cacheRAM:          set.Int("cache-ram", 8192, "prompt cache in MiB (-1 unlimited, 0 disabled)"),
		slotSavePath:      set.String("slot-save-path", "", "directory for saved slot KV cache"),
		noKVOffload:       set.Bool("no-kv-offload", false, "keep the KV cache on the CPU"),
		noOpOffload:       set.Bool("no-op-offload", false, "disable offloading host tensor operations"),
		loadMode:          set.String("load-mode", "", "model load mode: auto, none, mmap, mlock, mmap+mlock or dio"),
		lazyMode:          set.String("lazy-mode", "", "lazy tensor loading: on, auto or off"),
		mlock:             set.Bool("mlock", false, "lock the model in RAM (mmap+mlock)"),
		noMlock:           set.Bool("no-mlock", false, "do not lock the model in RAM"),
		noMmap:            set.Bool("no-mmap", false, "load the model without mmap"),
		batchSize:         set.Int("batch-size", 0, "logical batch size"),
		ubatchSize:        set.Int("ubatch-size", 0, "physical micro-batch size"),
		threadsBatch:      set.Int("threads-batch", 0, "threads used for prompt processing"),
		noContBatching:    set.Bool("no-cont-batching", false, "disable continuous batching"),
		parallel:          set.Int("parallel", 0, "number of server slots"),
		numa:              set.String("numa", "", "NUMA policy: distribute, isolate or numactl"),
		splitMode:         set.String("split-mode", "", "GPU split mode: none, layer, row or tensor"),
		tensorSplit:       set.String("tensor-split", "", "GPU proportions, for example 3,1"),
		cpuMoE:            set.Bool("cpu-moe", false, "keep all MoE experts on the CPU"),
		noCPUMoE:          set.Bool("no-cpu-moe", false, "do not force MoE experts onto the CPU"),
		nCPUMoE:           set.Int("n-cpu-moe", -1, "keep the experts of the first N layers on the CPU (0 = off)"),
		moeCacheMiB:       set.Int("moe-cache-mib", -1, "GPU cache in MiB for MoE experts kept on the CPU"),
		moeCacheShare:     set.Float64("moe-cache-cpu-share", -1, "share of MoE cache misses computed on the CPU (0..1)"),
		overrideTensor:    set.String("override-tensor", "", "llama.cpp tensor placement override"),
		speculative:       set.Bool("speculative", false, "enable ngram speculative decoding without a draft model"),
		specType:          set.String("spec-type", "", "speculative decoding type, for example ngram-simple or draft-simple"),
		draftModel:        set.String("draft-model", "", "draft GGUF model for speculative decoding"),
		draftMax:          set.Int("draft-max", -1, "maximum draft tokens"),
		draftMin:          set.Int("draft-min", -1, "minimum draft tokens"),
		draftPMin:         set.Float64("draft-p-min", -1, "minimum draft probability"),
		draftGPULayers:    set.Int("draft-gpu-layers", -1, "GPU layers for the draft model"),
		cacheReuse:        set.Int("cache-reuse", -1, "minimum chunk size for KV cache reuse (0 = off)"),
		contextShift:      set.Bool("context-shift", false, "enable context shifting"),
		noContextShift:    set.Bool("no-context-shift", false, "disable context shifting"),
		swaFull:           set.Bool("swa-full", false, "use the full-size SWA cache"),
		swaCheckpoints:    set.Int("swa-checkpoints", 32, "maximum SWA/context checkpoints per slot (0 disables)"),
		checkpointMinStep: set.Int("checkpoint-min-step", 8192, "minimum spacing between checkpoints in tokens (0 = none)"),
		ropeScaling:       set.String("rope-scaling", "", "RoPE scaling: none, linear or yarn"),
		ropeScale:         set.Float64("rope-scale", 0, "RoPE context scaling factor"),
		yarnOrigCtx:       set.Int("yarn-orig-ctx", 0, "YaRN original context size (0 = model default)"),
		cpuMask:           set.String("cpu-mask", "", "CPU affinity mask in hexadecimal"),
		cpuRange:          set.String("cpu-range", "", "CPU affinity range, for example 0-7"),
		noRepack:          set.Bool("no-repack", false, "disable runtime weight repacking"),
		repack:            set.Bool("repack", false, "enable runtime weight repacking"),
		noWarmup:          set.Bool("no-warmup", false, "skip the llama-server warmup run"),
		warmup:            set.Bool("warmup", false, "force the llama-server warmup run"),
		priority:          set.Int("priority", -999, "process priority: -1 low, 0 normal, 1 medium, 2 high, 3 realtime"),
		poll:              set.Int("poll", -1, "polling level 0..100"),
		adaptive:          set.Bool("adaptive", true, "fit parameters automatically to free RAM/VRAM"),
		effort:            set.String("effort", "auto", "prompt effort: auto, light, standard or deep"),
	}
}

func (f *optimizationFlags) options() optimization.Options {
	visited := map[string]bool{}
	f.set.Visit(func(item *flag.Flag) { visited[item.Name] = true })

	opts := optimization.Options{
		Profile:           *f.profile,
		FlashAttention:    *f.flashAttention,
		KVCacheType:       *f.kvCache,
		KVCacheTypeK:      *f.kvCacheK,
		KVCacheTypeV:      *f.kvCacheV,
		SlotSavePath:      *f.slotSavePath,
		NoKVOffload:       *f.noKVOffload,
		NoOpOffload:       *f.noOpOffload,
		LoadMode:          *f.loadMode,
		LazyMode:          *f.lazyMode,
		NoMmap:            *f.noMmap,
		BatchSize:         *f.batchSize,
		UBatchSize:        *f.ubatchSize,
		ThreadsBatch:      *f.threadsBatch,
		NoContBatching:    *f.noContBatching,
		Parallel:          *f.parallel,
		NumA:              *f.numa,
		SplitMode:         *f.splitMode,
		TensorSplit:       *f.tensorSplit,
		OverrideTensor:    *f.overrideTensor,
		Speculative:       *f.speculative,
		SpecType:          *f.specType,
		DraftModel:        *f.draftModel,
		CacheReuse:        -1,
		SWAFull:           *f.swaFull,
		SWACheckpoints:    -1,
		CheckpointMinStep: -1,
		RopeScaling:       *f.ropeScaling,
		RopeScale:         *f.ropeScale,
		YarnOrigCtx:       *f.yarnOrigCtx,
		CPUMask:           *f.cpuMask,
		CPURange:          *f.cpuRange,
		Priority:          -999,
		Poll:              -1,
		Adaptive:          *f.adaptive,
		Effort:            *f.effort,
		NCPUMoE:           -1,
		MoECacheMiB:       -1,
		MoECacheCPUShare:  -1,
		DraftMax:          -1,
		DraftMin:          -1,
		DraftPMin:         -1,
		DraftGPULayers:    -1,
	}
	if opts.KVCacheType == "auto" {
		opts.KVCacheType = ""
	}
	if opts.KVCacheTypeK == "auto" {
		opts.KVCacheTypeK = ""
	}
	if opts.KVCacheTypeV == "auto" {
		opts.KVCacheTypeV = ""
	}
	if visited["kv-unified"] {
		value := true
		opts.KVUnified = &value
	}
	if visited["no-kv-unified"] {
		value := false
		opts.KVUnified = &value
	}
	if visited["cache-ram"] {
		value := *f.cacheRAM
		opts.CacheRAMMiB = &value
	}
	if visited["swa-checkpoints"] {
		opts.SWACheckpoints = *f.swaCheckpoints
	}
	if visited["checkpoint-min-step"] {
		opts.CheckpointMinStep = *f.checkpointMinStep
	}
	if visited["no-repack"] {
		value := true
		opts.NoRepack = &value
	}
	if visited["repack"] {
		value := false
		opts.NoRepack = &value
	}
	if visited["no-warmup"] {
		value := true
		opts.NoWarmup = &value
	}
	if visited["warmup"] {
		value := false
		opts.NoWarmup = &value
	}
	if visited["mlock"] {
		value := true
		opts.Mlock = &value
	}
	if visited["no-mlock"] {
		value := false
		opts.Mlock = &value
	}
	if visited["cpu-moe"] {
		value := true
		opts.CPUMoE = &value
	}
	if visited["no-cpu-moe"] {
		value := false
		opts.CPUMoE = &value
	}
	if visited["context-shift"] {
		value := true
		opts.ContextShift = &value
	}
	if visited["no-context-shift"] {
		value := false
		opts.ContextShift = &value
	}
	if visited["n-cpu-moe"] {
		opts.NCPUMoE = *f.nCPUMoE
	}
	if visited["moe-cache-mib"] {
		opts.MoECacheMiB = *f.moeCacheMiB
	}
	if visited["moe-cache-cpu-share"] {
		opts.MoECacheCPUShare = *f.moeCacheShare
	}
	if visited["draft-max"] {
		opts.DraftMax = *f.draftMax
	}
	if visited["draft-min"] {
		opts.DraftMin = *f.draftMin
	}
	if visited["draft-p-min"] {
		opts.DraftPMin = *f.draftPMin
	}
	if visited["draft-gpu-layers"] {
		opts.DraftGPULayers = *f.draftGPULayers
	}
	if visited["cache-reuse"] {
		opts.CacheReuse = *f.cacheReuse
	}
	if visited["priority"] {
		opts.Priority = *f.priority
	}
	if visited["poll"] {
		opts.Poll = *f.poll
	}
	return opts
}

func optimizationValueFlags() map[string]bool {
	return map[string]bool{
		"--profile": true, "--flash-attn": true, "--kv-cache": true,
		"--kv-cache-k": true, "--kv-cache-v": true, "--cache-ram": true,
		"--slot-save-path": true, "--load-mode": true, "--lazy-mode": true,
		"--batch-size": true, "--ubatch-size": true, "--threads-batch": true,
		"--parallel": true, "--numa": true, "--split-mode": true, "--tensor-split": true,
		"--n-cpu-moe": true, "--moe-cache-mib": true, "--moe-cache-cpu-share": true,
		"--override-tensor": true, "--spec-type": true, "--draft-model": true,
		"--draft-max": true, "--draft-min": true, "--draft-p-min": true,
		"--draft-gpu-layers": true, "--cache-reuse": true, "--priority": true, "--poll": true,
		"--swa-checkpoints": true, "--checkpoint-min-step": true,
		"--rope-scaling": true, "--rope-scale": true, "--yarn-orig-ctx": true,
		"--cpu-mask": true, "--cpu-range": true, "--effort": true,
	}
}

func withOptimizationValueFlags(extra map[string]bool) map[string]bool {
	flags := optimizationValueFlags()
	for name, takesValue := range extra {
		flags[name] = takesValue
	}
	return flags
}

func (f *optimizationFlags) touched() bool {
	touched := false
	f.set.Visit(func(item *flag.Flag) {
		if isOptimizationFlag(item.Name) {
			touched = true
		}
	})
	return touched
}

func isOptimizationFlag(name string) bool {
	switch name {
	case "profile", "flash-attn", "kv-cache", "kv-cache-k", "kv-cache-v",
		"kv-unified", "no-kv-unified", "cache-ram", "slot-save-path",
		"no-kv-offload", "no-op-offload", "load-mode", "lazy-mode", "mlock", "no-mlock", "no-mmap",
		"batch-size", "ubatch-size", "threads-batch", "no-cont-batching", "parallel",
		"numa", "split-mode", "tensor-split", "cpu-moe", "no-cpu-moe", "n-cpu-moe",
		"moe-cache-mib", "moe-cache-cpu-share", "override-tensor",
		"speculative", "spec-type", "draft-model", "draft-max", "draft-min",
		"draft-p-min", "draft-gpu-layers", "cache-reuse", "context-shift",
		"no-context-shift", "swa-full", "swa-checkpoints", "checkpoint-min-step",
		"rope-scaling", "rope-scale", "yarn-orig-ctx", "cpu-mask", "cpu-range",
		"no-repack", "repack", "no-warmup", "warmup", "priority", "poll",
		"adaptive", "effort":
		return true
	}
	return false
}

func (f *optimizationFlags) validate() error {
	if _, err := optimization.ParseProfile(*f.profile); err != nil {
		return err
	}
	if _, err := adaptive.ParseEffort(*f.effort); err != nil {
		return err
	}
	if err := oneOf("flash-attn", *f.flashAttention, "auto", "on", "off"); err != nil {
		return err
	}
	kvTypes := []string{"auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"}
	if err := oneOf("kv-cache", *f.kvCache, kvTypes...); err != nil {
		return err
	}
	if err := oneOf("kv-cache-k", *f.kvCacheK, kvTypes...); err != nil {
		return err
	}
	if err := oneOf("kv-cache-v", *f.kvCacheV, kvTypes...); err != nil {
		return err
	}
	if err := oneOf("rope-scaling", *f.ropeScaling, "none", "linear", "yarn"); err != nil {
		return err
	}
	if err := oneOf("load-mode", *f.loadMode, "auto", "none", "mmap", "mlock", "mmap+mlock", "dio"); err != nil {
		return err
	}
	if err := oneOf("lazy-mode", *f.lazyMode, "auto", "on", "off"); err != nil {
		return err
	}
	if err := oneOf("split-mode", *f.splitMode, "none", "layer", "row", "tensor"); err != nil {
		return err
	}
	if err := oneOf("numa", *f.numa, "distribute", "isolate", "numactl"); err != nil {
		return err
	}
	if *f.moeCacheShare < -1 || *f.moeCacheShare > 1 {
		return fmt.Errorf("--moe-cache-cpu-share must be between 0 and 1")
	}
	if *f.draftPMin < -1 || *f.draftPMin > 1 {
		return fmt.Errorf("--draft-p-min must be between 0 and 1")
	}
	if *f.priority != -999 && (*f.priority < -1 || *f.priority > 3) {
		return fmt.Errorf("--priority must be between -1 and 3")
	}
	if *f.poll < -1 || *f.poll > 100 {
		return fmt.Errorf("--poll must be between 0 and 100")
	}
	cacheRAMSet := false
	f.set.Visit(func(item *flag.Flag) {
		if item.Name == "cache-ram" {
			cacheRAMSet = true
		}
	})
	if cacheRAMSet && *f.cacheRAM < -1 {
		return fmt.Errorf("--cache-ram must be -1 (unlimited), 0 (disabled) or a positive MiB value")
	}
	if *f.swaCheckpoints < 0 {
		return fmt.Errorf("--swa-checkpoints must be 0 or greater")
	}
	if *f.checkpointMinStep < 0 {
		return fmt.Errorf("--checkpoint-min-step must be 0 or greater")
	}
	if *f.ropeScale < 0 {
		return fmt.Errorf("--rope-scale must be positive")
	}
	if *f.yarnOrigCtx < 0 {
		return fmt.Errorf("--yarn-orig-ctx cannot be negative")
	}
	if err := validateCPUMask(*f.cpuMask); err != nil {
		return err
	}
	if err := validateCPURange(*f.cpuRange); err != nil {
		return err
	}
	for name, value := range map[string]int{
		"--batch-size": *f.batchSize, "--ubatch-size": *f.ubatchSize,
		"--threads-batch": *f.threadsBatch, "--parallel": *f.parallel,
		"--n-cpu-moe": *f.nCPUMoE, "--moe-cache-mib": *f.moeCacheMiB,
		"--draft-max": *f.draftMax, "--draft-min": *f.draftMin,
		"--draft-gpu-layers": *f.draftGPULayers, "--cache-reuse": *f.cacheReuse,
	} {
		if value < -1 {
			return fmt.Errorf("%s cannot be negative", name)
		}
	}
	if *f.ubatchSize > 0 && *f.batchSize > 0 && *f.ubatchSize > *f.batchSize {
		return fmt.Errorf("--ubatch-size cannot be larger than --batch-size")
	}
	if *f.mlock && *f.noMlock {
		return fmt.Errorf("--mlock and --no-mlock are mutually exclusive")
	}
	if *f.cpuMoE && *f.noCPUMoE {
		return fmt.Errorf("--cpu-moe and --no-cpu-moe are mutually exclusive")
	}
	if *f.contextShift && *f.noContextShift {
		return fmt.Errorf("--context-shift and --no-context-shift are mutually exclusive")
	}
	if *f.kvUnified && *f.noKVUnified {
		return fmt.Errorf("--kv-unified and --no-kv-unified are mutually exclusive")
	}
	if *f.cpuMask != "" && *f.cpuRange != "" {
		return fmt.Errorf("--cpu-mask and --cpu-range are mutually exclusive")
	}
	if *f.noRepack && *f.repack {
		return fmt.Errorf("--no-repack and --repack are mutually exclusive")
	}
	if *f.noWarmup && *f.warmup {
		return fmt.Errorf("--no-warmup and --warmup are mutually exclusive")
	}
	return nil
}

func validateCPUMask(value string) error {
	value = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(value)), "0x")
	if value == "" {
		return nil
	}
	for _, char := range value {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return fmt.Errorf("--cpu-mask must be a hexadecimal mask")
		}
	}
	if strings.Trim(value, "0") == "" {
		return fmt.Errorf("--cpu-mask cannot be zero")
	}
	return nil
}

func validateCPURange(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	loText, hiText, hasRange := strings.Cut(value, "-")
	lo, err := strconv.Atoi(strings.TrimSpace(loText))
	if err != nil || lo < 0 {
		return fmt.Errorf("--cpu-range must look like 0-7")
	}
	if !hasRange {
		return fmt.Errorf("--cpu-range must look like 0-7")
	}
	hi, err := strconv.Atoi(strings.TrimSpace(hiText))
	if err != nil || hi < lo {
		return fmt.Errorf("--cpu-range must look like 0-7 with low <= high")
	}
	return nil
}

func oneOf(name, value string, allowed ...string) error {
	if value == "" {
		return nil
	}
	for _, item := range allowed {
		if value == item {
			return nil
		}
	}
	return fmt.Errorf("--%s must be one of %v", name, allowed)
}
