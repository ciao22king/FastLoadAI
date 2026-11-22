package main

import (
	"flag"
	"fmt"

	"github.com/ciao22king/FastLoadAI/internal/optimization"
)

type optimizationFlags struct {
	set *flag.FlagSet

	profile        *string
	flashAttention *string
	kvCache        *string
	noKVOffload    *bool
	noOpOffload    *bool
	loadMode       *string
	lazyMode       *string
	mlock          *bool
	noMlock        *bool
	noMmap         *bool
	batchSize      *int
	ubatchSize     *int
	threadsBatch   *int
	noContBatching *bool
	parallel       *int
	numa           *string
	splitMode      *string
	tensorSplit    *string
	cpuMoE         *bool
	noCPUMoE       *bool
	nCPUMoE        *int
	moeCacheMiB    *int
	moeCacheShare  *float64
	overrideTensor *string
	speculative    *bool
	specType       *string
	draftModel     *string
	draftMax       *int
	draftMin       *int
	draftPMin      *float64
	draftGPULayers *int
	cacheReuse     *int
	contextShift   *bool
	noContextShift *bool
	swaFull        *bool
	priority       *int
	poll           *int
}

func registerOptimizationFlags(set *flag.FlagSet) *optimizationFlags {
	return &optimizationFlags{
		set:            set,
		profile:        set.String("profile", "", "optimization profile: auto, speed, balanced, memory or quality"),
		flashAttention: set.String("flash-attn", "", "Flash Attention: on, off or auto"),
		kvCache:        set.String("kv-cache", "", "KV cache type: f16, q8_0 or q4_0"),
		noKVOffload:    set.Bool("no-kv-offload", false, "keep the KV cache on the CPU"),
		noOpOffload:    set.Bool("no-op-offload", false, "disable offloading host tensor operations"),
		loadMode:       set.String("load-mode", "", "model load mode: auto, none, mmap, mlock, mmap+mlock or dio"),
		lazyMode:       set.String("lazy-mode", "", "lazy tensor loading: on, auto or off"),
		mlock:          set.Bool("mlock", false, "lock the model in RAM (mmap+mlock)"),
		noMlock:        set.Bool("no-mlock", false, "do not lock the model in RAM"),
		noMmap:         set.Bool("no-mmap", false, "load the model without mmap"),
		batchSize:      set.Int("batch-size", 0, "logical batch size"),
		ubatchSize:     set.Int("ubatch-size", 0, "physical micro-batch size"),
		threadsBatch:   set.Int("threads-batch", 0, "threads used for prompt processing"),
		noContBatching: set.Bool("no-cont-batching", false, "disable continuous batching"),
		parallel:       set.Int("parallel", 0, "number of server slots"),
		numa:           set.String("numa", "", "NUMA policy: distribute, isolate or numactl"),
		splitMode:      set.String("split-mode", "", "GPU split mode: none, layer, row or tensor"),
		tensorSplit:    set.String("tensor-split", "", "GPU proportions, for example 3,1"),
		cpuMoE:         set.Bool("cpu-moe", false, "keep all MoE experts on the CPU"),
		noCPUMoE:       set.Bool("no-cpu-moe", false, "do not force MoE experts onto the CPU"),
		nCPUMoE:        set.Int("n-cpu-moe", -1, "keep the experts of the first N layers on the CPU (0 = off)"),
		moeCacheMiB:    set.Int("moe-cache-mib", -1, "GPU cache in MiB for MoE experts kept on the CPU"),
		moeCacheShare:  set.Float64("moe-cache-cpu-share", -1, "share of MoE cache misses computed on the CPU (0..1)"),
		overrideTensor: set.String("override-tensor", "", "llama.cpp tensor placement override"),
		speculative:    set.Bool("speculative", false, "enable ngram speculative decoding without a draft model"),
		specType:       set.String("spec-type", "", "speculative decoding type, for example ngram-simple or draft-simple"),
		draftModel:     set.String("draft-model", "", "draft GGUF model for speculative decoding"),
		draftMax:       set.Int("draft-max", -1, "maximum draft tokens"),
		draftMin:       set.Int("draft-min", -1, "minimum draft tokens"),
		draftPMin:      set.Float64("draft-p-min", -1, "minimum draft probability"),
		draftGPULayers: set.Int("draft-gpu-layers", -1, "GPU layers for the draft model"),
		cacheReuse:     set.Int("cache-reuse", -1, "minimum chunk size for KV cache reuse (0 = off)"),
		contextShift:   set.Bool("context-shift", false, "enable context shifting"),
		noContextShift: set.Bool("no-context-shift", false, "disable context shifting"),
		swaFull:        set.Bool("swa-full", false, "use the full-size SWA cache"),
		priority:       set.Int("priority", -999, "process priority: -1 low, 0 normal, 1 medium, 2 high, 3 realtime"),
		poll:           set.Int("poll", -1, "polling level 0..100"),
	}
}

func (f *optimizationFlags) options() optimization.Options {
	visited := map[string]bool{}
	f.set.Visit(func(item *flag.Flag) { visited[item.Name] = true })

	opts := optimization.Options{
		Profile:          *f.profile,
		FlashAttention:   *f.flashAttention,
		KVCacheType:      *f.kvCache,
		NoKVOffload:      *f.noKVOffload,
		NoOpOffload:      *f.noOpOffload,
		LoadMode:         *f.loadMode,
		LazyMode:         *f.lazyMode,
		NoMmap:           *f.noMmap,
		BatchSize:        *f.batchSize,
		UBatchSize:       *f.ubatchSize,
		ThreadsBatch:     *f.threadsBatch,
		NoContBatching:   *f.noContBatching,
		Parallel:         *f.parallel,
		NumA:             *f.numa,
		SplitMode:        *f.splitMode,
		TensorSplit:      *f.tensorSplit,
		OverrideTensor:   *f.overrideTensor,
		Speculative:      *f.speculative,
		SpecType:         *f.specType,
		DraftModel:       *f.draftModel,
		CacheReuse:       -1,
		SWAFull:          *f.swaFull,
		Priority:         -999,
		Poll:             -1,
		NCPUMoE:          -1,
		MoECacheMiB:      -1,
		MoECacheCPUShare: -1,
		DraftMax:         -1,
		DraftMin:         -1,
		DraftPMin:        -1,
		DraftGPULayers:   -1,
	}
	if opts.KVCacheType == "auto" {
		opts.KVCacheType = ""
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
		"--load-mode": true, "--lazy-mode": true,
		"--batch-size": true, "--ubatch-size": true, "--threads-batch": true,
		"--parallel": true, "--numa": true, "--split-mode": true, "--tensor-split": true,
		"--n-cpu-moe": true, "--moe-cache-mib": true, "--moe-cache-cpu-share": true,
		"--override-tensor": true, "--spec-type": true, "--draft-model": true,
		"--draft-max": true, "--draft-min": true, "--draft-p-min": true,
		"--draft-gpu-layers": true, "--cache-reuse": true, "--priority": true, "--poll": true,
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
	case "profile", "flash-attn", "kv-cache", "no-kv-offload", "no-op-offload",
		"load-mode", "lazy-mode", "mlock", "no-mlock", "no-mmap",
		"batch-size", "ubatch-size", "threads-batch", "no-cont-batching", "parallel",
		"numa", "split-mode", "tensor-split", "cpu-moe", "no-cpu-moe", "n-cpu-moe",
		"moe-cache-mib", "moe-cache-cpu-share", "override-tensor",
		"speculative", "spec-type", "draft-model", "draft-max", "draft-min",
		"draft-p-min", "draft-gpu-layers", "cache-reuse", "context-shift",
		"no-context-shift", "swa-full", "priority", "poll":
		return true
	}
	return false
}

func (f *optimizationFlags) validate() error {
	if _, err := optimization.ParseProfile(*f.profile); err != nil {
		return err
	}
	if err := oneOf("flash-attn", *f.flashAttention, "auto", "on", "off"); err != nil {
		return err
	}
	if err := oneOf("kv-cache", *f.kvCache, "auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"); err != nil {
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
