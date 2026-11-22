package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"

	"github.com/ciao22king/FastLoadAI/internal/catalog"
	"github.com/ciao22king/FastLoadAI/internal/chat"
	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/hf"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func commandRun(args []string) int {
	positional, flags := splitArgs(args, withOptimizationValueFlags(map[string]bool{
		"--config": true, "--models-dir": true, "--prefer": true, "--token": true,
		"--revision": true, "--prompt": true, "--threads": true, "--ctx-size": true,
		"--gpu-layers": true, "--temperature": true, "--max-tokens": true, "--system": true,
		"--workers": true,
	}))
	set := flag.NewFlagSet("run", flag.ContinueOnError)
	configPath := set.String("config", defaultConfig, "config file")
	modelsDir := set.String("models-dir", "models", "models directory")
	prefer := set.String("prefer", "balanced", "speed, balanced or quality")
	token := set.String("token", "", "Hugging Face token")
	revision := set.String("revision", "", "Hugging Face revision")
	force := set.Bool("force", false, "download even when the model may not fit")
	workers := set.Int("workers", 0, "parallel downloads (0 = auto)")
	prompt := set.String("prompt", "", "send one message before the interactive prompt")
	once := set.Bool("once", false, "answer the prompt and exit")
	threads := set.Int("threads", 0, "override CPU threads")
	ctxSize := set.Int("ctx-size", 0, "override context window")
	gpuLayers := set.Int("gpu-layers", -1, "override GPU layers")
	temperature := set.Float64("temperature", 0.7, "sampling temperature")
	maxTokens := set.Int("max-tokens", 0, "maximum tokens per answer")
	system := set.String("system", "", "replace the system prompt")
	lowMemory := set.Bool("low-memory", false, "force a smaller context and small batches")
	noColor := set.Bool("no-color", false, "disable color")
	noTUI := set.Bool("no-tui", false, "use the plain line-based chat instead of the full-screen UI")
	optimize := registerOptimizationFlags(set)
	if !parseFlags(set, flags) {
		return 2
	}
	if err := optimize.validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Invalid optimization flag: %v\n", err)
		return 2
	}
	requested := ""
	if len(positional) > 0 {
		requested = positional[0]
	}
	cfg, exists, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid configuration: %v\n", err)
		return 2
	}
	if !exists {
		cfg = model.Config{}
	}
	specs := hardware.Detect()
	modelPath, resolveErr := resolveModel(requested, cfg, *modelsDir)
	if resolveErr != nil || modelPath == "" {
		repo := ""
		if requested != "" {
			if isRepoReference(requested) {
				repo = requested
			} else if matched, ok := catalog.Match(requested); ok {
				repo = matched.Repo
			}
		}
		if repo == "" && requested == "" {
			repo = cfg.ModelRepo
		}
		if repo == "" {
			if requested != "" {
				fmt.Fprintf(os.Stderr, "Model %q was not found. Use a downloaded name, a GGUF path, a Hugging Face repo ID, or a catalog name such as qwen2.5-0.5b.\n", requested)
			} else {
				fmt.Fprintln(os.Stderr, "No model configured. Run fastloadai run qwen2.5-0.5b or download a model first.")
			}
			return 2
		}
		if _, err := exec.LookPath("llama-server"); err != nil {
			fmt.Fprintln(os.Stderr, "llama-server not found. Install llama.cpp before running a model.")
			return 2
		}
		label := requested
		if label == "" {
			label = repo
		}
		fmt.Printf("Preparing %s for this machine...\n", label)
		downloaded, code := performDownload(context.Background(), downloadOptions{
			Repo: repo, Directory: *modelsDir, Output: *configPath,
			Token: tokenValue(*token), Revision: *revision, Prefer: *prefer, Force: *force,
			Workers: *workers, Specs: &specs,
		}, themeFor(*noColor))
		if code != 0 {
			return code
		}
		cfg = downloaded
		modelPath, err = resolveModel("", cfg, *modelsDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid configuration: %v\n", err)
			return 2
		}
	}
	info, err := os.Stat(modelPath)
	if err != nil || !info.Mode().IsRegular() {
		fmt.Fprintf(os.Stderr, "Model file not found: %s\n", modelPath)
		return 2
	}
	sizeBytes := int64(0)
	if cfg.ModelPath != "" && absModelPath(cfg.ModelPath) == absModelPath(modelPath) {
		sizeBytes = cfg.ModelSizeBytes
	}
	if sizeBytes <= 0 {
		sizeBytes = info.Size()
	}
	saved := cfg
	base := hardware.RecommendSettings(specs, modelPath, sizeBytes)
	if exists {
		base.Mmap = saved.Mmap
	}
	base.ModelPath = absModelPath(modelPath)
	sameModel := saved.ModelPath != "" && absModelPath(saved.ModelPath) == absModelPath(modelPath)
	base.ModelRepo = saved.ModelRepo
	if !sameModel && !isRepoReference(requested) {
		base.ModelRepo = ""
	}
	if requested != "" && isRepoReference(requested) {
		base.ModelRepo = requested
	}
	base.Quantization = saved.Quantization
	if !sameModel || base.Quantization == "" {
		base.Quantization = hf.QuantLabel(modelPath)
	}
	base.ModelSizeBytes = sizeBytes
	if *lowMemory {
		base.LowMemory = true
		if *ctxSize == 0 {
			base.ContextSize = min(base.ContextSize, 1024)
		}
	}
	if *threads > 0 {
		base.CPUThreads = *threads
	}
	if *ctxSize > 0 {
		base.ContextSize = *ctxSize
	}
	if *gpuLayers >= 0 {
		base.NGPULayers = *gpuLayers
	}
	if saved.OptimizationProfile != "" && !optimize.touched() && !*lowMemory && sameModel {
		cfg = optimization.Inherit(base, saved)
	} else {
		cfg = optimization.Apply(base, modelPath, base.ModelRepo, optimize.options())
	}
	if *threads > 0 {
		cfg.CPUThreads = *threads
	}
	if *ctxSize > 0 {
		cfg.ContextSize = *ctxSize
	}
	if *gpuLayers >= 0 {
		cfg.NGPULayers = *gpuLayers
	}
	cfg.CPUThreads = max(1, cfg.CPUThreads)
	cfg.ContextSize = max(256, cfg.ContextSize)
	cfg.NGPULayers = max(0, cfg.NGPULayers)
	if err := saveConfig(*configPath, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write %s: %v\n", *configPath, err)
		return 1
	}
	theme := themeFor(*noColor)
	if cfg.ModelFit == "too_large" {
		fmt.Fprintln(os.Stderr, theme.Paint("Warning: this model is larger than the estimated memory. Expect slow loading and possible failures; try a smaller quantization.", tui.Yellow))
	} else if cfg.ModelFit == "tight" {
		fmt.Fprintln(os.Stderr, theme.Paint("Note: this model is close to the memory limit. Close other apps if loading fails.", tui.Yellow))
	}
	options := chat.Options{
		InitialPrompt: *prompt, Temperature: *temperature, MaxTokens: *maxTokens,
		SystemPrompt: *system, Once: *once, NoTUI: *noTUI,
		Color: !*noColor, ColorSet: *noColor,
	}
	return chat.RunChat(modelPath, cfg, options)
}
