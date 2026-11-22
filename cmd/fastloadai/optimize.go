package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/hf"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func commandOptimize(args []string) int {
	positional, flags := splitArgs(args, withOptimizationValueFlags(map[string]bool{
		"--config": true, "--models-dir": true,
	}))
	set := flag.NewFlagSet("optimize", flag.ContinueOnError)
	configPath := set.String("config", defaultConfig, "config file to update")
	modelsDir := set.String("models-dir", "models", "directory containing GGUF files")
	asJSON := set.Bool("json", false, "print the optimized config as JSON")
	noColor := set.Bool("no-color", false, "disable color")
	optimize := registerOptimizationFlags(set)
	if !parseFlags(set, flags) {
		return 2
	}
	if err := optimize.validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Invalid optimization flag: %v\n", err)
		return 2
	}

	cfg, exists, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid configuration: %v\n", err)
		return 2
	}
	requested := ""
	if len(positional) > 0 {
		requested = positional[0]
	}
	modelPath, err := resolveModel(requested, cfg, *modelsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not find the model: %v\n", err)
		fmt.Fprintln(os.Stderr, "Download it first with fastloadai download, then run optimize again.")
		return 2
	}
	info, err := os.Stat(modelPath)
	if err != nil || !info.Mode().IsRegular() {
		fmt.Fprintf(os.Stderr, "Model file not found: %s\n", modelPath)
		return 2
	}
	sizeBytes := cfg.ModelSizeBytes
	if sizeBytes <= 0 {
		sizeBytes = info.Size()
	}

	specs := hardware.Detect()
	base := hardware.RecommendSettings(specs, modelPath, sizeBytes)
	if exists {
		base.Mmap = cfg.Mmap
	}
	base.ModelPath = absModelPath(modelPath)
	sameModel := cfg.ModelPath != "" && absModelPath(cfg.ModelPath) == absModelPath(modelPath)
	base.ModelRepo = cfg.ModelRepo
	if !sameModel && !isRepoReference(requested) {
		base.ModelRepo = ""
	}
	if requested != "" && isRepoReference(requested) {
		base.ModelRepo = requested
	}
	base.Quantization = cfg.Quantization
	if !sameModel || base.Quantization == "" {
		base.Quantization = hf.QuantLabel(modelPath)
	}
	base.ModelSizeBytes = sizeBytes
	if cfg.OptimizationProfile != "" && !optimize.touched() && sameModel {
		cfg = optimization.Inherit(base, cfg)
	} else {
		cfg = optimization.Apply(base, modelPath, base.ModelRepo, optimize.options())
	}
	if err := saveConfig(*configPath, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write %s: %v\n", *configPath, err)
		return 1
	}
	if *asJSON {
		return printJSON(cfg)
	}

	theme := themeFor(*noColor)
	rows := [][2]string{
		{"Model", modelPath},
		{"Size", tui.FormatSize(cfg.ModelSizeGB)},
		{"Context", fmt.Sprintf("%d", cfg.ContextSize)},
		{"GPU layers", fmt.Sprintf("%d", cfg.NGPULayers)},
		{"Config file", *configPath},
	}
	rows = append(rows, optimization.SummaryRows(cfg)...)
	fmt.Println(tui.RenderKV("Optimization plan", rows, theme, "  "))
	fmt.Printf("\n%s\n  fastloadai run %s --config %s\n", theme.Paint("Next:", tui.Dim), requestedOrPath(requested, modelPath), *configPath)
	return 0
}

func requestedOrPath(requested, modelPath string) string {
	if requested != "" {
		return requested
	}
	return modelPath
}
