package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/hf"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
	"github.com/ciao22king/FastLoadAI/internal/safetensors"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

type downloadOptions struct {
	Repo      string
	File      string
	Directory string
	Output    string
	Token     string
	Revision  string
	Prefer    string
	Workers   int
	Specs     *model.Specs
	DryRun    bool
	Force     bool
	Optimize  optimization.Options
}

func commandDownload(args []string) int {
	positional, flags := splitArgs(args, withOptimizationValueFlags(map[string]bool{
		"--file": true, "--directory": true, "--output": true, "--token": true,
		"--revision": true, "--prefer": true, "--workers": true,
	}))
	set := flag.NewFlagSet("download", flag.ContinueOnError)
	file := set.String("file", "", "explicit GGUF file or shard")
	directory := set.String("directory", "models", "download directory")
	output := set.String("output", defaultConfig, "config file to write")
	token := set.String("token", "", "Hugging Face token")
	revision := set.String("revision", "", "Hugging Face revision")
	prefer := set.String("prefer", "balanced", "speed, balanced or quality")
	workers := set.Int("workers", 0, "parallel downloads (0 = auto)")
	dryRun := set.Bool("dry-run", false, "show the plan without downloading")
	force := set.Bool("force", false, "download even when the model may not fit")
	noColor := set.Bool("no-color", false, "disable color")
	optimize := registerOptimizationFlags(set)
	if !parseFlags(set, flags) {
		return 2
	}
	if err := optimize.validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Invalid optimization flag: %v\n", err)
		return 2
	}
	if len(positional) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: fastloadai download <repo> [options]")
		return 2
	}
	opts := downloadOptions{
		Repo: positional[0], File: *file, Directory: *directory, Output: *output,
		Token: tokenValue(*token), Revision: *revision, Prefer: *prefer,
		Workers: *workers, DryRun: *dryRun, Force: *force, Optimize: optimize.options(),
	}
	_, code := performDownload(context.Background(), opts, themeFor(*noColor))
	return code
}

func performDownload(ctx context.Context, opts downloadOptions, theme tui.Theme) (model.Config, int) {
	specsCh := make(chan model.Specs, 1)
	if opts.Specs != nil {
		specsCh <- *opts.Specs
	} else {
		go func() { specsCh <- hardware.Detect() }()
	}
	client := hf.NewClient(opts.Token)
	info, err := client.ModelInfo(ctx, opts.Repo, opts.Revision)
	specs := <-specsCh
	budget := hardware.UsableModelMemoryGB(specs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not inspect %s: %v\n", opts.Repo, err)
		if isAuthError(err) {
			fmt.Fprintln(os.Stderr, "This repository may be gated. Accept its license on huggingface.co, then pass --token or set HF_TOKEN.")
		}
		return model.Config{}, 1
	}
	group, err := hf.SelectGGUF(hf.GGUFGroups(info.Siblings), budget, opts.File, opts.Prefer)
	convertSafetensors := false
	if err != nil {
		group, err = selectSafetensorsGroup(info.Siblings)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot use %s: %v\n", opts.Repo, err)
			fmt.Fprintln(os.Stderr, "Choose a repository with GGUF files or a Hugging Face safetensors checkpoint.")
			return model.Config{}, 1
		}
		convertSafetensors = true
	}
	sizeGB := hf.GroupSizeGB(group)
	fit := hardware.ModelFit(specs, sizeGB)
	workers := hf.ClampWorkers(opts.Workers, len(group.Files))
	format := "GGUF"
	if convertSafetensors {
		format = "safetensors → GGUF"
	}
	rows := [][2]string{
		{"Repository", opts.Repo},
		{"Format", format},
		{"File", group.Name},
		{"Files", fmt.Sprintf("%d", len(group.Files))},
		{"Parallel", workerLabel(workers)},
		{"Size", sizeLabel(group)},
		{"Memory budget", tui.FormatSize(budget)},
		{"Model fit", theme.Paint(fit, fitStyle(fit))},
	}
	if opts.Revision != "" {
		rows = append([][2]string{{"Revision", opts.Revision}}, rows...)
	}
	if opts.DryRun {
		profile := opts.Optimize.Profile
		if profile == "" {
			profile = "auto"
		}
		rows = append(rows, [2]string{"Optimization", profile})
		if optimization.DetectMoE("", opts.Repo) {
			rows = append(rows, [2]string{"Model type", "Mixture of Experts (MoE)"})
		}
		fmt.Println(tui.RenderKV("Download plan (dry run)", rows, theme, "  "))
		if fit == "too_large" {
			fmt.Println(theme.Paint("This file is larger than the estimated memory. Use --prefer speed, a smaller model, or --force.", tui.Yellow))
		}
		return model.Config{}, 0
	}
	if fit == "too_large" && opts.File == "" && !opts.Force {
		fmt.Println(tui.RenderKV("Download plan", rows, theme, "  "))
		fmt.Fprintln(os.Stderr, theme.Paint("No quantization fits the estimated memory. Nothing was downloaded.", tui.Red))
		fmt.Fprintln(os.Stderr, "Try fastloadai suggest, --prefer speed, a smaller repository, or --force.")
		return model.Config{}, 1
	}
	if group.SizeKnown && specs.DiskFreeGB > 0 && sizeGB*1.05 > specs.DiskFreeGB {
		fmt.Fprintf(os.Stderr, "Not enough disk space: need about %s, %s free.\n", tui.FormatSize(sizeGB*1.05), tui.FormatSize(specs.DiskFreeGB))
		return model.Config{}, 1
	}
	if fit == "too_large" {
		fmt.Fprintln(os.Stderr, theme.Paint("Warning: this model is larger than the estimated memory. It may load slowly or fail.", tui.Yellow))
	}
	localDir := filepath.Join(opts.Directory, filepath.FromSlash(opts.Repo))
	spinner := tui.StartSpinner("downloading "+group.Name+" with "+workerLabel(workers), theme)
	paths, err := client.DownloadGroup(ctx, opts.Repo, opts.Revision, localDir, group, workers)
	spinner.Stop()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Download failed: %v\n", err)
		if isAuthError(err) {
			fmt.Fprintln(os.Stderr, "Accept the model license on huggingface.co, then pass --token or set HF_TOKEN.")
		}
		return model.Config{}, 1
	}
	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "Download finished without model files.")
		return model.Config{}, 1
	}
	modelPath := absModelPath(paths[0])
	sizeBytes := group.Size
	quantization := hf.QuantLabel(group.Name)
	quantizationFile := group.Name
	if convertSafetensors {
		if !safetensors.IsModelDir(localDir) {
			fmt.Fprintf(os.Stderr, "Downloaded safetensors to %s, but the checkpoint is incomplete.\n", localDir)
			return model.Config{}, 1
		}
		fmt.Fprintln(os.Stderr, "Converting safetensors to GGUF...")
		converted, err := safetensors.EnsureConverted(localDir, opts.Directory)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Conversion failed: %v\n", err)
			return model.Config{}, 1
		}
		modelPath = absModelPath(converted)
		if info, err := os.Stat(modelPath); err == nil {
			sizeBytes = info.Size()
		}
		quantization, _ = safetensors.NativeDtype(localDir)
		if quantization == "" {
			quantization = "native"
		}
		quantizationFile = filepath.Base(modelPath)
	}
	cfg := hardware.RecommendSettings(specs, modelPath, sizeBytes)
	cfg.ModelPath = modelPath
	cfg.ModelRepo = opts.Repo
	cfg.Revision = opts.Revision
	cfg.Quantization = quantization
	cfg.QuantizationFile = quantizationFile
	cfg.ModelSizeBytes = sizeBytes
	cfg = optimization.Apply(cfg, modelPath, opts.Repo, opts.Optimize)
	if err := saveConfig(opts.Output, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write %s: %v\n", opts.Output, err)
		return model.Config{}, 1
	}
	rows = [][2]string{
		{"Model", modelPath},
		{"Quantization", cfg.Quantization},
		{"Size", sizeLabel(group)},
		{"Model fit", cfg.ModelFit},
		{"Context", fmt.Sprintf("%d", cfg.ContextSize)},
		{"Threads", fmt.Sprintf("%d", cfg.CPUThreads)},
		{"Profile", cfg.OptimizationProfile},
	}
	if cfg.MoE {
		rows = append(rows, [2]string{"Model type", "Mixture of Experts (MoE)"})
	}
	rows = append(rows, [2]string{"Flash attention", cfg.FlashAttention}, [2]string{"KV cache", cfg.KVCacheTypeK})
	rows = append(rows, [2]string{"Config file", opts.Output})
	fmt.Println(tui.RenderKV("Model ready", rows, theme, "  "))
	return cfg, 0
}

func selectSafetensorsGroup(siblings []hf.Sibling) (hf.Group, error) {
	required := map[string]bool{
		"config.json": true, "tokenizer.json": true, "tokenizer_config.json": true,
		"generation_config.json": true, "model.safetensors.index.json": true,
		"special_tokens_map.json": true, "vocab.json": true, "merges.txt": true,
	}
	group := hf.Group{Name: "safetensors checkpoint", FileSizes: map[string]int64{}, SizeKnown: true}
	for _, sibling := range siblings {
		name := sibling.Filename
		if !required[name] && !strings.HasSuffix(name, ".safetensors") {
			continue
		}
		group.Files = append(group.Files, name)
		if sibling.Size == nil {
			group.SizeKnown = false
			continue
		}
		group.FileSizes[name] = *sibling.Size
		group.Size += *sibling.Size
	}
	hasWeights := false
	for _, name := range group.Files {
		if strings.HasSuffix(name, ".safetensors") {
			hasWeights = true
			break
		}
	}
	if !hasWeights {
		return hf.Group{}, fmt.Errorf("the repository has neither GGUF nor safetensors weights")
	}
	return group, nil
}

func workerLabel(workers int) string {
	if workers == 1 {
		return "1 worker"
	}
	return fmt.Sprintf("%d workers", workers)
}

func sizeLabel(group hf.Group) string {
	if !group.SizeKnown {
		return "unknown"
	}
	return tui.FormatSize(hf.GroupSizeGB(group))
}

func fitStyle(fit string) string {
	switch fit {
	case "comfortable":
		return tui.Green
	case "tight":
		return tui.Yellow
	default:
		return tui.Red
	}
}

func isAuthError(err error) bool {
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "gated") || strings.Contains(message, "401") || strings.Contains(message, "403")
}
