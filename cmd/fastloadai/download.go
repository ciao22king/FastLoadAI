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
	DryRun    bool
	Force     bool
}

func commandDownload(args []string) int {
	positional, flags := splitArgs(args, map[string]bool{
		"--file": true, "--directory": true, "--output": true, "--token": true,
		"--revision": true, "--prefer": true,
	})
	set := flag.NewFlagSet("download", flag.ContinueOnError)
	file := set.String("file", "", "explicit GGUF file or shard")
	directory := set.String("directory", "models", "download directory")
	output := set.String("output", defaultConfig, "config file to write")
	token := set.String("token", "", "Hugging Face token")
	revision := set.String("revision", "", "Hugging Face revision")
	prefer := set.String("prefer", "balanced", "speed, balanced or quality")
	dryRun := set.Bool("dry-run", false, "show the plan without downloading")
	force := set.Bool("force", false, "download even when the model may not fit")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, flags) {
		return 2
	}
	if len(positional) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: fastloadai download <repo> [options]")
		return 2
	}
	opts := downloadOptions{
		Repo: positional[0], File: *file, Directory: *directory, Output: *output,
		Token: tokenValue(*token), Revision: *revision, Prefer: *prefer,
		DryRun: *dryRun, Force: *force,
	}
	_, code := performDownload(context.Background(), opts, themeFor(*noColor))
	return code
}

func performDownload(ctx context.Context, opts downloadOptions, theme tui.Theme) (model.Config, int) {
	specs := hardware.Detect()
	budget := hardware.UsableModelMemoryGB(specs)
	client := hf.NewClient(opts.Token)
	info, err := client.ModelInfo(ctx, opts.Repo, opts.Revision)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not inspect %s: %v\n", opts.Repo, err)
		if isAuthError(err) {
			fmt.Fprintln(os.Stderr, "This repository may be gated. Accept its license on huggingface.co, then pass --token or set HF_TOKEN.")
		}
		return model.Config{}, 1
	}
	group, err := hf.SelectGGUF(hf.GGUFGroups(info.Siblings), budget, opts.File, opts.Prefer)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Cannot use %s: %v\n", opts.Repo, err)
		fmt.Fprintln(os.Stderr, "Choose a repository with GGUF files, usually one whose name ends in -GGUF.")
		return model.Config{}, 1
	}
	sizeGB := hf.GroupSizeGB(group)
	fit := hardware.ModelFit(specs, sizeGB)
	rows := [][2]string{
		{"Repository", opts.Repo},
		{"File", group.Name},
		{"Files", fmt.Sprintf("%d", len(group.Files))},
		{"Size", sizeLabel(group)},
		{"Memory budget", tui.FormatSize(budget)},
		{"Model fit", theme.Paint(fit, fitStyle(fit))},
	}
	if opts.Revision != "" {
		rows = append([][2]string{{"Revision", opts.Revision}}, rows...)
	}
	if opts.DryRun {
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
	spinner := tui.StartSpinner("downloading "+group.Name, theme)
	paths, err := client.DownloadGroup(ctx, opts.Repo, opts.Revision, localDir, group, 4)
	spinner.Stop()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Download failed: %v\n", err)
		if isAuthError(err) {
			fmt.Fprintln(os.Stderr, "Accept the model license on huggingface.co, then pass --token or set HF_TOKEN.")
		}
		return model.Config{}, 1
	}
	if len(paths) == 0 {
		fmt.Fprintln(os.Stderr, "Download finished without a GGUF file.")
		return model.Config{}, 1
	}
	modelPath := absModelPath(paths[0])
	sizeBytes := group.Size
	cfg := hardware.RecommendSettings(specs, modelPath, sizeBytes)
	cfg.ModelPath = modelPath
	cfg.ModelRepo = opts.Repo
	cfg.Revision = opts.Revision
	cfg.Quantization = hf.QuantLabel(group.Name)
	cfg.QuantizationFile = group.Name
	cfg.ModelSizeBytes = sizeBytes
	cfg.LlamaCPPMoe = "Depends on architecture support in the installed llama.cpp build."
	if err := saveConfig(opts.Output, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write %s: %v\n", opts.Output, err)
		return model.Config{}, 1
	}
	fmt.Println(tui.RenderKV("Model ready", [][2]string{
		{"Model", modelPath},
		{"Quantization", cfg.Quantization},
		{"Size", sizeLabel(group)},
		{"Model fit", cfg.ModelFit},
		{"Context", fmt.Sprintf("%d", cfg.ContextSize)},
		{"Threads", fmt.Sprintf("%d", cfg.CPUThreads)},
		{"Config file", opts.Output},
	}, theme, "  "))
	return cfg, 0
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
