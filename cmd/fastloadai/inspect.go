package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ciao22king/FastLoadAI/internal/catalog"
	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func commandScan(args []string) int {
	set := flag.NewFlagSet("scan", flag.ContinueOnError)
	asJSON := set.Bool("json", false, "print JSON")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, args) {
		return 2
	}
	specs := hardware.Detect()
	if *asJSON {
		return printJSON(specs)
	}
	fmt.Println(tui.RenderHardware(specs, themeFor(*noColor)))
	return 0
}

func commandPlan(args []string) int {
	positional, flags := splitArgs(args, withOptimizationValueFlags(map[string]bool{
		"--output": true, "--prompt": true,
	}))
	set := flag.NewFlagSet("plan", flag.ContinueOnError)
	output := set.String("output", defaultConfig, "config file to write")
	lowMemory := set.Bool("low-memory", false, "force a smaller context and small batches")
	noColor := set.Bool("no-color", false, "disable color")
	prompt := set.String("prompt", "", "prompt hint used to choose the effort tier")
	optimize := registerOptimizationFlags(set)
	if !parseFlags(set, flags) {
		return 2
	}
	if err := optimize.validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Invalid optimization flag: %v\n", err)
		return 2
	}
	specs := hardware.Detect()
	modelPath := ""
	if len(positional) > 0 {
		modelPath = absModelPath(positional[0])
	}
	cfg := hardware.RecommendSettings(specs, modelPath, 0)
	cfg.ModelPath = modelPath
	if *lowMemory {
		cfg.LowMemory = true
		cfg.ContextSize = min(cfg.ContextSize, 1024)
	}
	opts := optimize.options()
	opts.Prompt = *prompt
	cfg = optimization.Apply(cfg, modelPath, "", opts)
	if err := saveConfig(*output, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Could not write %s: %v\n", *output, err)
		return 1
	}
	fmt.Println(tui.RenderPlan(cfg, *output, themeFor(*noColor)))
	return 0
}

func commandSuggest(args []string) int {
	set := flag.NewFlagSet("suggest", flag.ContinueOnError)
	asJSON := set.Bool("json", false, "print JSON")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, args) {
		return 2
	}
	specs := hardware.Detect()
	rows := catalog.Rows(specs)
	if *asJSON {
		return printJSON(rows)
	}
	view := make([]tui.SuggestionRow, 0, len(rows))
	for _, row := range rows {
		view = append(view, tui.SuggestionRow{
			Name: row.Name, Repo: row.Repo, SizeGB: row.SizeGB, Note: row.Note,
			Fit: row.Fit, Recommended: row.Recommended,
		})
	}
	theme := themeFor(*noColor)
	fmt.Println(tui.RenderSuggestions(view, catalog.Budget(specs), theme))
	for _, row := range rows {
		if row.Recommended {
			fmt.Printf("\n%s\n  fastloadai run %s\n", theme.Paint("Recommended for this machine:", tui.Dim), row.Repo)
			return 0
		}
	}
	fmt.Println("\nNo catalog model fits comfortably. Try --prefer speed with --force, or a smaller repository.")
	return 0
}

func commandModels(args []string) int {
	set := flag.NewFlagSet("models", flag.ContinueOnError)
	root := set.String("models-dir", "models", "directory containing GGUF files")
	asJSON := set.Bool("json", false, "print JSON")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, args) {
		return 2
	}
	specs := hardware.Detect()
	groups, err := localGGUFGroups(*root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not scan %s: %v\n", *root, err)
		return 1
	}
	if *asJSON {
		return printJSON(groups)
	}
	theme := themeFor(*noColor)
	view := make([]tui.ModelRow, 0, len(groups))
	for _, group := range groups {
		relative, _ := filepath.Rel(*root, group.Path)
		view = append(view, tui.ModelRow{
			Name: group.Name, Path: relative, SizeGB: float64(group.SizeBytes) / (1024 * 1024 * 1024),
			Quantization: group.Quantization, Fit: hardware.ModelFit(specs, float64(group.SizeBytes)/(1024*1024*1024)),
		})
	}
	fmt.Println(tui.RenderModels(view, *root, theme))
	return 0
}
