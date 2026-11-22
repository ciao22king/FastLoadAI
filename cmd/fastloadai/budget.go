package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/adaptive"
	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

type budgetReport struct {
	Model    adaptive.ModelProfile `json:"model"`
	Hardware model.Specs           `json:"hardware"`
	Budget   adaptive.Budget       `json:"budget"`
}

func commandBudget(args []string) int {
	positional, flags := splitArgs(args, map[string]bool{
		"--config": true, "--models-dir": true, "--prompt": true, "--effort": true,
	})
	set := flag.NewFlagSet("budget", flag.ContinueOnError)
	configPath := set.String("config", defaultConfig, "config file")
	modelsDir := set.String("models-dir", "models", "models directory")
	prompt := set.String("prompt", "", "prompt hint used to choose the effort tier")
	effort := set.String("effort", "auto", "effort tier: auto, light, standard or deep")
	asJSON := set.Bool("json", false, "print JSON")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, flags) {
		return 2
	}
	parsedEffort, err := adaptive.ParseEffort(*effort)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid effort: %v\n", err)
		return 2
	}
	requested := ""
	if len(positional) > 0 {
		requested = positional[0]
	}
	cfg, _, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid configuration: %v\n", err)
		return 2
	}
	modelPath, err := resolveModel(requested, cfg, *modelsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not find the model: %v\n", err)
		return 2
	}
	profile := adaptive.LoadProfile(modelPath)
	specs := hardware.Detect()
	resolved := adaptive.ResolveEffort(parsedEffort, *prompt)
	plan := adaptive.Plan(profile, specs, resolved, cfg.ContextSize)
	if *asJSON {
		return printJSON(budgetReport{Model: profile, Hardware: specs, Budget: plan})
	}
	theme := themeFor(*noColor)
	rows := [][2]string{
		{"Model", modelPath},
		{"Architecture", fallbackValue(profile.Architecture, "unknown")},
		{"Parameters", formatCount(profile.Parameters)},
		{"Layers", fmt.Sprintf("%d", profile.Layers)},
		{"Experts", expertSummary(profile)},
		{"File", tui.FormatSize(float64(profile.FileBytes) / (1024 * 1024 * 1024))},
		{"Quantization", fallbackValue(profile.Quantization, "unknown")},
		{"Effort", string(plan.Effort)},
		{"Active params", formatCount(plan.ActiveParameters)},
		{"Parameter budget", formatCount(plan.ParameterBudget)},
		{"Resident params", formatCount(plan.ResidentParameters)},
		{"Streamed params", formatCount(plan.StreamedParameters)},
		{"KV per token", fmt.Sprintf("%d bytes", plan.KVBytesPerToken)},
		{"Context", fmt.Sprintf("%d", plan.ContextSize)},
		{"GPU layers", fmt.Sprintf("%d / %d", plan.GPULayers, profile.Layers)},
		{"MoE placement", moePlacement(plan)},
		{"Batch", fmt.Sprintf("%d / %d", plan.BatchSize, plan.UBatchSize)},
		{"KV cache", plan.KVCacheType},
		{"Load mode", plan.LoadMode},
		{"Risk", plan.Risk},
	}
	fmt.Println(tui.RenderKV("Adaptive parameter budget", rows, theme, "  "))
	fmt.Printf("\n  %s %s  %s\n", theme.Paint("Residency", tui.Dim), capacityBar(plan), theme.Paint(strings.ToUpper(plan.Risk), riskColor(plan.Risk)))
	fmt.Printf("%s\n  %s\n", theme.Paint("Why:", tui.Dim), plan.Reason)
	return 0
}

func fallbackValue(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func expertSummary(profile adaptive.ModelProfile) string {
	if profile.Experts <= 0 {
		return "dense"
	}
	return fmt.Sprintf("%d experts, %d active per token", profile.Experts, profile.ExpertsUsed)
}

func moePlacement(plan adaptive.Budget) string {
	switch {
	case plan.CPUMoE:
		return "experts on CPU, attention on GPU"
	case plan.NCPUMoE > 0:
		return fmt.Sprintf("first %d layers' experts on CPU", plan.NCPUMoE)
	default:
		return "device default"
	}
}

func formatCount(value int64) string {
	switch {
	case value >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(value)/1e9)
	case value >= 1_000_000:
		return fmt.Sprintf("%.0fM", float64(value)/1e6)
	case value > 0:
		return fmt.Sprintf("%d", value)
	default:
		return "0"
	}
}

func capacityBar(plan adaptive.Budget) string {
	const width = 28
	filled := width
	if plan.TotalParameters > 0 {
		filled = int(float64(plan.ResidentParameters) / float64(plan.TotalParameters) * width)
	}
	filled = clampInt(filled, 0, width)
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", width-filled) + "]"
}

func riskColor(risk string) string {
	switch risk {
	case "streaming":
		return tui.Red
	case "tight":
		return tui.Yellow
	default:
		return tui.Green
	}
}

func clampInt(value, low, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
