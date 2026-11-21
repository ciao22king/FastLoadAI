package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/hf"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func commandDoctor(args []string) int {
	positional, flags := splitArgs(args, map[string]bool{"--config": true})
	_ = positional
	set := flag.NewFlagSet("doctor", flag.ContinueOnError)
	configPath := set.String("config", defaultConfig, "config file")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, flags) {
		return 2
	}
	theme := themeFor(*noColor)
	healthy := true
	rows := [][2]string{
		{"Platform", runtime.GOOS + "/" + runtime.GOARCH},
		{"Go", runtime.Version() + " " + theme.Paint("ok", tui.Green)},
	}
	specs := hardware.Detect()
	if specs.CPUName != "" {
		rows = append(rows, [2]string{"CPU", specs.CPUName})
	}
	memory := tui.FormatSize(specs.RAMGB)
	if specs.AvailableRAMGB > 0 {
		memory += " (" + tui.FormatSize(specs.AvailableRAMGB) + " free)"
	}
	rows = append(rows, [2]string{"System RAM", memory})
	if specs.DiskFreeGB > 0 {
		rows = append(rows, [2]string{"Disk free", tui.FormatSize(specs.DiskFreeGB)})
	}
	if specs.GPUName != "" {
		kind := "VRAM"
		if specs.UnifiedMemory {
			kind = "unified memory"
		}
		rows = append(rows, [2]string{"GPU", fmt.Sprintf("%s (%s %s)", specs.GPUName, tui.FormatSize(specs.GPUVramGB), kind)})
	} else {
		rows = append(rows, [2]string{"GPU", "none - CPU only"})
	}
	rows = append(rows, [2]string{"Model budget", tui.FormatSize(hardware.UsableModelMemoryGB(specs))})

	server, serverErr := exec.LookPath("llama-server")
	if serverErr != nil {
		server = theme.Paint("missing - install llama.cpp", tui.Red)
		healthy = false
	}
	rows = append(rows, [2]string{"llama-server", server})
	quantizer, quantErr := exec.LookPath("llama-quantize")
	if quantErr != nil {
		quantizer = theme.Paint("optional, missing", tui.Yellow)
	}
	rows = append(rows, [2]string{"llama-quantize", quantizer})

	client := &http.Client{Timeout: 5 * time.Second}
	if response, err := client.Head("https://huggingface.co"); err != nil {
		rows = append(rows, [2]string{"Hugging Face", theme.Paint("unreachable", tui.Yellow)})
	} else {
		response.Body.Close()
		rows = append(rows, [2]string{"Hugging Face", theme.Paint("reachable", tui.Green)})
	}

	cfg, exists, err := loadConfig(*configPath)
	if err != nil {
		rows = append(rows, [2]string{"Config", theme.Paint(err.Error(), tui.Red)})
		healthy = false
	} else if !exists {
		rows = append(rows, [2]string{"Config", theme.Paint("not found - run download or plan", tui.Yellow)})
	} else {
		rows = append(rows, [2]string{"Config", *configPath + " " + theme.Paint("ok", tui.Green)})
		if cfg.ModelPath != "" {
			if info, statErr := os.Stat(cfg.ModelPath); statErr == nil && info.Mode().IsRegular() {
				rows = append(rows, [2]string{"Model", cfg.ModelPath + " " + theme.Paint("ok", tui.Green)})
				fit := hardware.ModelFit(specs, float64(info.Size())/(1024*1024*1024))
				rows = append(rows, [2]string{"Model fit", theme.Paint(fit, fitStyle(fit))})
			} else {
				rows = append(rows, [2]string{"Model", cfg.ModelPath + " " + theme.Paint("missing", tui.Red)})
				healthy = false
			}
		}
	}
	_ = hf.DefaultBaseURL
	fmt.Println(tui.RenderKV("Environment check", rows, theme, "  "))
	if healthy {
		return 0
	}
	return 1
}
