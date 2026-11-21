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
	"github.com/ciao22king/FastLoadAI/internal/model"
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

	type lookPathResult struct {
		path string
		err  error
	}
	serverCh := make(chan lookPathResult, 1)
	quantCh := make(chan lookPathResult, 1)
	specsCh := make(chan model.Specs, 1)
	hfCh := make(chan bool, 1)
	go func() { path, err := exec.LookPath("llama-server"); serverCh <- lookPathResult{path, err} }()
	go func() { path, err := exec.LookPath("llama-quantize"); quantCh <- lookPathResult{path, err} }()
	go func() { specsCh <- hardware.Detect() }()
	go func() {
		client := &http.Client{Timeout: 5 * time.Second}
		response, err := client.Head(hf.DefaultBaseURL)
		if err != nil {
			hfCh <- false
			return
		}
		response.Body.Close()
		hfCh <- true
	}()

	specs := <-specsCh
	rows := [][2]string{
		{"Platform", runtime.GOOS + "/" + runtime.GOARCH},
		{"Go", runtime.Version() + " " + theme.Paint("ok", tui.Green)},
	}
	if specs.CPUName != "" {
		rows = append(rows, [2]string{"CPU", specs.CPUName})
	}
	if specs.PhysicalCores > 0 {
		rows = append(rows, [2]string{"CPU cores", fmt.Sprintf("%d physical / %d logical", specs.PhysicalCores, specs.CPUThreads)})
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

	server := <-serverCh
	serverPath := server.path
	if server.err != nil {
		serverPath = theme.Paint("missing - install llama.cpp", tui.Red)
		healthy = false
	}
	rows = append(rows, [2]string{"llama-server", serverPath})
	quantizer := <-quantCh
	quantizerPath := quantizer.path
	if quantizer.err != nil {
		quantizerPath = theme.Paint("optional, missing", tui.Yellow)
	}
	rows = append(rows, [2]string{"llama-quantize", quantizerPath})

	if <-hfCh {
		rows = append(rows, [2]string{"Hugging Face", theme.Paint("reachable", tui.Green)})
	} else {
		rows = append(rows, [2]string{"Hugging Face", theme.Paint("unreachable", tui.Yellow)})
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
