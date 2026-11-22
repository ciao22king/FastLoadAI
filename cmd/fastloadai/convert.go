package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/ciao22king/FastLoadAI/internal/safetensors"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func commandConvert(args []string) int {
	positional, flags := splitArgs(args, map[string]bool{"--output": true})
	set := flag.NewFlagSet("convert", flag.ContinueOnError)
	output := set.String("output", "", "output GGUF file")
	asJSON := set.Bool("json", false, "print JSON")
	noColor := set.Bool("no-color", false, "disable color")
	if !parseFlags(set, flags) {
		return 2
	}
	if len(positional) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: fastloadai convert <safetensors-dir> [--output model.gguf]")
		return 2
	}
	dir := absModelPath(positional[0])
	if !safetensors.IsModelDir(dir) {
		fmt.Fprintf(os.Stderr, "%s is not a safetensors model directory: it needs config.json and *.safetensors\n", dir)
		return 2
	}
	target := *output
	if target == "" {
		base := filepath.Base(filepath.Clean(dir))
		target = filepath.Join(dir, base+"-converted.gguf")
	}
	target = absModelPath(target)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "Could not create %s: %v\n", filepath.Dir(target), err)
		return 1
	}
	result, err := safetensors.Convert(dir, target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Conversion failed: %v\n", err)
		return 1
	}
	if *asJSON {
		return printJSON(result)
	}
	theme := themeFor(*noColor)
	rows := [][2]string{
		{"Source", dir},
		{"Output", result.Path},
		{"Architecture", result.Arch},
		{"Tensors", fmt.Sprintf("%d", result.Tensors)},
		{"Parameters", formatCount(result.Parameters)},
		{"Size", tui.FormatSize(float64(result.Bytes) / (1024 * 1024 * 1024))},
	}
	fmt.Println(tui.RenderKV("Safetensors converted", rows, theme, "  "))
	fmt.Printf("\n  %s\n", theme.Paint("The GGUF file is self-contained: tokenizer, chat template and weights are embedded.", tui.Dim))
	return 0
}
