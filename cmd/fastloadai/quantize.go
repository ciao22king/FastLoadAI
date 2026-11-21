package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
)

func commandQuantize(args []string) int {
	positional, flags := splitArgs(args, map[string]bool{"--type": true})
	set := flag.NewFlagSet("quantize", flag.ContinueOnError)
	quantType := set.String("type", "Q4_K_M", "llama.cpp quantization type")
	if !parseFlags(set, flags) {
		return 2
	}
	if len(positional) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: fastloadai quantize <input.gguf> <output.gguf> [--type Q4_K_M]")
		return 2
	}
	input, output := positional[0], positional[1]
	if info, err := os.Stat(input); err != nil || !info.Mode().IsRegular() {
		fmt.Fprintf(os.Stderr, "Model file not found: %s\n", input)
		return 2
	}
	executable, err := exec.LookPath("llama-quantize")
	if err != nil {
		fmt.Fprintln(os.Stderr, "llama-quantize not found. Install a llama.cpp build that includes the CLI tools.")
		return 2
	}
	command := exec.Command(executable, input, output, *quantType)
	command.Stdin = os.Stdin
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := command.Run(); err != nil {
		if exit, ok := err.(*exec.ExitError); ok {
			return exit.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "Could not run llama-quantize: %v\n", err)
		return 1
	}
	return 0
}
