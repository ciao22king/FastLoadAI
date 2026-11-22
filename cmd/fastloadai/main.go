// Command fastloadai prepares GGUF models for the local machine and chats with them.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/config"
	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

const defaultConfig = config.DefaultPath

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		printUsage()
		return 2
	}
	switch args[0] {
	case "-h", "--help", "help":
		printUsage()
		return 0
	case "-v", "--version", "version":
		fmt.Printf("fastloadai %s (%s/%s)\n", tui.Version, runtime.GOOS, runtime.GOARCH)
		return 0
	case "scan":
		return commandScan(args[1:])
	case "plan":
		return commandPlan(args[1:])
	case "download":
		return commandDownload(args[1:])
	case "suggest":
		return commandSuggest(args[1:])
	case "models":
		return commandModels(args[1:])
	case "doctor":
		return commandDoctor(args[1:])
	case "run":
		return commandRun(args[1:])
	case "quantize":
		return commandQuantize(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q. Run fastloadai --help.\n", args[0])
		return 2
	}
}

func printUsage() {
	fmt.Print(`FastLoadAI - local GGUF models for your hardware

Usage:
  fastloadai <command> [options]

Commands:
  scan       Detect CPU, RAM, GPU, disk and the model memory budget.
  suggest    Recommend catalog models that fit this machine.
  download   Download a GGUF quantization from Hugging Face.
  plan       Build a runtime plan and write the config file.
  run        Download if needed, then open the full-screen chat.
  models     List downloaded GGUF models.
  doctor     Check the runtime, model files and configuration.
  quantize   Quantize a GGUF file with llama-quantize.

Examples:
  fastloadai scan
  fastloadai suggest
  fastloadai run qwen2.5-0.5b
  fastloadai run Qwen/Qwen2.5-0.5B-Instruct-GGUF --prompt "Hello"
  fastloadai run qwen2.5-0.5b --no-tui
  fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --dry-run
  fastloadai download Qwen/Qwen2.5-7B-Instruct-GGUF --workers 8
  fastloadai models --json
`)
}

func themeFor(noColor bool) tui.Theme {
	return tui.Theme{Enabled: !noColor && tui.ColorEnabled()}
}

func parseFlags(set *flag.FlagSet, args []string) bool {
	set.SetOutput(os.Stderr)
	if err := set.Parse(args); err != nil {
		return false
	}
	return true
}

func printJSON(value any) int {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not encode JSON: %v\n", err)
		return 1
	}
	fmt.Println(string(data))
	return 0
}

func tokenValue(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	if value := os.Getenv("HF_TOKEN"); value != "" {
		return value
	}
	return os.Getenv("HUGGING_FACE_HUB_TOKEN")
}

func absModelPath(value string) string {
	if value == "" {
		return ""
	}
	absolute, err := filepath.Abs(value)
	if err != nil {
		return value
	}
	return absolute
}

func specsAndConfig(cfg model.Config) model.Config {
	specs := hardware.Detect()
	cfg.Hardware = specs
	return cfg
}

func loadConfig(path string) (model.Config, bool, error) {
	return config.Load(path)
}

func saveConfig(path string, cfg model.Config) error {
	return config.Write(path, cfg)
}

func lower(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

// splitArgs separates positionals from flags so a model or repo can appear anywhere.
func splitArgs(args []string, valueFlags map[string]bool) ([]string, []string) {
	var positionals []string
	var flags []string
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if !strings.HasPrefix(arg, "-") {
			positionals = append(positionals, arg)
			continue
		}
		flags = append(flags, arg)
		name := arg
		if cut := strings.IndexByte(arg, '='); cut >= 0 {
			name = arg[:cut]
		}
		if valueFlags[name] && !strings.Contains(arg, "=") && index+1 < len(args) {
			index++
			flags = append(flags, args[index])
		}
	}
	return positionals, flags
}
