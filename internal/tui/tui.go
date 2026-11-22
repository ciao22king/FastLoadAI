// Package tui renders the FastLoadAI terminal interface.
package tui

import (
	"fmt"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/optimization"
)

// Version is shown in the banner and by --version.
const Version = "1.1.0"

const (
	Reset   = "\033[0m"
	Bold    = "1"
	Dim     = "2"
	Red     = "31"
	Green   = "32"
	Yellow  = "33"
	Blue    = "34"
	Magenta = "35"
	Cyan    = "36"
	Grey    = "90"
	Reverse = "7"
)

var ansiRE = regexp.MustCompile(`\x1b\[[0-9;]*m`)

// Theme applies ANSI styles when colour is enabled.
type Theme struct {
	Enabled bool
}

// Paint styles text, or returns it unchanged when colour is disabled.
func (t Theme) Paint(text string, styles ...string) string {
	if !t.Enabled || len(styles) == 0 || text == "" {
		return text
	}
	return "\033[" + strings.Join(styles, ";") + "m" + text + Reset
}

func isTerminal(file *os.File) bool {
	info, err := file.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// ColorEnabled reports whether ANSI colour should be used.
func ColorEnabled() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	if force := os.Getenv("FORCE_COLOR"); force != "" && force != "0" {
		return true
	}
	if os.Getenv("TERM") == "dumb" {
		return false
	}
	return isTerminal(os.Stdout)
}

// StdinIsTerminal reports whether interactive input is available.
func StdinIsTerminal() bool {
	return isTerminal(os.Stdin)
}

// StdoutIsTerminal reports whether the output is an interactive terminal.
func StdoutIsTerminal() bool {
	return isTerminal(os.Stdout)
}

// Spinner animates a single status line while work is in progress.
type Spinner struct {
	stop   chan struct{}
	done   chan struct{}
	label  string
	theme  Theme
	active bool
}

// StartSpinner starts an animated status line when stdout is a terminal.
func StartSpinner(label string, theme Theme) *Spinner {
	spinner := &Spinner{stop: make(chan struct{}), done: make(chan struct{}), label: label, theme: theme}
	if !isTerminal(os.Stdout) {
		return spinner
	}
	spinner.active = true
	go func() {
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		ticker := time.NewTicker(90 * time.Millisecond)
		defer ticker.Stop()
		index := 0
		for {
			select {
			case <-spinner.stop:
				fmt.Print("\r\033[K")
				close(spinner.done)
				return
			case <-ticker.C:
				fmt.Printf("\r\033[K%s %s", theme.Paint(frames[index%len(frames)], Cyan), theme.Paint(spinner.label, Dim))
				index++
			}
		}
	}()
	return spinner
}

// Stop clears the status line.
func (s *Spinner) Stop() {
	if s == nil || !s.active {
		return
	}
	close(s.stop)
	<-s.done
	s.active = false
}

// VisibleWidth returns the printable length of text without ANSI sequences.
func VisibleWidth(text string) int {
	return len([]rune(ansiRE.ReplaceAllString(text, "")))
}

// TerminalWidth returns a readable terminal width.
func TerminalWidth() int {
	width := 80
	if columns := os.Getenv("COLUMNS"); columns != "" {
		if value, err := strconv.Atoi(columns); err == nil {
			width = value
		}
	}
	if width < 40 {
		return 40
	}
	if width > 96 {
		return 96
	}
	return width
}

// EstimateTokens is the cheap ~4 characters per token estimate.
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return (len([]rune(text)) + 2) / 4
}

// FormatDuration renders milliseconds, seconds or minutes.
func FormatDuration(seconds float64) string {
	if seconds < 0 {
		seconds = 0
	}
	if seconds < 1 {
		return fmt.Sprintf("%.0f ms", seconds*1000)
	}
	if seconds < 60 {
		return fmt.Sprintf("%.1fs", seconds)
	}
	minutes := int(seconds / 60)
	remainder := seconds - float64(minutes*60)
	return fmt.Sprintf("%dm %02.0fs", minutes, remainder)
}

// FormatSize renders gigabytes as MB or GB.
func FormatSize(gigabytes float64) string {
	if gigabytes <= 0 {
		return "unknown"
	}
	if gigabytes < 1 {
		return fmt.Sprintf("%.0f MB", gigabytes*1024)
	}
	return fmt.Sprintf("%.1f GB", gigabytes)
}

func fitText(text string, width int) string {
	if width <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= width {
		return text
	}
	if width == 1 {
		return "…"
	}
	return string(runes[:width-1]) + "…"
}

// RenderKV renders an aligned key/value block.
func RenderKV(title string, rows [][2]string, theme Theme, indent string) string {
	if len(rows) == 0 {
		return theme.Paint(title, Bold)
	}
	labelWidth := 0
	for _, row := range rows {
		if len(row[0]) > labelWidth {
			labelWidth = len(row[0])
		}
	}
	var lines []string
	lines = append(lines, theme.Paint(title, Bold))
	for _, row := range rows {
		label := row[0] + strings.Repeat(" ", max(0, labelWidth-len(row[0])))
		lines = append(lines, indent+theme.Paint(label, Cyan)+"  "+row[1])
	}
	return strings.Join(lines, "\n")
}

// fitDetails drops trailing details instead of truncating the whole line.
func fitDetails(details []string, width int) string {
	for len(details) > 0 {
		text := strings.Join(details, " · ")
		if VisibleWidth(text) <= width {
			return text
		}
		details = details[:len(details)-1]
	}
	return ""
}

// RenderBanner renders the boxed header shown at chat startup.
func RenderBanner(modelName string, config model.Config, theme Theme, width int) string {
	if width == 0 {
		width = TerminalWidth()
	}
	if width < 36 {
		width = 36
	}
	inner := width - 4
	var details []string
	var advanced []string
	if config.OptimizationProfile != "" && config.OptimizationProfile != "auto" {
		details = append(details, "profile "+config.OptimizationProfile)
	}
	if config.Quantization != "" {
		details = append(details, config.Quantization)
	}
	if config.ContextSize != 0 {
		details = append(details, fmt.Sprintf("ctx %d", config.ContextSize))
	}
	if config.CPUThreads != 0 {
		details = append(details, fmt.Sprintf("%d thr", config.CPUThreads))
	}
	if config.NGPULayers == 0 {
		details = append(details, "CPU")
	} else {
		details = append(details, fmt.Sprintf("GPU %d", config.NGPULayers))
	}
	if config.Mmap {
		details = append(details, "mmap")
	} else {
		details = append(details, "no-mmap")
	}
	if config.LowMemory {
		details = append(details, "low-mem")
	}
	if config.FlashAttention == "on" {
		details = append(details, "FA")
	}
	kv := config.KVCacheTypeK
	if kv == "" {
		kv = config.KVCacheTypeV
	} else if config.KVCacheTypeV != "" && config.KVCacheTypeV != kv {
		kv += "/" + config.KVCacheTypeV
	}
	if kv != "" && kv != "f16" {
		details = append(details, "KV "+kv)
	}
	if config.KVUnified != nil {
		if *config.KVUnified {
			advanced = append(advanced, "KV unified")
		} else {
			advanced = append(advanced, "KV per-slot")
		}
	}
	if config.CacheRAMMiB != nil && *config.CacheRAMMiB == 0 {
		advanced = append(advanced, "cache off")
	}
	if config.SlotSavePath != "" {
		advanced = append(advanced, "slots")
	}
	if config.RopeScaling != "" {
		detail := "rope " + config.RopeScaling
		if config.RopeScale > 0 {
			detail += fmt.Sprintf(" x%.0f", config.RopeScale)
		}
		advanced = append(advanced, detail)
	}
	if config.CPURange != "" {
		advanced = append(advanced, "P-cores "+config.CPURange)
	} else if config.CPUMask != "" {
		advanced = append(advanced, "P-cores")
	}
	if config.SWACheckpoints != nil && *config.SWACheckpoints > 0 {
		advanced = append(advanced, fmt.Sprintf("SWA %d", *config.SWACheckpoints))
	}
	if config.NoRepack {
		advanced = append(advanced, "no-repack")
	}
	if config.NoWarmup {
		advanced = append(advanced, "no-warmup")
	}
	if config.MoE {
		details = append(details, "MoE")
	}
	if config.ModelSizeGB > 0 {
		details = append(details, FormatSize(config.ModelSizeGB))
	}
	row := func(text string, styles ...string) string {
		fitted := fitText(text, inner)
		padding := strings.Repeat(" ", max(0, inner-VisibleWidth(fitted)))
		return theme.Paint("│", Grey) + " " + theme.Paint(fitted, styles...) + padding + " " + theme.Paint("│", Grey)
	}
	lines := []string{
		theme.Paint("╭"+strings.Repeat("─", inner)+"╮", Grey),
		row("FastLoadAI  ·  local GGUF chat", Bold, Cyan),
		row("v"+Version+"  ·  hardware-aware, privacy-first", Dim),
		theme.Paint("├"+strings.Repeat("─", inner)+"┤", Grey),
		row("model   "+modelName, Bold),
		row(fitDetails(details, inner), Dim),
	}
	if extra := fitDetails(advanced, inner); extra != "" {
		lines = append(lines, row(extra, Dim))
	}
	lines = append(lines,
		row("/help commands  ·  /exit quit", Dim),
		theme.Paint("╰"+strings.Repeat("─", inner)+"╯", Grey),
	)
	return strings.Join(lines, "\n")
}

var helpEntries = [][2]string{
	{"/help", "Show this help"},
	{"/clear", "Reset the conversation history"},
	{"/stats", "Show session statistics"},
	{"/context", "Show context window usage"},
	{"/retry", "Regenerate the last answer"},
	{"/exit", "Close the chat and stop the server"},
}

// RenderHelp renders the in-chat command list.
func RenderHelp(theme Theme) string {
	width := 0
	for _, entry := range helpEntries {
		width = max(width, len(entry[0]))
	}
	lines := []string{theme.Paint("Commands", Bold)}
	for _, entry := range helpEntries {
		lines = append(lines, "  "+theme.Paint(entry[0]+strings.Repeat(" ", width-len(entry[0])), Cyan)+"  "+entry[1])
	}
	lines = append(lines, "", theme.Paint("Ctrl+C or Ctrl+D also ends the chat.", Dim))
	return strings.Join(lines, "\n")
}

// ChatStats accumulates usage for the current chat.
type ChatStats struct {
	Turns           int
	PromptTokens    int
	GeneratedTokens int
	Elapsed         time.Duration
}

// Record adds one completed turn.
func (s *ChatStats) Record(promptTokens, answerTokens int, elapsed time.Duration) {
	s.Turns++
	s.PromptTokens += promptTokens
	s.GeneratedTokens += answerTokens
	s.Elapsed += elapsed
}

// TokensPerSecond returns the average generation speed.
func (s ChatStats) TokensPerSecond() float64 {
	if s.Elapsed <= 0 {
		return 0
	}
	return float64(s.GeneratedTokens) / s.Elapsed.Seconds()
}

// RenderSummary renders a compact end-of-session summary.
func (s ChatStats) RenderSummary(theme Theme) string {
	if s.Turns == 0 {
		return ""
	}
	return RenderKV("Session summary", [][2]string{
		{"turns", strconv.Itoa(s.Turns)},
		{"tokens", fmt.Sprintf("~%d in / ~%d out", s.PromptTokens, s.GeneratedTokens)},
		{"speed", fmt.Sprintf("~%.1f tok/s", s.TokensPerSecond())},
		{"time", FormatDuration(s.Elapsed.Seconds())},
	}, theme, "  ")
}

// RenderStats renders the /stats block.
func (s ChatStats) RenderStats(theme Theme) string {
	return RenderKV("Session", [][2]string{
		{"turns", strconv.Itoa(s.Turns)},
		{"prompt", fmt.Sprintf("~%d tokens", s.PromptTokens)},
		{"generated", fmt.Sprintf("~%d tokens", s.GeneratedTokens)},
		{"speed", fmt.Sprintf("~%.1f tok/s", s.TokensPerSecond())},
		{"time", FormatDuration(s.Elapsed.Seconds())},
	}, theme, "  ")
}

// ConversationTokens estimates the tokens held in the conversation.
func ConversationTokens(messages []model.Message) int {
	total := 0
	for _, message := range messages {
		total += EstimateTokens(message.Content)
	}
	return total
}

// RenderContext renders context-window usage.
func RenderContext(used, limit int, theme Theme) string {
	if limit <= 0 {
		return RenderKV("Context", [][2]string{
			{"used", fmt.Sprintf("~%d tokens", used)},
			{"window", "unknown"},
		}, theme, "  ")
	}
	ratio := float64(used) / float64(limit)
	if ratio > 1 {
		ratio = 1
	}
	if ratio < 0 {
		ratio = 0
	}
	width := 24
	filled := int(ratio*float64(width) + 0.5)
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	style := Green
	if ratio >= 0.9 {
		style = Red
	} else if ratio >= 0.7 {
		style = Yellow
	}
	return RenderKV("Context", [][2]string{
		{"used", fmt.Sprintf("~%d / %d tokens", used, limit)},
		{"window", theme.Paint(bar, style) + fmt.Sprintf(" %.0f%%", ratio*100)},
	}, theme, "  ")
}

// RenderTiming renders the speed line after an answer.
func RenderTiming(tokens int, elapsed time.Duration, contextSize, conversationTokens int, theme Theme) string {
	rate := 0.0
	if elapsed > 0 {
		rate = float64(tokens) / elapsed.Seconds()
	}
	style := Red
	if rate >= 10 {
		style = Green
	} else if rate >= 3 {
		style = Yellow
	}
	detail := fmt.Sprintf("  %s · ~%d tokens · ", FormatDuration(elapsed.Seconds()), tokens)
	line := theme.Paint(detail, Dim) + theme.Paint(fmt.Sprintf("~%.0f tok/s", rate), style)
	if contextSize > 0 {
		line += theme.Paint(fmt.Sprintf(" · ctx %d/%d", conversationTokens, contextSize), Dim)
	}
	return line
}

// RenderHardware renders the scan table.
func RenderHardware(specs model.Specs, theme Theme) string {
	memory := FormatSize(specs.RAMGB)
	if specs.AvailableRAMGB > 0 {
		memory += " (" + FormatSize(specs.AvailableRAMGB) + " free)"
	}
	rows := [][2]string{{"Platform", runtime.GOOS + "/" + runtime.GOARCH}}
	if specs.CPUName != "" {
		rows = append(rows, [2]string{"CPU", fitText(specs.CPUName, 48)})
	}
	if specs.PhysicalCores > 0 {
		rows = append(rows, [2]string{"CPU cores", strconv.Itoa(specs.PhysicalCores)})
	}
	if specs.Hybrid {
		detail := fmt.Sprintf("%d performance / %d efficiency", specs.PerformanceThreads, specs.EfficiencyThreads)
		if specs.PerformanceCPUs != "" {
			detail += " · P CPUs " + specs.PerformanceCPUs
		}
		rows = append(rows, [2]string{"Hybrid CPU", detail})
	}
	rows = append(rows,
		[2]string{"CPU threads", strconv.Itoa(specs.CPUThreads)},
		[2]string{"System RAM", memory},
	)
	if specs.DiskFreeGB > 0 {
		rows = append(rows, [2]string{"Disk free", FormatSize(specs.DiskFreeGB)})
	}
	if specs.GPUName != "" {
		kind := "VRAM"
		if specs.UnifiedMemory {
			kind = "unified memory"
		}
		rows = append(rows, [2]string{"GPU", fmt.Sprintf("%s (%s %s)", specs.GPUName, FormatSize(specs.GPUVramGB), kind)})
	} else {
		rows = append(rows, [2]string{"GPU", "none (CPU only)"})
	}
	if specs.IsLaptop {
		rows = append(rows, [2]string{"Form factor", "laptop"})
	}
	rows = append(rows, [2]string{"Model budget", FormatSize(hardware.UsableModelMemoryGB(specs))})
	return RenderKV("Hardware detected", rows, theme, "  ")
}

// RenderSuggestions renders the catalog table.
func RenderSuggestions(rows []SuggestionRow, budget float64, theme Theme) string {
	lines := []string{
		theme.Paint("Suggested models", Bold, Cyan),
		theme.Paint("  memory budget "+FormatSize(budget)+"  ·  * recommended for this machine", Dim),
		"",
		"  " + theme.Paint(fmt.Sprintf("%-30s %8s  %-12s %s", "MODEL", "SIZE", "FIT", "NOTE"), Bold),
	}
	for _, row := range rows {
		style := Red
		if row.Fit == "comfortable" {
			style = Green
		} else if row.Fit == "tight" {
			style = Yellow
		}
		marker := " "
		if row.Recommended {
			marker = theme.Paint("*", Bold, Green)
		}
		lines = append(lines,
			fmt.Sprintf("  %s %-30s %8s  %s %s", marker, fitText(row.Name, 30), FormatSize(row.SizeGB), theme.Paint(fmt.Sprintf("%-12s", row.Fit), style), theme.Paint(row.Note, Dim)),
			"    "+theme.Paint(row.Repo, Dim),
		)
	}
	return strings.Join(lines, "\n")
}

// SuggestionRow is the rendering shape of a catalog row.
type SuggestionRow struct {
	Name        string
	Repo        string
	SizeGB      float64
	Note        string
	Fit         string
	Recommended bool
}

// ModelRow is one downloaded GGUF model.
type ModelRow struct {
	Name         string
	Path         string
	SizeGB       float64
	Quantization string
	Fit          string
}

// RenderModels renders the downloaded model table.
func RenderModels(rows []ModelRow, root string, theme Theme) string {
	if len(rows) == 0 {
		return RenderKV("Local models", [][2]string{{"directory", root}, {"models", "none found"}}, theme, "  ")
	}
	lines := []string{
		theme.Paint("Local models", Bold, Cyan),
		theme.Paint("  "+root, Dim),
		"",
		"  " + theme.Paint(fmt.Sprintf("%-34s %8s  %-10s %s", "FILE", "SIZE", "QUANT", "FIT"), Bold),
	}
	for _, row := range rows {
		style := Red
		if row.Fit == "comfortable" {
			style = Green
		} else if row.Fit == "tight" {
			style = Yellow
		}
		lines = append(lines, fmt.Sprintf("  %-34s %8s  %-10s %s", fitText(row.Path, 34), FormatSize(row.SizeGB), fitText(row.Quantization, 10), theme.Paint(row.Fit, style)))
	}
	return strings.Join(lines, "\n")
}

// RenderPlan renders the saved runtime plan.
func RenderPlan(config model.Config, output string, theme Theme) string {
	modelPath := config.ModelPath
	if modelPath == "" {
		modelPath = "not specified"
	}
	mmap := "off"
	if config.Mmap {
		mmap = "on"
	}
	rows := [][2]string{
		{"Model", modelPath},
		{"Model size", FormatSize(config.ModelSizeGB)},
		{"Model fit", config.ModelFit},
		{"Context", strconv.Itoa(config.ContextSize)},
		{"Threads", strconv.Itoa(config.CPUThreads)},
		{"GPU layers", strconv.Itoa(config.NGPULayers)},
		{"mmap", mmap},
		{"Memory budget", FormatSize(config.UsableModelMemoryGB)},
	}
	if config.LowMemory {
		rows = append(rows, [2]string{"Low memory mode", "on"})
	}
	rows = append(rows, optimization.SummaryRows(config)...)
	rows = append(rows, [2]string{"Config file", output})
	return RenderKV("Runtime plan", rows, theme, "  ")
}
