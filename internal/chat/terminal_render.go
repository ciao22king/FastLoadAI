package chat

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func (ui *terminalChat) screen() []string {
	header := ui.headerLines()
	input := ui.inputLines()
	conversationHeight := ui.height - len(header) - 3 - len(input)
	if conversationHeight < 1 {
		conversationHeight = 1
	}
	conversation := ui.conversationLines()
	maxScroll := max(0, len(conversation)-conversationHeight)
	if ui.scroll > maxScroll {
		ui.scroll = maxScroll
	}
	if ui.scroll < 0 {
		ui.scroll = 0
	}
	start := len(conversation) - conversationHeight - ui.scroll
	if start < 0 {
		start = 0
	}
	end := start + conversationHeight
	if end > len(conversation) {
		end = len(conversation)
	}
	view := append([]string(nil), conversation[start:end]...)
	for len(view) < conversationHeight {
		view = append(view, "")
	}

	lines := append([]string(nil), header...)
	lines = append(lines, ui.divider())
	lines = append(lines, view...)
	lines = append(lines, ui.divider())
	lines = append(lines, input...)
	lines = append(lines, ui.statusLine())
	return lines
}

func (ui *terminalChat) headerLines() []string {
	title := ui.renderSegments([]segment{
		{" FastLoadAI ", []string{tui.Bold, tui.Cyan}},
		{"· " + filepath.Base(ui.config.ModelPath), []string{tui.Dim}},
	}, ui.width)
	quant := displayQuant(ui.config.ModelPath, ui.config.Quantization)
	device := "CPU"
	if ui.config.NGPULayers > 0 {
		device = fmt.Sprintf("GPU %d", ui.config.NGPULayers)
	}
	parts := []string{}
	if ui.config.OptimizationProfile != "" && ui.config.OptimizationProfile != "auto" {
		parts = append(parts, "profile "+ui.config.OptimizationProfile)
	}
	parts = append(parts, quant, fmt.Sprintf("ctx %d", ui.config.ContextSize), fmt.Sprintf("%d thr", ui.config.CPUThreads), device)
	if ui.config.Mmap {
		parts = append(parts, "mmap")
	} else {
		parts = append(parts, "no-mmap")
	}
	if ui.config.LowMemory {
		parts = append(parts, "low-mem")
	}
	if ui.config.FlashAttention == "on" {
		parts = append(parts, "FA")
	}
	if ui.config.KVCacheTypeK != "" && ui.config.KVCacheTypeK != "f16" {
		parts = append(parts, "KV "+ui.config.KVCacheTypeK)
	}
	if ui.config.MoE {
		parts = append(parts, "MoE")
	}
	if ui.config.ModelSizeBytes > 0 {
		parts = append(parts, tui.FormatSize(float64(ui.config.ModelSizeBytes)/(1024*1024*1024)))
	}
	if ui.tools != nil {
		parts = append(parts, fmt.Sprintf("tools %d", len(ui.tools.Names())))
	}
	detail := " " + fitHeaderDetails(parts, ui.width-1)
	return []string{title, ui.renderSegments([]segment{{detail, []string{tui.Dim}}}, ui.width)}
}

func displayQuant(modelPath, quant string) string {
	base := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
	if quant == "" || quant == base {
		return "GGUF"
	}
	return quant
}

func fitHeaderDetails(parts []string, width int) string {
	for len(parts) > 0 {
		text := strings.Join(parts, " · ")
		if tui.VisibleWidth(text) <= width {
			return text
		}
		parts = parts[:len(parts)-1]
	}
	return ""
}

func (ui *terminalChat) divider() string {
	return ui.theme.Paint(strings.Repeat("─", ui.width), tui.Grey)
}

func (ui *terminalChat) statusLine() string {
	left := fmt.Sprintf(" %d turns · ~%d/%d ctx", ui.stats.Turns, tui.ConversationTokens(ui.messages), ui.config.ContextSize)
	if ui.stats.Turns > 0 {
		left += fmt.Sprintf(" · %.0f tok/s", ui.stats.TokensPerSecond())
	}
	if ui.busy && !ui.started.IsZero() && ui.streaming != "" {
		elapsed := time.Since(ui.started).Seconds()
		if elapsed > 0 {
			left += fmt.Sprintf(" · %.0f tok/s", float64(tui.EstimateTokens(ui.streaming))/elapsed)
		}
	}
	if ui.tools != nil && ui.toolRounds > 0 {
		left += fmt.Sprintf(" · tools %d/%d", ui.toolRounds, ui.maxToolRounds)
	}
	if ui.scroll > 0 {
		left += fmt.Sprintf(" · ↑%d", ui.scroll)
	}
	right := " Enter send · Ctrl+J newline · /help · /copy "
	if ui.width < 110 {
		right = " Enter send · Ctrl+J · /help · /copy "
	}
	if ui.width < 90 {
		right = " Enter send · /help · /copy "
	}
	if ui.width < 70 {
		right = " Enter send · Ctrl+C "
	}
	style := tui.Dim
	if ui.busy {
		frame := spinnerFrames[ui.spinner%len(spinnerFrames)]
		label := "streaming…"
		if ui.streaming == "" {
			label = "thinking…"
		}
		right = " " + frame + " " + label + " · Ctrl+C cancel "
		if ui.width < 80 {
			right = " " + frame + " " + label + " "
		}
		style = tui.Yellow
	}
	return ui.theme.Paint(padBetween(left, right, ui.width), style)
}

func (ui *terminalChat) inputLines() []string {
	contentWidth := max(4, ui.width-4)
	lines, cursorRow, cursorRune := wrapInput(ui.input, ui.cursor, contentWidth)
	maxVisible := ui.maxInputLines()
	if len(lines) > maxVisible {
		start := max(0, cursorRow-maxVisible+1)
		if start+maxVisible > len(lines) {
			start = len(lines) - maxVisible
		}
		lines = lines[start:]
		cursorRow -= start
	}
	top := ui.theme.Paint("╭─ message "+strings.Repeat("─", max(0, ui.width-12))+"╮", tui.Grey)
	bottom := ui.theme.Paint("╰"+strings.Repeat("─", max(0, ui.width-2))+"╯", tui.Grey)
	out := []string{top}
	for index, line := range lines {
		cursor := -1
		if index == cursorRow {
			cursor = cursorRune
		}
		out = append(out, "│ "+ui.renderInputLine(line, cursor, contentWidth)+" │")
	}
	out = append(out, bottom)
	return out
}

func (ui *terminalChat) renderInputLine(line []rune, cursor, width int) string {
	var builder strings.Builder
	used := 0
	for index, r := range line {
		runes := runeWidth(r)
		if used+runes > width {
			break
		}
		if cursor >= 0 && index == cursor {
			builder.WriteString("\033[" + tui.Reverse + "m" + string(r) + tui.Reset)
		} else {
			builder.WriteString(string(r))
		}
		used += runes
	}
	if cursor >= 0 && cursor >= len(line) && used < width {
		builder.WriteString("\033[" + tui.Reverse + "m \033[0m")
		used++
	}
	if used < width {
		builder.WriteString(strings.Repeat(" ", width-used))
	}
	return builder.String()
}

func (ui *terminalChat) conversationLines() []string {
	if ui.renderedWidth != ui.width || ui.renderedVersion != ui.transcriptVersion {
		ui.rendered = nil
		for _, item := range ui.transcript {
			ui.rendered = append(ui.rendered, ui.renderItem(item)...)
			ui.rendered = append(ui.rendered, "")
		}
		ui.renderedWidth = ui.width
		ui.renderedVersion = ui.transcriptVersion
	}
	if len(ui.transcript) == 0 && !ui.busy {
		return ui.welcomeLines()
	}
	out := ui.rendered
	if ui.busy {
		out = append(append([]string(nil), out...), ui.renderAssistant(ui.streaming, true)...)
		out = append(out, "")
	}
	return out
}

func (ui *terminalChat) welcomeLines() []string {
	width := ui.width
	if width > 88 {
		width = 88
	}
	inner := max(12, width-6)
	device := "CPU"
	if ui.config.NGPULayers > 0 {
		device = fmt.Sprintf("GPU %d", ui.config.NGPULayers)
	}
	model := fmt.Sprintf("%s · %s · ctx %d · %s", filepath.Base(ui.config.ModelPath), displayQuant(ui.config.ModelPath, ui.config.Quantization), ui.config.ContextSize, device)
	rows := [][2]string{
		{"FastLoadAI is ready", ""},
		{"model", model},
	}
	if ui.tools != nil {
		rows = append(rows, [2]string{"tools", fmt.Sprintf("%d enabled · /tools shows the workspace root", len(ui.tools.Names()))})
	}
	rows = append(rows,
		[2]string{"ask", "Type a question. Paste text with Ctrl+Shift+V."},
		[2]string{"help", "/help commands · Tab completes · ↑/↓ history"},
		[2]string{"save", "/save writes Markdown · /copy copies the answer"},
	)
	out := []string{"  " + ui.theme.Paint("╭"+strings.Repeat("─", max(2, width-4))+"╮", tui.Grey)}
	for index, row := range rows {
		style := []string{tui.Dim}
		if index == 0 {
			style = []string{tui.Bold, tui.Cyan}
		}
		content := row[0]
		if row[1] != "" {
			content = fmt.Sprintf("%-7s %s", row[0], row[1])
		}
		out = append(out, "  "+ui.theme.Paint("│", tui.Grey)+" "+ui.renderSegments([]segment{{content, style}}, inner)+" "+ui.theme.Paint("│", tui.Grey))
	}
	out = append(out, "  "+ui.theme.Paint("╰"+strings.Repeat("─", max(2, width-4))+"╯", tui.Grey))
	return out
}

func (ui *terminalChat) renderItem(item chatItem) []string {
	switch item.role {
	case "user":
		return ui.renderLabeled("you › ", []string{tui.Bold, tui.Cyan}, item.content, nil)
	case "assistant":
		return ui.renderAssistant(item.content, false)
	case "tool":
		parts := strings.SplitN(item.content, "\n", 2)
		out := ui.renderLabeled("tool › ", []string{tui.Bold, tui.Cyan}, parts[0], nil)
		if len(parts) == 2 {
			out = append(out, ui.renderLabeled("       ", nil, parts[1], []string{item.style})...)
		}
		return out
	default:
		return ui.renderInfo(item.content, item.style)
	}
}

func (ui *terminalChat) renderLabeled(label string, labelStyles []string, content string, contentStyles []string) []string {
	labelWidth := tui.VisibleWidth(label)
	contentWidth := ui.width - labelWidth
	indent := strings.Repeat(" ", labelWidth)
	if contentWidth < 10 {
		label = "› "
		labelWidth = 2
		indent = "  "
		contentWidth = ui.width - 2
	}
	raw := wrapPlain(content, max(4, contentWidth))
	out := make([]string, 0, len(raw))
	for index, line := range raw {
		if index == 0 {
			out = append(out, ui.renderSegments([]segment{{label, labelStyles}, {line, contentStyles}}, ui.width))
		} else {
			out = append(out, ui.renderSegments([]segment{{indent, nil}, {line, contentStyles}}, ui.width))
		}
	}
	return out
}

func (ui *terminalChat) renderAssistant(content string, streaming bool) []string {
	label := "fastloadai › "
	labelStyles := []string{tui.Bold, tui.Green}
	labelWidth := tui.VisibleWidth(label)
	contentWidth := ui.width - labelWidth
	indent := strings.Repeat(" ", labelWidth)
	if contentWidth < 10 {
		label = "ai › "
		labelStyles = []string{tui.Bold, tui.Green}
		labelWidth = 5
		indent = "     "
		contentWidth = ui.width - 5
	}
	if streaming && content == "" {
		content = spinnerFrames[ui.spinner%len(spinnerFrames)] + " thinking…"
	}
	raw := wrapPlain(content, max(4, contentWidth))
	inCode := false
	out := make([]string, 0, len(raw))
	for index, line := range raw {
		style := []string(nil)
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inCode = !inCode
			if inCode {
				line = "── code ──"
			} else {
				line = "──────────"
			}
			style = []string{tui.Grey}
		} else if inCode {
			style = []string{tui.Grey}
		} else if strings.HasPrefix(trimmed, "#") {
			line = strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			style = []string{tui.Bold, tui.Cyan}
		} else if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			line = "• " + strings.TrimSpace(trimmed[2:])
		}
		prefix := indent
		prefixStyles := []string(nil)
		if index == 0 {
			prefix = label
			prefixStyles = labelStyles
		}
		segments := []segment{{prefix, prefixStyles}, {line, style}}
		if streaming && index == len(raw)-1 {
			segments = append(segments, segment{"▌", []string{tui.Green}})
		}
		out = append(out, ui.renderSegments(segments, ui.width))
	}
	return out
}

func (ui *terminalChat) renderInfo(content, style string) []string {
	if style == "" {
		style = tui.Dim
	}
	raw := wrapPlain(content, max(4, ui.width-2))
	out := make([]string, 0, len(raw))
	for _, line := range raw {
		out = append(out, ui.renderSegments([]segment{{"  ", nil}, {line, []string{style}}}, ui.width))
	}
	return out
}

func (ui *terminalChat) renderSegments(segments []segment, width int) string {
	var builder strings.Builder
	used := 0
	for _, item := range segments {
		if used >= width {
			break
		}
		text := clipPlain(sanitizeText(item.text), width-used)
		if text == "" {
			continue
		}
		builder.WriteString(ui.theme.Paint(text, item.styles...))
		used += displayWidth(text)
	}
	if used < width {
		builder.WriteString(strings.Repeat(" ", width-used))
	}
	return builder.String()
}

func wrapPlain(text string, width int) []string {
	if width < 1 {
		width = 1
	}
	var out []string
	for _, paragraph := range strings.Split(text, "\n") {
		if paragraph == "" {
			out = append(out, "")
			continue
		}
		line := []rune{}
		lineWidth := 0
		for _, r := range paragraph {
			runes := runeWidth(r)
			if lineWidth+runes > width && len(line) > 0 {
				cut := len(line)
				for index := len(line) - 1; index >= 0; index-- {
					if line[index] == ' ' || line[index] == '\t' {
						cut = index
						break
					}
				}
				out = append(out, string(line[:cut]))
				if cut < len(line) {
					line = append([]rune(nil), line[cut+1:]...)
					lineWidth = displayWidth(string(line))
				} else {
					line = nil
					lineWidth = 0
				}
			}
			line = append(line, r)
			lineWidth += runes
		}
		out = append(out, string(line))
	}
	return out
}

func runeWidth(r rune) int {
	switch {
	case r == 0:
		return 0
	case unicode.Is(unicode.Mn, r) || unicode.Is(unicode.Me, r) || r == 0x200b:
		return 0
	case r >= 0x1100 && (r <= 0x115f ||
		r == 0x2329 || r == 0x232a ||
		(r >= 0x2e80 && r <= 0xa4cf && r != 0x303f) ||
		(r >= 0xac00 && r <= 0xd7a3) ||
		(r >= 0xf900 && r <= 0xfaff) ||
		(r >= 0xfe10 && r <= 0xfe19) ||
		(r >= 0xfe30 && r <= 0xfe6f) ||
		(r >= 0xff00 && r <= 0xff60) ||
		(r >= 0xffe0 && r <= 0xffe6) ||
		(r >= 0x1f300 && r <= 0x1faff) ||
		(r >= 0x20000 && r <= 0x3fffd)):
		return 2
	}
	return 1
}

func displayWidth(text string) int {
	total := 0
	for _, r := range text {
		total += runeWidth(r)
	}
	return total
}

func sanitizeText(text string) string {
	var builder strings.Builder
	for _, r := range text {
		switch {
		case r == '\r':
			continue
		case r == '\t':
			builder.WriteString("    ")
		case r < 32 || r == 0x7f:
			continue
		default:
			builder.WriteRune(r)
		}
	}
	return builder.String()
}

func clipPlain(text string, width int) string {
	if width <= 0 {
		return ""
	}
	var builder strings.Builder
	used := 0
	for _, r := range text {
		runes := runeWidth(r)
		if used+runes > width {
			break
		}
		builder.WriteRune(r)
		used += runes
	}
	return builder.String()
}

func padBetween(left, right string, width int) string {
	leftWidth := tui.VisibleWidth(left)
	rightWidth := tui.VisibleWidth(right)
	if leftWidth+rightWidth >= width {
		if rightWidth >= width {
			return clipPlain(right, width)
		}
		left = clipPlain(left, width-rightWidth-1)
		leftWidth = tui.VisibleWidth(left)
	}
	return left + strings.Repeat(" ", max(1, width-leftWidth-rightWidth)) + right
}
