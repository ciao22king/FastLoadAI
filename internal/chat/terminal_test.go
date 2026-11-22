package chat

import (
	"strings"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

func TestWrapInputTracksCursor(t *testing.T) {
	cases := []struct {
		name   string
		input  string
		cursor int
		width  int
		row    int
		column int
	}{
		{"empty", "", 0, 10, 0, 0},
		{"end", "hello", 5, 10, 0, 5},
		{"wrapped", "hello world", 6, 5, 1, 1},
		{"newline", "ab\ncd", 3, 10, 1, 0},
		{"on newline", "ab\ncd", 2, 10, 0, 2},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			_, row, column := wrapInput([]rune(item.input), item.cursor, item.width)
			if row != item.row || column != item.column {
				t.Fatalf("cursor = %d:%d, want %d:%d", row, column, item.row, item.column)
			}
		})
	}
}

func TestWrapPlain(t *testing.T) {
	lines := wrapPlain("hello world\nsecond line", 8)
	if len(lines) < 3 {
		t.Fatalf("got %d lines: %#v", len(lines), lines)
	}
	if lines[0] != "hello" || lines[1] != "world" {
		t.Fatalf("unexpected wrap: %#v", lines)
	}
	for _, line := range lines {
		if displayWidth(line) > 8 {
			t.Fatalf("line too wide: %q", line)
		}
	}
}

func TestRenderAssistantStreamingCursor(t *testing.T) {
	ui := &terminalChat{width: 60, theme: tui.Theme{}}
	lines := ui.renderAssistant("Hello", true)
	if len(lines) == 0 {
		t.Fatal("no lines")
	}
	if !strings.Contains(lines[len(lines)-1], "▌") {
		t.Fatalf("missing streaming cursor: %q", lines[len(lines)-1])
	}
	if !strings.Contains(lines[0], "fastloadai") {
		t.Fatalf("missing assistant label: %q", lines[0])
	}
}

func TestParseEscape(t *testing.T) {
	cases := map[string]keyKind{
		"A": keyUp, "B": keyDown, "C": keyRight, "D": keyLeft,
		"H": keyHome, "F": keyEnd, "3~": keyDelete, "5~": keyPageUp, "6~": keyPageDown,
	}
	for sequence, want := range cases {
		if got := parseEscape(sequence); got.kind != want {
			t.Fatalf("parseEscape(%q) = %v, want %v", sequence, got.kind, want)
		}
	}
}

func TestRenderSegmentsClipsAndPads(t *testing.T) {
	ui := &terminalChat{theme: tui.Theme{}}
	line := ui.renderSegments([]segment{{"hello", nil}, {" world", nil}}, 8)
	if line != "hello wo" {
		t.Fatalf("line = %q", line)
	}
}

func TestHistoryUpReturnsLatestPrompt(t *testing.T) {
	ui := &terminalChat{}
	ui.input = []rune("/help")
	ui.submit()
	ui.input = []rune("/stats")
	ui.submit()
	ui.historyUp()
	if got := string(ui.input); got != "/stats" {
		t.Fatalf("history = %q, want /stats", got)
	}
}

func TestSanitizeText(t *testing.T) {
	if got := sanitizeText("a\tb\r\x00c"); got != "a    bc" {
		t.Fatalf("sanitizeText = %q", got)
	}
}

func TestMaxInputLinesFitsSmallTerminal(t *testing.T) {
	ui := &terminalChat{height: 10}
	if got := ui.maxInputLines(); got != 2 {
		t.Fatalf("maxInputLines = %d, want 2", got)
	}
}

func TestConversationCacheRebuildsAfterClear(t *testing.T) {
	ui := &terminalChat{width: 80, theme: tui.Theme{}}
	ui.appendTranscript(chatItem{role: "user", content: "hello"})
	first := ui.conversationLines()
	if len(first) == 0 {
		t.Fatal("no conversation lines")
	}
	ui.transcript = nil
	ui.transcriptVersion++
	ui.appendTranscript(chatItem{role: "info", content: "cleared", style: tui.Dim})
	second := ui.conversationLines()
	if len(second) == 0 || second[0] == first[0] {
		t.Fatalf("cache was not rebuilt: %#v", second)
	}
}

func TestWelcomeLinesFitWidth(t *testing.T) {
	ui := &terminalChat{width: 80, theme: tui.Theme{}, config: model.Config{ModelPath: "qwen.gguf", Quantization: "BF16", ContextSize: 4096}}
	lines := ui.welcomeLines()
	if len(lines) < 5 {
		t.Fatalf("welcome has %d lines", len(lines))
	}
	for _, line := range lines {
		if tui.VisibleWidth(line) > 80 {
			t.Fatalf("welcome line too wide: %q", line)
		}
	}
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "FastLoadAI is ready") || !strings.Contains(joined, "qwen.gguf") {
		t.Fatalf("welcome content: %q", joined)
	}
}

func TestTranscriptMarkdown(t *testing.T) {
	ui := &terminalChat{
		config: model.Config{ModelPath: "qwen.gguf", Quantization: "BF16", ContextSize: 4096},
		transcript: []chatItem{
			{role: "user", content: "hello"},
			{role: "assistant", content: "hi"},
		},
	}
	text := ui.transcriptMarkdown()
	for _, want := range []string{"# FastLoadAI chat", "## You", "hello", "## FastLoadAI", "hi", "qwen.gguf"} {
		if !strings.Contains(text, want) {
			t.Fatalf("markdown missing %q: %s", want, text)
		}
	}
}

func TestLastAssistant(t *testing.T) {
	ui := &terminalChat{transcript: []chatItem{{role: "user", content: "q"}, {role: "assistant", content: "a"}}}
	if got := ui.lastAssistant(); got != "a" {
		t.Fatalf("lastAssistant = %q", got)
	}
}

func TestRenderAssistantHeading(t *testing.T) {
	ui := &terminalChat{width: 60, theme: tui.Theme{}}
	lines := ui.renderAssistant("# Title\n- item", false)
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "Title") || !strings.Contains(joined, "• item") {
		t.Fatalf("formatted output: %q", joined)
	}
}
