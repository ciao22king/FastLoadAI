package chat

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"
	"unicode"

	"golang.org/x/term"

	"github.com/ciao22king/FastLoadAI/internal/llama"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

type chatItem struct {
	role    string
	content string
	style   string
}

type segment struct {
	text   string
	styles []string
}

type keyKind int

const (
	keyRune keyKind = iota
	keyEnter
	keyNewline
	keyBackspace
	keyDelete
	keyUp
	keyDown
	keyLeft
	keyRight
	keyHome
	keyEnd
	keyPageUp
	keyPageDown
	keyTab
	keyEsc
	keyCtrlC
	keyCtrlD
	keyCtrlU
	keyCtrlW
	keyCtrlK
	keyCtrlA
	keyCtrlE
	keyCtrlL
	keyEOF
	keyIgnore
)

type keyEvent struct {
	kind keyKind
	r    rune
}

type terminalChat struct {
	server   *llama.Server
	config   model.Config
	options  Options
	theme    tui.Theme
	settings sessionSettings
	state    *term.State
	left     bool

	width  int
	height int
	dirty  bool
	quit   bool

	messages          []model.Message
	transcript        []chatItem
	transcriptVersion int
	rendered          []string
	renderedVersion   int
	renderedWidth     int
	stats             tui.ChatStats

	input      []rune
	cursor     int
	history    []string
	historyPos int
	draft      []rune

	scroll int

	busy      bool
	streaming string
	streamErr error
	cancelled bool
	chunks    <-chan string
	errs      <-chan error
	cancel    context.CancelFunc
	started   time.Time
	spinner   int
	exitCode  int
}

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

var chatCommands = []string{"/help", "/clear", "/stats", "/context", "/retry", "/copy", "/save", "/exit", "/quit"}

// runTerminalChat opens a full-screen chat when both stdin and stdout are TTYs.
func runTerminalChat(server *llama.Server, config model.Config, options Options, theme tui.Theme, settings sessionSettings) (int, bool) {
	width, height, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || width < 24 || height < 10 {
		return 0, false
	}
	state, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		return 0, false
	}
	ui := &terminalChat{
		server: server, config: config, options: options, theme: theme,
		settings: settings, state: state, width: width, height: height,
		messages: []model.Message{{Role: "system", Content: settings.systemPrompt}},
	}
	ui.enter()
	defer ui.leave()

	keys := make(chan keyEvent, 64)
	go readKeyEvents(os.Stdin, keys)
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt)
	defer signal.Stop(signals)
	ticker := time.NewTicker(80 * time.Millisecond)
	defer ticker.Stop()

	ui.draw()
	if options.InitialPrompt != "" {
		ui.history = append(ui.history, options.InitialPrompt)
		ui.historyPos = len(ui.history)
		ui.sendPrompt(options.InitialPrompt)
		ui.draw()
	}
	for !ui.quit {
		select {
		case event := <-keys:
			ui.handleKey(event)
			ui.draw()
		case <-signals:
			ui.quit = true
		case chunk, ok := <-ui.chunks:
			if !ok {
				ui.chunks = nil
				if ui.errs == nil {
					ui.finishStream()
				}
				continue
			}
			ui.streaming += chunk
			ui.dirty = true
		case streamErr, ok := <-ui.errs:
			if !ok {
				ui.errs = nil
				if ui.chunks == nil {
					ui.finishStream()
				}
				continue
			}
			ui.streamErr = streamErr
		case <-ticker.C:
			ui.spinner++
			if ui.busy || ui.dirty {
				ui.draw()
			}
			if ui.resize() {
				ui.draw()
			}
		}
	}
	if ui.cancel != nil {
		ui.cancel()
	}
	ui.leave()
	if ui.stats.Turns > 0 {
		printChatSummary(ui.stats, theme)
	} else {
		fmt.Println(theme.Paint("Chat ended.", tui.Dim))
	}
	return ui.exitCode, true
}

func (ui *terminalChat) enter() {
	fmt.Print("\033[?1049h\033[?25l\033[2J\033[H")
}

func (ui *terminalChat) leave() {
	if ui.left {
		return
	}
	ui.left = true
	fmt.Print("\033[?25h\033[?1049l")
	if ui.state != nil {
		_ = term.Restore(int(os.Stdin.Fd()), ui.state)
	}
}

func (ui *terminalChat) resize() bool {
	width, height, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil || width < 24 || height < 10 {
		return false
	}
	if width == ui.width && height == ui.height {
		return false
	}
	ui.width, ui.height = width, height
	ui.dirty = true
	return true
}

func (ui *terminalChat) handleKey(event keyEvent) {
	switch event.kind {
	case keyEOF:
		ui.quit = true
	case keyCtrlC:
		if ui.busy {
			ui.cancelStream()
		} else {
			ui.quit = true
		}
	case keyCtrlD:
		if len(ui.input) == 0 {
			ui.quit = true
		} else {
			ui.deleteAtCursor()
		}
	case keyEnter:
		ui.submit()
	case keyNewline:
		ui.insertRune('\n')
	case keyBackspace:
		ui.backspace()
	case keyDelete:
		ui.deleteAtCursor()
	case keyLeft:
		if ui.cursor > 0 {
			ui.cursor--
		}
	case keyRight:
		if ui.cursor < len(ui.input) {
			ui.cursor++
		}
	case keyHome, keyCtrlA:
		ui.cursor = 0
	case keyEnd, keyCtrlE:
		ui.cursor = len(ui.input)
	case keyUp:
		ui.historyUp()
	case keyDown:
		ui.historyDown()
	case keyPageUp:
		ui.scroll += max(1, ui.conversationHeight()/2)
	case keyPageDown:
		ui.scroll = max(0, ui.scroll-max(1, ui.conversationHeight()/2))
	case keyTab:
		ui.complete()
	case keyCtrlU:
		ui.input = append([]rune(nil), ui.input[ui.cursor:]...)
		ui.cursor = 0
	case keyCtrlW:
		ui.deleteWord()
	case keyCtrlK:
		ui.input = ui.input[:ui.cursor]
	case keyCtrlL:
		ui.dirty = true
	case keyRune:
		ui.insertRune(event.r)
	}
	ui.dirty = true
}

func (ui *terminalChat) insertRune(r rune) {
	if r == 0 || r == '\r' {
		return
	}
	ui.input = append(ui.input, 0)
	copy(ui.input[ui.cursor+1:], ui.input[ui.cursor:])
	ui.input[ui.cursor] = r
	ui.cursor++
}

func (ui *terminalChat) backspace() {
	if ui.cursor == 0 {
		return
	}
	ui.input = append(ui.input[:ui.cursor-1], ui.input[ui.cursor:]...)
	ui.cursor--
}

func (ui *terminalChat) deleteAtCursor() {
	if ui.cursor >= len(ui.input) {
		return
	}
	ui.input = append(ui.input[:ui.cursor], ui.input[ui.cursor+1:]...)
}

func (ui *terminalChat) deleteWord() {
	for ui.cursor > 0 && unicode.IsSpace(ui.input[ui.cursor-1]) {
		ui.backspace()
	}
	for ui.cursor > 0 && !unicode.IsSpace(ui.input[ui.cursor-1]) {
		ui.backspace()
	}
}

func (ui *terminalChat) historyUp() {
	if len(ui.history) == 0 {
		return
	}
	if ui.historyPos == len(ui.history) {
		ui.draft = append([]rune(nil), ui.input...)
	}
	if ui.historyPos > 0 {
		ui.historyPos--
		ui.input = []rune(ui.history[ui.historyPos])
		ui.cursor = len(ui.input)
	}
}

func (ui *terminalChat) historyDown() {
	if ui.historyPos >= len(ui.history) {
		return
	}
	ui.historyPos++
	if ui.historyPos == len(ui.history) {
		ui.input = append([]rune(nil), ui.draft...)
	} else {
		ui.input = []rune(ui.history[ui.historyPos])
	}
	ui.cursor = len(ui.input)
}

func (ui *terminalChat) complete() {
	text := string(ui.input)
	if !strings.HasPrefix(text, "/") || strings.ContainsAny(text, " \n") {
		return
	}
	var matches []string
	for _, command := range chatCommands {
		if strings.HasPrefix(command, text) {
			matches = append(matches, command)
		}
	}
	switch len(matches) {
	case 1:
		ui.input = []rune(matches[0] + " ")
		ui.cursor = len(ui.input)
	case 0:
		return
	default:
		ui.addInfo("Commands: "+strings.Join(matches, "  "), tui.Dim)
	}
}

func (ui *terminalChat) submit() {
	text := strings.TrimSpace(string(ui.input))
	if text == "" {
		return
	}
	ui.input = nil
	ui.cursor = 0
	if len(ui.history) == 0 || ui.history[len(ui.history)-1] != text {
		ui.history = append(ui.history, text)
	}
	ui.historyPos = len(ui.history)
	if strings.HasPrefix(text, "/") {
		ui.command(text)
		return
	}
	ui.sendPrompt(text)
}

func (ui *terminalChat) command(text string) {
	switch strings.ToLower(text) {
	case "/exit", "/quit":
		ui.quit = true
	case "/help":
		ui.addInfo(tui.RenderHelp(tui.Theme{}), tui.Dim)
	case "/clear":
		if ui.busy {
			ui.addInfo("Wait for the current answer, or press Ctrl+C to cancel it.", tui.Yellow)
			return
		}
		ui.messages = ui.messages[:1]
		ui.transcript = nil
		ui.transcriptVersion++
		ui.stats = tui.ChatStats{}
		ui.addInfo("Conversation cleared.", tui.Dim)
	case "/stats":
		ui.addInfo(ui.stats.RenderStats(tui.Theme{}), tui.Dim)
	case "/context":
		ui.addInfo(tui.RenderContext(tui.ConversationTokens(ui.messages), ui.config.ContextSize, tui.Theme{}), tui.Dim)
	case "/copy":
		answer := ui.lastAssistant()
		if answer == "" {
			ui.addInfo("No assistant answer to copy yet.", tui.Dim)
			return
		}
		fmt.Printf("\033]52;c;%s\a", base64.StdEncoding.EncodeToString([]byte(answer)))
		ui.addInfo("Copied the last answer to the terminal clipboard.", tui.Green)
	case "/save":
		path, err := ui.saveTranscript()
		if err != nil {
			ui.addInfo("Could not save the transcript: "+err.Error(), tui.Red)
			return
		}
		ui.addInfo("Saved transcript to "+path, tui.Green)
	case "/retry":
		if ui.busy {
			ui.addInfo("Wait for the current answer, or press Ctrl+C to cancel it.", tui.Yellow)
			return
		}
		if len(ui.messages) < 2 || ui.messages[len(ui.messages)-1].Role != "assistant" {
			ui.addInfo("Nothing to retry yet.", tui.Dim)
			return
		}
		lastUser := ui.messages[len(ui.messages)-2].Content
		ui.messages = ui.messages[:len(ui.messages)-2]
		for index := len(ui.transcript) - 1; index >= 0; index-- {
			if ui.transcript[index].role == "assistant" {
				ui.transcript = append(ui.transcript[:index], ui.transcript[index+1:]...)
				break
			}
		}
		for index := len(ui.transcript) - 1; index >= 0; index-- {
			if ui.transcript[index].role == "user" {
				ui.transcript = append(ui.transcript[:index], ui.transcript[index+1:]...)
				break
			}
		}
		ui.transcriptVersion++
		ui.sendPrompt(lastUser)
	default:
		ui.addInfo("Unknown command "+text+". Type /help for the list.", tui.Yellow)
	}
}

func (ui *terminalChat) lastAssistant() string {
	for index := len(ui.transcript) - 1; index >= 0; index-- {
		if ui.transcript[index].role == "assistant" {
			return ui.transcript[index].content
		}
	}
	return ""
}

func (ui *terminalChat) saveTranscript() (string, error) {
	path := "fastloadai-chat-" + time.Now().Format("20060102-150405") + ".md"
	return path, os.WriteFile(path, []byte(ui.transcriptMarkdown()), 0o644)
}

func (ui *terminalChat) transcriptMarkdown() string {
	var builder strings.Builder
	builder.WriteString("# FastLoadAI chat\n\n")
	builder.WriteString("- model: `" + filepath.Base(ui.config.ModelPath) + "`\n")
	if ui.config.Quantization != "" {
		builder.WriteString("- quantization: `" + ui.config.Quantization + "`\n")
	}
	builder.WriteString("- context: " + fmt.Sprintf("%d tokens", ui.config.ContextSize) + "\n\n")
	for _, item := range ui.transcript {
		switch item.role {
		case "user":
			builder.WriteString("## You\n\n" + item.content + "\n\n")
		case "assistant":
			builder.WriteString("## FastLoadAI\n\n" + item.content + "\n\n")
		}
	}
	return builder.String()
}

func (ui *terminalChat) appendTranscript(item chatItem) {
	ui.transcript = append(ui.transcript, item)
	ui.transcriptVersion++
}

func (ui *terminalChat) addInfo(text, style string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}
	if style == "" {
		style = tui.Dim
	}
	ui.appendTranscript(chatItem{role: "info", content: text, style: style})
	ui.scroll = 0
	ui.dirty = true
}

func (ui *terminalChat) sendPrompt(prompt string) {
	if ui.busy {
		ui.addInfo("Wait for the current answer, or press Esc to cancel it.", tui.Yellow)
		return
	}
	if ui.settings.historyBudget > 0 {
		ui.messages = TrimHistory(ui.messages, max(256, ui.settings.historyBudget-tui.EstimateTokens(prompt)))
	} else {
		for len(ui.messages) >= MaxHistoryMessages {
			ui.messages = append(ui.messages[:1], ui.messages[3:]...)
		}
	}
	ui.messages = append(ui.messages, model.Message{Role: "user", Content: prompt})
	ui.appendTranscript(chatItem{role: "user", content: prompt})
	ui.streaming = ""
	ui.streamErr = nil
	ui.cancelled = false
	ui.busy = true
	ui.started = time.Now()
	ui.scroll = 0
	ctx, cancel := context.WithCancel(context.Background())
	ui.cancel = cancel
	ui.chunks, ui.errs = StreamChat(ctx, ui.server.BaseURL, ui.messages, ui.settings.temperature, ui.settings.tokenLimit)
	ui.dirty = true
}

func (ui *terminalChat) cancelStream() {
	if !ui.busy {
		return
	}
	ui.cancelled = true
	if ui.cancel != nil {
		ui.cancel()
	}
	ui.dirty = true
}

func (ui *terminalChat) finishStream() {
	if !ui.busy {
		return
	}
	answer := strings.TrimSpace(ui.streaming)
	switch {
	case ui.cancelled:
		if answer != "" {
			ui.appendTranscript(chatItem{role: "assistant", content: answer})
			ui.messages = append(ui.messages, model.Message{Role: "assistant", Content: answer})
			ui.stats.Record(tui.EstimateTokens(lastUserPrompt(ui.messages)), tui.EstimateTokens(answer), time.Since(ui.started))
		} else {
			ui.dropLastUser()
		}
		ui.addInfo("Generation cancelled.", tui.Yellow)
	case ui.streamErr != nil:
		ui.dropLastUser()
		ui.addInfo("Error: "+ui.streamErr.Error(), tui.Red)
	case answer == "":
		ui.dropLastUser()
		ui.addInfo("The model returned an empty answer. Try again.", tui.Yellow)
	default:
		ui.appendTranscript(chatItem{role: "assistant", content: answer})
		ui.messages = append(ui.messages, model.Message{Role: "assistant", Content: answer})
		ui.stats.Record(tui.EstimateTokens(lastUserPrompt(ui.messages)), tui.EstimateTokens(answer), time.Since(ui.started))
	}
	ui.streaming = ""
	ui.streamErr = nil
	ui.cancelled = false
	ui.busy = false
	ui.chunks = nil
	ui.errs = nil
	if ui.cancel != nil {
		ui.cancel()
		ui.cancel = nil
	}
	ui.scroll = 0
	ui.dirty = true
}

func (ui *terminalChat) dropLastUser() {
	if len(ui.messages) > 1 && ui.messages[len(ui.messages)-1].Role == "user" {
		ui.messages = ui.messages[:len(ui.messages)-1]
	}
}

func lastUserPrompt(messages []model.Message) string {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == "user" {
			return messages[index].Content
		}
	}
	return ""
}

func (ui *terminalChat) conversationHeight() int {
	header := 2
	input := ui.inputHeight()
	height := ui.height - header - 3 - input
	return max(1, height)
}

func (ui *terminalChat) maxInputLines() int {
	return min(6, max(1, ui.height-8))
}

func (ui *terminalChat) inputHeight() int {
	lines, _, _ := wrapInput(ui.input, ui.cursor, max(1, ui.width-4))
	return min(max(1, len(lines)), ui.maxInputLines()) + 2
}

func (ui *terminalChat) draw() {
	if ui.width < 24 || ui.height < 10 {
		return
	}
	lines := ui.screen()
	var builder strings.Builder
	builder.WriteString("\033[H")
	for index, line := range lines {
		if index >= ui.height {
			break
		}
		builder.WriteString(line)
		if index < ui.height-1 {
			builder.WriteString("\r\n")
		}
	}
	builder.WriteString("\033[J")
	fmt.Print(builder.String())
	ui.dirty = false
}
