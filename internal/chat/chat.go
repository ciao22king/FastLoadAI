// Package chat runs the FastLoadAI streaming chat over llama-server.
package chat

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/llama"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/tui"
)

const (
	// MaxHistoryMessages bounds the conversation when no context is known.
	MaxHistoryMessages = 21
	// SystemPrompt is the default chat instruction.
	SystemPrompt = "You are a helpful assistant. Answer in English unless the user asks for another language. You can only see this conversation: you have no access to files or a shell."
)

// Options configures one chat session.
type Options struct {
	InitialPrompt string
	Temperature   float64
	MaxTokens     int
	SystemPrompt  string
	Once          bool
	NoTUI         bool
	Color         bool
	ColorSet      bool
	Tools         *Toolbox
	MaxToolRounds int
}

type sessionSettings struct {
	systemPrompt  string
	temperature   float64
	tokenLimit    int
	historyBudget int
	tools         *Toolbox
	maxToolRounds int
}

func resolveSettings(config model.Config, options Options) sessionSettings {
	systemPrompt := options.SystemPrompt
	if systemPrompt == "" {
		systemPrompt = SystemPrompt
		if options.Tools != nil {
			systemPrompt = "You are FastLoadAI, a concise local assistant running on the user's computer."
		}
	}
	temperature := options.Temperature
	if temperature <= 0 {
		temperature = 0.7
	}
	tokenLimit := options.MaxTokens
	if tokenLimit <= 0 {
		tokenLimit = 1024
		if config.ContextSize > 0 {
			tokenLimit = min(1024, max(128, config.ContextSize/4))
		}
	}
	maxToolRounds := options.MaxToolRounds
	if maxToolRounds <= 0 {
		maxToolRounds = 4
	}
	if options.Tools != nil {
		systemPrompt += "\n\n" + options.Tools.NativeInstructions()
	}
	return sessionSettings{
		systemPrompt:  systemPrompt,
		temperature:   temperature,
		tokenLimit:    tokenLimit,
		historyBudget: int(float64(config.ContextSize) * 0.6),
		tools:         options.Tools,
		maxToolRounds: maxToolRounds,
	}
}

// StreamChat sends messages and yields content chunks as they arrive.
func StreamChat(ctx context.Context, baseURL string, messages []model.Message, temperature float64, maxTokens int) (<-chan string, <-chan error) {
	chunks := make(chan string)
	errs := make(chan error, 1)
	go func() {
		defer close(chunks)
		defer close(errs)
		payload, _ := json.Marshal(map[string]any{
			"model": "local-model", "messages": messages,
			"temperature": temperature, "max_tokens": maxTokens, "stream": true,
		})

		request, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", strings.NewReader(string(payload)))
		if err != nil {
			errs <- err
			return
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := (&http.Client{Timeout: 0}).Do(request)
		if err != nil {
			errs <- fmt.Errorf("lost connection to the model: %w", err)
			return
		}
		defer response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
			errs <- fmt.Errorf("llama-server rejected the request: %s", strings.TrimSpace(string(body)))
			return
		}
		scanner := bufio.NewScanner(response.Body)
		scanner.Buffer(make([]byte, 4096), 1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if !strings.HasPrefix(line, "data:") {
				continue
			}

			data := strings.TrimSpace(line[5:])

			if data == "[DONE]" {
				break
			}
			var event struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if json.Unmarshal([]byte(data), &event) != nil {
				continue
			}
			if len(event.Choices) > 0 && event.Choices[0].Delta.Content != "" {
				chunks <- event.Choices[0].Delta.Content
			}
		}
		if err := scanner.Err(); err != nil {
			errs <- fmt.Errorf("lost connection to the model: %w", err)
		}
	}()
	return chunks, errs
}

func streamAnswer(server *llama.Server, messages []model.Message, temperature float64, tokenLimit int, theme tui.Theme, interactive bool, assistantLabel string, tools *Toolbox) (string, []ToolCall, time.Duration, error) {
	fmt.Printf("\n%s ", assistantLabel)
	spinner := tui.StartSpinner("waiting for the first token…", theme)
	started := time.Now()
	var parts []string
	firstChunk := true
	streamCtx, streamCancel := context.WithCancel(context.Background())
	defer streamCancel()
	var chunks <-chan string
	var callCh <-chan []ToolCall
	var errs <-chan error
	if tools != nil {
		chunks, callCh, errs = StreamChatWithTools(streamCtx, server.BaseURL, messages, temperature, tokenLimit, tools.Definitions())
	} else {
		chunks, errs = StreamChat(streamCtx, server.BaseURL, messages, temperature, tokenLimit)
	}
	for chunk := range chunks {
		if firstChunk {
			firstChunk = false
			spinner.Stop()
			if interactive {
				fmt.Print("\r\033[K" + assistantLabel + " ")
			}
		}
		fmt.Print(chunk)
		parts = append(parts, chunk)
	}
	spinner.Stop()
	var streamErr error
	select {
	case streamErr = <-errs:
	default:
	}
	var calls []ToolCall
	if callCh != nil {
		if native, ok := <-callCh; ok {
			calls = native
		}
	}
	return strings.TrimSpace(strings.Join(parts, "")), calls, time.Since(started), streamErr
}

// TrimHistory drops the oldest user/assistant pairs until the estimate fits.
func TrimHistory(messages []model.Message, budget int) []model.Message {
	for len(messages) > 3 && tui.ConversationTokens(messages) > budget {
		messages = append(messages[:1], messages[3:]...)
	}
	return messages
}

func printChatSummary(stats tui.ChatStats, theme tui.Theme) {
	fmt.Println("\n" + theme.Paint("Chat ended.", tui.Dim))
	if summary := stats.RenderSummary(theme); summary != "" {
		fmt.Println()
		fmt.Println(summary)
	}
}

// RunChat starts llama-server and opens the streaming chat.
func RunChat(modelPath string, config model.Config, options Options) int {
	theme := tui.Theme{Enabled: tui.ColorEnabled()}
	if options.ColorSet {
		theme.Enabled = options.Color
	}
	executable, err := exec.LookPath("llama-server")
	if err != nil {
		fmt.Fprintln(os.Stderr, theme.Paint("llama-server not found. Install a llama.cpp build that includes the server and put it on PATH.", tui.Red))
		return 2
	}
	if options.Tools != nil {
		config.Jinja = true
		// Quantized KV caches measurably degrade tool selection on small models.
		config.KVCacheTypeK = "f16"
		config.KVCacheTypeV = "f16"
	}
	fmt.Println(tui.RenderBanner(filepath.Base(modelPath), config, theme, 0))
	loading := tui.StartSpinner("loading model, this can take a while…", theme)
	server, err := llama.Start(executable, modelPath, config)
	if err != nil {
		loading.Stop()
		fmt.Fprintf(os.Stderr, "Could not start llama-server: %v\n", err)
		return 1
	}
	defer server.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	waitErr := server.WaitReady(ctx, 300*time.Second)
	loading.Stop()
	if waitErr != nil {
		fmt.Fprintf(os.Stderr, "Error while starting the chat: %v\n", waitErr)
		if details := server.LogTail(); details != "" {
			fmt.Fprintln(os.Stderr, details)
			if llama.IsMemoryError(details) {
				fmt.Fprintln(os.Stderr, theme.Paint("llama-server ran out of memory. Try --ctx-size 1024, --prefer speed, or a smaller model (fastloadai suggest).", tui.Yellow))
			}
		}
		return 1
	}
	interactive := tui.StdinIsTerminal() && tui.StdoutIsTerminal()
	oneShot := options.Once || (options.InitialPrompt != "" && !tui.StdinIsTerminal())
	settings := resolveSettings(config, options)
	if !oneShot && interactive && !options.NoTUI {
		if code, ok := runTerminalChat(server, config, options, theme, settings); ok {
			return code
		}
	}
	fmt.Println(theme.Paint("Ready.", tui.Green) + " " + theme.Paint("Type /help for commands, /exit to quit.", tui.Dim))
	systemPrompt := settings.systemPrompt
	temperature := settings.temperature
	tokenLimit := settings.tokenLimit
	historyBudget := settings.historyBudget
	stats := tui.ChatStats{}
	messages := []model.Message{{Role: "system", Content: systemPrompt}}
	nextPrompt := options.InitialPrompt
	hasNext := nextPrompt != ""
	stdin := bufio.NewReader(os.Stdin)
	userLabel := theme.Paint("you ›", tui.Bold, tui.Cyan)
	assistantLabel := theme.Paint("fastloadai ›", tui.Bold, tui.Green)
	for {
		var prompt string
		if hasNext {
			prompt = nextPrompt
			hasNext = false
			fmt.Printf("\n%s %s\n", userLabel, prompt)
		} else {
			fmt.Printf("\n%s ", userLabel)
			line, err := stdin.ReadString(0x0a)
			if err != nil && strings.TrimSpace(line) == "" {
				printChatSummary(stats, theme)
				return 0
			}
			prompt = strings.TrimSpace(line)
		}
		if prompt == "" {
			continue
		}
		command := strings.ToLower(prompt)
		switch command {
		case "/exit", "/quit":
			printChatSummary(stats, theme)
			return 0
		case "/help":
			fmt.Println()
			fmt.Println(tui.RenderHelp(theme))
			continue
		case "/clear":
			messages = messages[:1]
			stats = tui.ChatStats{}
			fmt.Println(theme.Paint("Conversation cleared.", tui.Dim))
			continue
		case "/stats":
			fmt.Println()
			fmt.Println(stats.RenderStats(theme))
			continue
		case "/context":
			fmt.Println()
			fmt.Println(tui.RenderContext(tui.ConversationTokens(messages), config.ContextSize, theme))
			continue
		case "/tools":
			fmt.Println()
			if settings.tools == nil {
				fmt.Println(theme.Paint("Tools are off. Start FastLoadAI with --tools to enable file and web tools.", tui.Dim))
			} else {
				fmt.Println(theme.Paint(settings.tools.Status(), tui.Dim))
			}
			continue
		}
		if command == "/retry" {
			if len(messages) < 2 || messages[len(messages)-1].Role != "assistant" {
				fmt.Println(theme.Paint("Nothing to retry yet.", tui.Dim))
				continue
			}
			lastUser := messages[len(messages)-2].Content
			messages = messages[:len(messages)-2]
			nextPrompt = lastUser
			hasNext = true
			continue
		}
		if strings.HasPrefix(prompt, "/") {
			fmt.Println(theme.Paint("Unknown command "+prompt+". Type /help for the list.", tui.Yellow))
			continue
		}
		if historyBudget > 0 {
			messages = TrimHistory(messages, max(256, historyBudget-tui.EstimateTokens(prompt)))
		} else {
			for len(messages) >= MaxHistoryMessages {
				messages = append(messages[:1], messages[3:]...)
			}
		}
		messages = append(messages, model.Message{Role: "user", Content: prompt})
		toolRounds := 0
		var answer string
		var nativeCalls []ToolCall
		var elapsed time.Duration
		var streamErr error
		for {
			answer, nativeCalls, elapsed, streamErr = streamAnswer(server, messages, temperature, tokenLimit, theme, interactive, assistantLabel, settings.tools)
			if streamErr != nil {
				messages = messages[:len(messages)-1]
				fmt.Println()
				fmt.Println(theme.Paint("Error: "+streamErr.Error(), tui.Red))
				if oneShot {
					return 1
				}
				break
			}
			if answer == "" && len(nativeCalls) == 0 {
				messages = messages[:len(messages)-1]
				fmt.Println()
				fmt.Println(theme.Paint("The model returned an empty answer. Try again.", tui.Yellow))
				if oneShot {
					return 1
				}
				break
			}
			calls := nativeCalls
			visible := answer
			if len(calls) == 0 && settings.tools != nil {
				calls, visible = ParseToolCalls(answer, settings.tools.Known())
			}
			if len(calls) > 0 && toolRounds < settings.maxToolRounds {
				toolRounds++
				if visible != "" {
					fmt.Println()
					fmt.Println(visible)
				}
				native := len(nativeCalls) > 0
				if native {
					assistant := model.Message{Role: "assistant", Content: visible}
					for index, call := range calls {
						id := call.NativeID
						if id == "" {
							id = fmt.Sprintf("call-%d", index+1)
						}
						arguments, _ := json.Marshal(call.Arguments)
						assistant.ToolCalls = append(assistant.ToolCalls, model.ToolCall{
							ID: id, Type: "function",
							Function: model.FunctionCall{Name: call.Name, Arguments: string(arguments)},
						})
					}
					messages = append(messages, assistant)
				} else {
					messages = append(messages, model.Message{Role: "assistant", Content: answer})
				}
				for index, call := range calls {
					fmt.Println()
					fmt.Println(theme.Paint("tool ›", tui.Bold, tui.Cyan), CallSummary(call))
					result := settings.tools.Execute(call)
					if result.OK {
						fmt.Println(theme.Paint("  ✓ "+result.Summary, tui.Green))
					} else {
						fmt.Println(theme.Paint("  ✗ "+result.Error, tui.Red))
					}
					if native {
						id := call.NativeID
						if id == "" {
							id = fmt.Sprintf("call-%d", index+1)
						}
						messages = append(messages, model.Message{Role: "tool", ToolCallID: id, Content: FormatToolResult(call, result)})
					} else {
						messages = append(messages, model.Message{Role: "user", Content: FormatToolResult(call, result)})
					}
				}
				nativeCalls = nil
				continue
			}
			if len(calls) > 0 {
				fmt.Println(theme.Paint("Tool round limit reached; answer without more tools.", tui.Yellow))
				if visible != "" {
					answer = visible
				}
			}
			break
		}
		if answer == "" {
			continue
		}
		stats.Record(tui.EstimateTokens(prompt), tui.EstimateTokens(answer), elapsed)
		fmt.Println()
		fmt.Println(tui.RenderTiming(tui.EstimateTokens(answer), elapsed, config.ContextSize, tui.ConversationTokens(messages), theme))
		messages = append(messages, model.Message{Role: "assistant", Content: answer})
		if oneShot {
			return 0
		}
	}
}

func StreamChatWithTools(ctx context.Context, baseURL string, messages []model.Message, temperature float64, maxTokens int, tools []ToolDefinition) (<-chan string, <-chan []ToolCall, <-chan error) {
	chunks := make(chan string)
	callCh := make(chan []ToolCall, 1)
	errs := make(chan error, 1)
	go func() {
		defer close(chunks)
		defer close(callCh)
		defer close(errs)
		payload := map[string]any{
			"model": "local-model", "messages": messages,
			"temperature": temperature, "max_tokens": maxTokens, "stream": false,
		}
		if len(tools) > 0 {
			payload["tools"] = tools
			payload["tool_choice"] = "auto"
		}
		encoded, _ := json.Marshal(payload)
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", strings.NewReader(string(encoded)))

		if err != nil {
			errs <- err
			return
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := (&http.Client{Timeout: 0}).Do(request)
		if err != nil {
			errs <- fmt.Errorf("lost connection to the model: %w", err)
			return
		}
		defer response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
			errs <- fmt.Errorf("llama-server rejected the request: %s", strings.TrimSpace(string(body)))
			return
		}
		var event struct {
			Choices []struct {
				Message struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(&event); err != nil {
			errs <- fmt.Errorf("could not read the model response: %w", err)
			return
		}
		if len(event.Choices) == 0 {
			errs <- fmt.Errorf("the model returned no choices")
			return
		}
		message := event.Choices[0].Message

		if message.Content != "" {
			chunks <- message.Content
		}
		var calls []ToolCall
		for _, native := range message.ToolCalls {
			call := ToolCall{Name: native.Function.Name, Arguments: map[string]json.RawMessage{}}
			if native.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(native.Function.Arguments), &call.Arguments); err != nil {
					errs <- fmt.Errorf("invalid arguments for %s: %w", native.Function.Name, err)
					return
				}
			}
			call.NativeID = native.ID
			calls = append(calls, call)
		}
		if len(calls) > 0 {
			callCh <- calls
		}
	}()
	return chunks, callCh, errs
}
