package chat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestParseToolCallsFenced(t *testing.T) {
	text := "I will write it.\n```tool\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.txt\",\"content\":\"hi\"}}\n```\nDone."
	calls, visible := ParseToolCalls(text, map[string]bool{"write_file": true})
	if len(calls) != 1 || calls[0].Name != "write_file" {
		t.Fatalf("calls = %#v", calls)
	}
	if strings.Contains(visible, "write_file") || !strings.Contains(visible, "I will write it.") {
		t.Fatalf("visible = %q", visible)
	}
	path, err := stringArgument(calls[0], "path")
	if err != nil || path != "a.txt" {
		t.Fatalf("path = %q err = %v", path, err)
	}
}

func TestParseToolCallsTopLevel(t *testing.T) {
	calls, visible := ParseToolCalls(`{"tool":"read_file","path":"a.txt"}`, map[string]bool{"read_file": true})
	if len(calls) != 1 || calls[0].Name != "read_file" || visible != "" {
		t.Fatalf("calls = %#v visible = %q", calls, visible)
	}
}

func TestParseToolCallsIgnoresUnknown(t *testing.T) {
	calls, visible := ParseToolCalls("```json\n{\"name\":\"drop_database\"}\n```", map[string]bool{"write_file": true})
	if len(calls) != 0 || !strings.Contains(visible, "drop_database") {
		t.Fatalf("calls = %#v visible = %q", calls, visible)
	}
}

func TestToolboxWriteReadList(t *testing.T) {
	root := t.TempDir()
	toolbox, err := NewToolbox(root)
	if err != nil {
		t.Fatal(err)
	}
	write := toolbox.Execute(toolCall("write_file", map[string]any{"path": "src/main.go", "content": "package main\n"}))
	if !write.OK {
		t.Fatalf("write: %s", write.Error)
	}
	data, err := os.ReadFile(filepath.Join(root, "src", "main.go"))
	if err != nil || string(data) != "package main\n" {
		t.Fatalf("file = %q err = %v", data, err)
	}
	read := toolbox.Execute(toolCall("read_file", map[string]any{"path": "src/main.go"}))
	if !read.OK || read.Content != "package main\n" {
		t.Fatalf("read: %#v", read)
	}
	list := toolbox.Execute(toolCall("list_files", map[string]any{"path": "."}))
	if !list.OK || !strings.Contains(list.Content, "src/main.go") {
		t.Fatalf("list: %#v", list)
	}
}

func TestToolboxRejectsEscapes(t *testing.T) {
	root := t.TempDir()
	toolbox, err := NewToolbox(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"../outside.txt", "/etc/passwd", ".git/config", ".ssh/id_rsa"} {
		result := toolbox.Execute(toolCall("write_file", map[string]any{"path": path, "content": "x"}))
		if result.OK {
			t.Fatalf("path %q was accepted", path)
		}
	}
}

func TestToolboxRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	toolbox, err := NewToolbox(root)
	if err != nil {
		t.Fatal(err)
	}
	result := toolbox.Execute(toolCall("write_file", map[string]any{"path": "link/escape.txt", "content": "x"}))
	if result.OK {
		t.Fatal("symlink escape was accepted")
	}
}

func TestSearchWebWithFakeEndpoints(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
			json.NewEncoder(w).Encode(map[string]any{
				"Heading": "Go", "AbstractText": "Go is a language.", "AbstractURL": "https://go.dev/",
			})
		case "/wiki":
			json.NewEncoder(w).Encode(map[string]any{
				"query": map[string]any{"search": []map[string]any{{"title": "Go (programming language)", "snippet": "<b>Go</b> is compiled."}}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	toolbox, err := NewToolbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	toolbox.SearchURL = server.URL + "/"
	toolbox.WikiURL = server.URL + "/wiki"
	result := toolbox.Execute(toolCall("search_web", map[string]any{"query": "golang"}))
	if !result.OK || !strings.Contains(result.Content, "https://go.dev/") || !strings.Contains(result.Content, "Go is a language") {
		t.Fatalf("search: %#v", result)
	}
}

func TestFetchURLWithFakeServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("<html><body><h1>Hello</h1><script>bad()</script><p>World</p></body></html>"))
	}))
	defer server.Close()
	toolbox, err := NewToolbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	toolbox.HTTPClient = server.Client()
	toolbox.AllowPrivateURLs = true
	result := toolbox.Execute(toolCall("fetch_url", map[string]any{"url": server.URL}))
	if !result.OK || !strings.Contains(result.Content, "Hello") || !strings.Contains(result.Content, "World") || strings.Contains(result.Content, "bad()") {
		t.Fatalf("fetch: %#v", result)
	}
}

func TestFormatToolResult(t *testing.T) {
	message := FormatToolResult(ToolCall{Name: "read_file"}, ToolResult{OK: true, Content: "hello"})
	if !strings.Contains(message, `"ok":true`) || !strings.Contains(message, "untrusted") {
		t.Fatalf("message = %q", message)
	}
}

func TestResolveSettingsAddsToolInstructions(t *testing.T) {
	toolbox, err := NewToolbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	settings := resolveSettings(model.Config{}, Options{Tools: toolbox})
	if !strings.Contains(settings.systemPrompt, "write_file") || !strings.Contains(settings.systemPrompt, "search_web") {
		t.Fatalf("system prompt = %q", settings.systemPrompt)
	}
	if settings.maxToolRounds != 4 {
		t.Fatalf("maxToolRounds = %d", settings.maxToolRounds)
	}
}

func toolCall(name string, args map[string]any) ToolCall {
	call := ToolCall{Name: name, Arguments: map[string]json.RawMessage{}}
	for key, value := range args {
		data, _ := json.Marshal(value)
		call.Arguments[key] = data
	}
	return call
}

func TestParseToolCallsLooseFormat(t *testing.T) {
	text := "```tool\nwrite_file\narguments:\n{\"path\":\"hello.py\",\"content\":\"print('hello')\"}\n"
	calls, visible := ParseToolCalls(text, map[string]bool{"write_file": true})
	if len(calls) != 1 || calls[0].Name != "write_file" || visible != "" {
		t.Fatalf("calls = %#v visible = %q", calls, visible)
	}
	path, _ := stringArgument(calls[0], "path")
	if path != "hello.py" {
		t.Fatalf("path = %q", path)
	}
}

func TestNormalizeToolPath(t *testing.T) {
	root := t.TempDir()
	cases := map[string]string{
		"/path/to/your/hello.py":    "hello.py",
		"/workspace/hello.py":       "hello.py",
		"./hello.py":                "hello.py",
		".\\nhello.py":              "hello.py",
		"src/./main.go":             "src/./main.go",
		filepath.Join(root, "a.py"): "a.py",
	}
	for input, want := range cases {
		if got := normalizeToolPath(root, input); got != want {
			t.Errorf("normalizeToolPath(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestResolveSettingsReplacesOfflinePromptWhenToolsOn(t *testing.T) {
	toolbox, err := NewToolbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	settings := resolveSettings(model.Config{}, Options{Tools: toolbox})
	if strings.Contains(settings.systemPrompt, "no access to files") {
		t.Fatalf("system prompt still claims no file access: %q", settings.systemPrompt)
	}
}
