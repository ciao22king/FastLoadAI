package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestTrimHistoryKeepsSystemAndLastPair(t *testing.T) {
	messages := []model.Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: strings.Repeat("a", 40)},
		{Role: "assistant", Content: strings.Repeat("b", 40)},
		{Role: "user", Content: strings.Repeat("c", 40)},
		{Role: "assistant", Content: strings.Repeat("d", 40)},
		{Role: "user", Content: "latest"},
		{Role: "assistant", Content: "answer"},
	}
	trimmed := TrimHistory(messages, 10)
	if len(trimmed) != 3 {
		t.Fatalf("len = %d, want 3: %+v", len(trimmed), trimmed)
	}
	if trimmed[0].Role != "system" || trimmed[1].Content != "latest" || trimmed[2].Content != "answer" {
		t.Fatalf("unexpected history: %+v", trimmed)
	}
}

func TestStreamChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":" world"}}]}`+"\n\n")
	}))
	defer server.Close()
	chunks, errs := StreamChat(context.Background(), server.URL, []model.Message{{Role: "user", Content: "hi"}}, 0.1, 8)
	var got strings.Builder
	for chunk := range chunks {
		got.WriteString(chunk)
	}
	if got.String() != "Hello world" {
		t.Fatalf("stream = %q", got.String())
	}
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
}

func TestStreamChatWithToolsReturnsNativeCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools []ToolDefinition `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.Tools) != 1 || body.Tools[0].Function.Name != "write_file" {
			t.Errorf("tools = %+v", body.Tools)
		}
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"write_file","arguments":"{\"path\":\"hello.py\",\"content\":\"print('hello')\"}"}}]}}]}`)
	}))
	defer server.Close()
	tools := []ToolDefinition{{Type: "function", Function: FunctionDefinition{Name: "write_file"}}}
	chunks, callsCh, errs := StreamChatWithTools(context.Background(), server.URL, []model.Message{{Role: "user", Content: "hi"}}, 0.1, 32, tools)
	for range chunks {
	}
	calls := <-callsCh
	if len(calls) != 1 || calls[0].Name != "write_file" || calls[0].NativeID != "call_1" {
		t.Fatalf("calls = %+v", calls)
	}
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
}
