package chat

import (
	"context"
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
