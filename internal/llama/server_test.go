package llama

import (
	"strings"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestBuildCommand(t *testing.T) {
	args := BuildCommand("llama-server", "/models/a.gguf", model.Config{
		CPUThreads: 6, ContextSize: 4096, NGPULayers: 10, Mmap: false, LowMemory: true,
	}, 8123)
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--model /models/a.gguf", "--port 8123", "--threads 6", "--ctx-size 4096",
		"--n-gpu-layers 10", "--no-mmap", "--batch-size 128",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q missing %q", joined, want)
		}
	}
}

func TestAvailablePort(t *testing.T) {
	port, err := AvailablePort()
	if err != nil || port <= 0 {
		t.Fatalf("port=%d err=%v", port, err)
	}
}
