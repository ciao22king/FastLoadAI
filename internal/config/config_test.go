package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestWriteAndLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "config.json")
	want := model.Config{CPUThreads: 4, ContextSize: 2048, Mmap: true, ModelPath: "/tmp/model.gguf"}
	if err := Write(path, want); err != nil {
		t.Fatal(err)
	}
	got, exists, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !exists || got.CPUThreads != 4 || got.ContextSize != 2048 || !got.Mmap || got.ModelPath != want.ModelPath {
		t.Fatalf("loaded %+v exists=%v", got, exists)
	}
}

func TestLoadMissing(t *testing.T) {
	_, exists, err := Load(filepath.Join(t.TempDir(), "missing.json"))
	if err != nil || exists {
		t.Fatalf("exists=%v err=%v", exists, err)
	}
}

func TestLoadInvalid(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(path); err == nil {
		t.Fatal("expected a JSON error")
	}
}
