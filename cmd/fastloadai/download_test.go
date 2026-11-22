package main

import (
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/hf"
)

func TestSelectSafetensorsGroup(t *testing.T) {
	size := int64(1024)
	group, err := selectSafetensorsGroup([]hf.Sibling{
		{Filename: "README.md", Size: &size},
		{Filename: "config.json", Size: &size},
		{Filename: "tokenizer.json", Size: &size},
		{Filename: "model.safetensors", Size: &size},
	})
	if err != nil {
		t.Fatalf("selectSafetensorsGroup: %v", err)
	}
	if len(group.Files) != 3 {
		t.Fatalf("files = %v", group.Files)
	}
	if group.Size != 3*size || !group.SizeKnown {
		t.Fatalf("size = %d known = %v", group.Size, group.SizeKnown)
	}
}

func TestSelectSafetensorsGroupUnknownSize(t *testing.T) {
	group, err := selectSafetensorsGroup([]hf.Sibling{
		{Filename: "config.json"},
		{Filename: "model.safetensors"},
	})
	if err != nil {
		t.Fatalf("selectSafetensorsGroup: %v", err)
	}
	if group.SizeKnown {
		t.Fatal("unknown size should be reported")
	}
}

func TestSelectSafetensorsGroupWithoutWeights(t *testing.T) {
	if _, err := selectSafetensorsGroup([]hf.Sibling{{Filename: "config.json"}}); err == nil {
		t.Fatal("expected an error for a repository without weights")
	}
}
