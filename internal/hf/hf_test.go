package hf

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func size(value int64) *int64 { return &value }

func TestGGUFGroupsShards(t *testing.T) {
	groups := GGUFGroups([]Sibling{
		{Filename: "model-Q4_K_M-00002-of-00002.gguf", Size: size(20)},
		{Filename: "model-Q4_K_M-00001-of-00002.gguf", Size: size(10)},
		{Filename: "other-Q4_K_M.gguf", Size: size(5)},
		{Filename: "README.md"},
	})
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	if groups[0].Name != "model-Q4_K_M.gguf" || len(groups[0].Files) != 2 {
		t.Fatalf("unexpected shard group: %+v", groups[0])
	}
	if groups[0].Size != 30 || !groups[0].SizeKnown {
		t.Fatalf("size = %d known=%v, want 30 true", groups[0].Size, groups[0].SizeKnown)
	}
}

func TestSelectGGUF(t *testing.T) {
	groups := []Group{
		{Name: "m-Q2_K.gguf", Size: 2 << 30, SizeKnown: true},
		{Name: "m-Q4_K_M.gguf", Size: 4 << 30, SizeKnown: true},
		{Name: "m-Q8_0.gguf", Size: 8 << 30, SizeKnown: true},
	}
	selected, err := SelectGGUF(groups, 5, "", "balanced")
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name != "m-Q4_K_M.gguf" {
		t.Fatalf("selected %s, want m-Q4_K_M.gguf", selected.Name)
	}
	selected, err = SelectGGUF(groups, 1, "", "balanced")
	if err != nil {
		t.Fatal(err)
	}
	if selected.Name != "m-Q2_K.gguf" {
		t.Fatalf("fallback selected %s, want m-Q2_K.gguf", selected.Name)
	}
	selected, err = SelectGGUF(groups, 1, "m-Q8_0.gguf", "balanced")
	if err != nil || selected.Name != "m-Q8_0.gguf" {
		t.Fatalf("explicit selection failed: %v %+v", err, selected)
	}
	if _, err := SelectGGUF(groups, 1, "missing.gguf", "balanced"); err == nil {
		t.Fatal("expected an error for an unknown file")
	}
}

func TestQuantLabelAndRank(t *testing.T) {
	cases := map[string]string{
		"qwen2.5-0.5b-instruct-q4_k_m.gguf": "Q4_K_M",
		"llama-3.2-1b-instruct-q8_0.gguf":   "Q8_0",
		"model-f16.gguf":                    "F16",
		"plain.gguf":                        "plain",
	}
	for name, want := range cases {
		if got := QuantLabel(name); got != want {
			t.Errorf("QuantLabel(%q) = %q, want %q", name, got, want)
		}
	}
	if QuantRank("m-q2_k.gguf") >= QuantRank("m-q8_0.gguf") {
		t.Fatal("Q2 should rank below Q8")
	}
}

func TestDownloadGroup(t *testing.T) {
	files := map[string]string{
		"/org/repo/resolve/main/a-00001-of-00002.gguf": "first",
		"/org/repo/resolve/main/a-00002-of-00002.gguf": "second",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content, ok := files[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, content)
	}))
	defer server.Close()

	client := NewClient("")
	client.BaseURL = server.URL
	dir := t.TempDir()
	group := Group{
		Name: "a.gguf",
		Files: []string{
			"a-00001-of-00002.gguf", "a-00002-of-00002.gguf",
		},
		FileSizes: map[string]int64{
			"a-00001-of-00002.gguf": 5, "a-00002-of-00002.gguf": 6,
		},
	}
	paths, err := client.DownloadGroup(context.Background(), "org/repo", "main", dir, group, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 {
		t.Fatalf("got %d paths", len(paths))
	}
	for index, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		want := files[fmt.Sprintf("/org/repo/resolve/main/%s", group.Files[index])]
		if string(data) != want {
			t.Fatalf("%s = %q, want %q", filepath.Base(path), data, want)
		}
	}
}
