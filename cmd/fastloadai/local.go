package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/hf"
	"github.com/ciao22king/FastLoadAI/internal/model"
	"github.com/ciao22king/FastLoadAI/internal/safetensors"
)

// LocalGroup is a downloaded GGUF file, or all shards of one model.
type LocalGroup struct {
	Name         string   `json:"name"`
	Path         string   `json:"path"`
	Files        []string `json:"files"`
	SizeBytes    int64    `json:"size_bytes"`
	Quantization string   `json:"quantization"`
}

var localShardRE = regexp.MustCompile(`(?i)-[0-9]{5}-of-[0-9]{5}[.]gguf$`)

func localGGUFGroups(root string) ([]LocalGroup, error) {
	if _, err := os.Stat(root); errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	groups := map[string]*LocalGroup{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
			return nil
		}
		key := localShardRE.ReplaceAllString(entry.Name(), ".gguf")
		group := groups[key]
		if group == nil {
			group = &LocalGroup{Name: key, Quantization: hf.QuantLabel(entry.Name())}
			groups[key] = group
		}
		group.Files = append(group.Files, path)
		if info, statErr := entry.Info(); statErr == nil {
			group.SizeBytes += info.Size()
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	result := make([]LocalGroup, 0, len(groups))
	for _, group := range groups {
		sort.Strings(group.Files)
		group.Path = group.Files[0]
		result = append(result, *group)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Path < result[j].Path })
	return result, nil
}

func isRepoReference(value string) bool {
	if value == "" || strings.HasSuffix(strings.ToLower(value), ".gguf") {
		return false
	}
	if info, err := os.Stat(value); err == nil && info.Mode().IsRegular() {
		return false
	}
	return strings.Contains(value, "/")
}

func resolveModel(requested string, cfg model.Config, modelsDir string) (string, error) {
	configured := ""
	if cfg.ModelPath != "" {
		configured = absModelPath(cfg.ModelPath)
	}
	if requested == "" {
		if configured == "" {
			return "", errors.New("no model configured: pass a model name or run download first")
		}
		if info, err := os.Stat(configured); err != nil || !info.Mode().IsRegular() {
			return "", fmt.Errorf("configured model not found: %s", configured)
		}
		return configured, nil
	}
	if info, err := os.Stat(requested); err == nil {
		if info.Mode().IsRegular() {
			return absModelPath(requested), nil
		}
		if info.IsDir() && safetensors.IsModelDir(requested) {
			return absModelPath(requested), nil
		}
	}
	query := strings.ToLower(requested)
	if configured != "" {
		if info, err := os.Stat(configured); err == nil && info.Mode().IsRegular() {
			if strings.Contains(strings.ToLower(filepath.Base(configured)), query) || strings.Contains(strings.ToLower(cfg.ModelRepo), query) {
				return configured, nil
			}
		}
	}
	groups, err := localGGUFGroups(modelsDir)
	if err != nil {
		return "", err
	}
	var matches []LocalGroup
	for _, group := range groups {
		if strings.Contains(strings.ToLower(group.Path), query) || strings.Contains(strings.ToLower(group.Name), query) {
			matches = append(matches, group)
		}
	}
	if len(matches) == 1 {
		return absModelPath(matches[0].Path), nil
	}
	if len(matches) > 1 {
		var choices []string
		for _, match := range matches {
			choices = append(choices, "  "+match.Path)
		}
		return "", fmt.Errorf("ambiguous model name %q, choose one of:\n%s", requested, strings.Join(choices, "\n"))
	}
	return "", fmt.Errorf("model %q was not found in the config or in %s", requested, modelsDir)
}
