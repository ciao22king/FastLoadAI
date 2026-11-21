// Package config reads and writes the FastLoadAI runtime plan.
package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

// DefaultPath is the config file used when --config is omitted.
const DefaultPath = "fastloadai-config.json"

// Load reads a config. The boolean is false when the file does not exist.
func Load(path string) (model.Config, bool, error) {
	var cfg model.Config
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return cfg, false, nil
	}
	if err != nil {
		return cfg, false, err
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, true, err
	}
	return cfg, true, nil
}

// Write stores the config as indented JSON.
func Write(path string, cfg model.Config) error {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}
