// Package catalog contains small GGUF models that fit modest machines.
package catalog

import (
	"regexp"
	"sort"
	"strings"

	"github.com/ciao22king/FastLoadAI/internal/hardware"
	"github.com/ciao22king/FastLoadAI/internal/model"
)

// Model is one curated GGUF repository.
type Model struct {
	Name   string  `json:"name"`
	Repo   string  `json:"repo"`
	SizeGB float64 `json:"size_gb"`
	Note   string  `json:"note"`
}

// Row is a catalog entry annotated with fit and recommendation state.
type Row struct {
	Name        string  `json:"name"`
	Repo        string  `json:"repo"`
	SizeGB      float64 `json:"size_gb"`
	Note        string  `json:"note"`
	Fit         string  `json:"fit"`
	Recommended bool    `json:"recommended"`
}

var Catalog = []Model{
	{Name: "SmolLM2 135M Instruct", Repo: "unsloth/SmolLM2-135M-Instruct-GGUF", SizeGB: 0.11, Note: "tiny, fastest"},
	{Name: "SmolLM2 360M Instruct", Repo: "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", SizeGB: 0.25, Note: "very small"},
	{Name: "Qwen2.5 0.5B Instruct", Repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", SizeGB: 0.46, Note: "small and capable"},
	{Name: "Llama 3.2 1B Instruct", Repo: "unsloth/Llama-3.2-1B-Instruct-GGUF", SizeGB: 0.80, Note: "good balance"},
	{Name: "Qwen2.5 1.5B Instruct", Repo: "Qwen/Qwen2.5-1.5B-Instruct-GGUF", SizeGB: 1.00, Note: "better quality"},
	{Name: "SmolLM2 1.7B Instruct", Repo: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF", SizeGB: 1.10, Note: "small general model"},
	{Name: "Llama 3.2 3B Instruct", Repo: "unsloth/Llama-3.2-3B-Instruct-GGUF", SizeGB: 2.00, Note: "needs 6 GB+"},
	{Name: "Qwen2.5 3B Instruct", Repo: "Qwen/Qwen2.5-3B-Instruct-GGUF", SizeGB: 2.00, Note: "needs 6 GB+"},
	{Name: "Qwen2.5 7B Instruct", Repo: "Qwen/Qwen2.5-7B-Instruct-GGUF", SizeGB: 4.70, Note: "needs 10 GB+"},
	{Name: "Llama 3.1 8B Instruct", Repo: "unsloth/Llama-3.1-8B-Instruct-GGUF", SizeGB: 5.00, Note: "needs 12 GB+"},
}

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

func normalizeName(value string) string {
	return strings.TrimSpace(nonAlnum.ReplaceAllString(strings.ToLower(value), " "))
}

// Match resolves a short name such as qwen2.5-0.5b to one catalog model.
func Match(query string) (Model, bool) {
	normalized := normalizeName(query)
	if normalized == "" {
		return Model{}, false
	}
	var matches []Model
	for _, item := range Catalog {
		if strings.Contains(normalizeName(item.Name), normalized) ||
			strings.Contains(normalizeName(item.Repo), normalized) {
			matches = append(matches, item)
		}
	}
	if len(matches) != 1 {
		return Model{}, false
	}
	return matches[0], true
}

// Rows annotates every catalog model with its fit and one recommendation.
func Rows(specs model.Specs) []Row {
	rows := make([]Row, 0, len(Catalog))
	for _, item := range Catalog {
		rows = append(rows, Row{
			Name:   item.Name,
			Repo:   item.Repo,
			SizeGB: item.SizeGB,
			Note:   item.Note,
			Fit:    hardware.ModelFit(specs, item.SizeGB),
		})
	}
	cpuCap := 5.0
	switch {
	case specs.CPUThreads <= 2:
		cpuCap = 0.5
	case specs.CPUThreads <= 4:
		cpuCap = 1.0
	case specs.CPUThreads <= 8:
		cpuCap = 2.0
	}
	best := -1
	for index, row := range rows {
		if row.Fit == "comfortable" && row.SizeGB <= cpuCap {
			if best < 0 || row.SizeGB > rows[best].SizeGB {
				best = index
			}
		}
	}
	if best >= 0 {
		rows[best].Recommended = true
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].SizeGB < rows[j].SizeGB })
	return rows
}

// Budget returns the usable model memory for this machine.
func Budget(specs model.Specs) float64 {
	return hardware.UsableModelMemoryGB(specs)
}
