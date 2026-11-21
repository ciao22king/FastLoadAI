package hardware

import (
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestModelFitAndBudget(t *testing.T) {
	specs := model.Specs{CPUThreads: 8, RAMGB: 16, AvailableRAMGB: 8}
	budget := UsableModelMemoryGB(specs)
	if budget < 4.7 || budget > 4.9 {
		t.Fatalf("budget = %.2f, want about 4.8", budget)
	}
	if got := ModelFit(specs, 1); got != "comfortable" {
		t.Fatalf("1 GB fit = %s", got)
	}
	if got := ModelFit(specs, 4.5); got != "tight" {
		t.Fatalf("4.5 GB fit = %s", got)
	}
	if got := ModelFit(specs, 6); got != "too_large" {
		t.Fatalf("6 GB fit = %s", got)
	}
	if got := ModelFit(specs, 0); got != "unknown" {
		t.Fatalf("0 GB fit = %s", got)
	}
}

func TestRecommendedContextSize(t *testing.T) {
	specs := model.Specs{CPUThreads: 8, RAMGB: 16, AvailableRAMGB: 8}
	if got := RecommendedContextSize(specs, 1); got != 2048 {
		t.Fatalf("context = %d, want 2048", got)
	}
	small := model.Specs{CPUThreads: 2, RAMGB: 4, AvailableRAMGB: 2}
	if got := RecommendedContextSize(small, 1); got != 1024 {
		t.Fatalf("small context = %d, want 1024", got)
	}
}

func TestDetectReportsUsableHardware(t *testing.T) {
	specs := Detect()
	if specs.CPUThreads < 1 {
		t.Fatalf("CPU threads = %d", specs.CPUThreads)
	}
	if specs.RAMGB <= 0 {
		t.Fatalf("RAM = %.2f", specs.RAMGB)
	}
	if specs.PhysicalCores < 1 {
		t.Fatalf("physical cores = %d", specs.PhysicalCores)
	}
}

func TestRecommendedThreads(t *testing.T) {
	cases := []struct {
		specs model.Specs
		want  int
	}{
		{model.Specs{}, 1},
		{model.Specs{CPUThreads: 4}, 4},
		{model.Specs{CPUThreads: 32, PhysicalCores: 16}, 32},
		{model.Specs{CPUThreads: 4, PhysicalCores: 8}, 4},
		{model.Specs{CPUThreads: 64, PhysicalCores: 64}, 32},
	}
	for _, item := range cases {
		if got := RecommendedThreads(item.specs); got != item.want {
			t.Fatalf("RecommendedThreads(%+v) = %d, want %d", item.specs, got, item.want)
		}
	}
}
