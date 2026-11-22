package adaptive

import "sort"

// Candidate is one model that the router may choose.
type Candidate struct {
	Name   string  `json:"name"`
	Repo   string  `json:"repo,omitempty"`
	Path   string  `json:"path,omitempty"`
	SizeGB float64 `json:"size_gb"`
	Local  bool    `json:"local,omitempty"`
}

// SelectModel picks the largest model that fits the effort tier and the memory
// budget. Light prompts are deliberately capped to a small model; deep prompts
// may use the whole budget. Local models win ties because they need no download.
func SelectModel(candidates []Candidate, budgetGB float64, effort Effort) (Candidate, bool) {
	if len(candidates) == 0 || budgetGB <= 0 {
		return Candidate{}, false
	}
	limit := budgetGB * 0.95
	switch effort {
	case EffortLight:
		limit = minFloat(1.5, budgetGB*0.35)
	case EffortStandard:
		limit = minFloat(8, budgetGB*0.45)
	case EffortDeep:
		limit = budgetGB * 0.95
	}
	eligible := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.SizeGB > 0 && candidate.SizeGB <= limit {
			eligible = append(eligible, candidate)
		}
	}
	if len(eligible) == 0 {
		// A very small machine may have no model below the light cap. Fall back
		// to the smallest model that still fits the real budget.
		for _, candidate := range candidates {
			if candidate.SizeGB > 0 && candidate.SizeGB <= budgetGB {
				eligible = append(eligible, candidate)
			}
		}
	}
	if len(eligible) == 0 {
		return Candidate{}, false
	}
	sort.SliceStable(eligible, func(i, j int) bool {
		if eligible[i].SizeGB != eligible[j].SizeGB {
			return eligible[i].SizeGB > eligible[j].SizeGB
		}
		return eligible[i].Local && !eligible[j].Local
	})
	return eligible[0], true
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
