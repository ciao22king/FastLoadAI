package adaptive

import (
	"fmt"
	"strings"
	"unicode"
)

// Effort is how much model work a prompt deserves.
type Effort string

const (
	EffortAuto     Effort = "auto"
	EffortLight    Effort = "light"
	EffortStandard Effort = "standard"
	EffortDeep     Effort = "deep"
)

// ParseEffort validates an effort name.
func ParseEffort(value string) (Effort, error) {
	switch Effort(strings.ToLower(strings.TrimSpace(value))) {
	case "", EffortAuto:
		return EffortAuto, nil
	case EffortLight:
		return EffortLight, nil
	case EffortStandard:
		return EffortStandard, nil
	case EffortDeep:
		return EffortDeep, nil
	default:
		return "", fmt.Errorf("unknown effort %q (use auto, light, standard or deep)", value)
	}
}

var deepMarkers = []string{
	"step by step", "chain of thought", "prove", "derive", "architect",
	"design", "refactor", "debug", "root cause", "compare", "trade-off",
	"algorithm", "complexity", "implement", "write code", "write a program",
	"multi-step", "reason", "analyze", "analyse", "optimize", "optimise",
	"sql", "regex", "mathematical", "calculate", "plan", "migrate",
}

var lightMarkers = []string{
	"hello", "hi", "hey", "ciao", "thanks", "thank you", "good morning",
	"good night", "who are you", "what is your name", "translate",
	"one sentence", "briefly", "short answer", "define", "definition",
	"capital of", "how are you",
}

// ClassifyPrompt estimates the reasoning effort from the prompt alone.
// It deliberately stays cheap: no tokenizer or model call is needed.
func ClassifyPrompt(prompt string) Effort {
	text := strings.TrimSpace(prompt)
	if text == "" {
		return EffortStandard
	}
	lower := strings.ToLower(text)
	words := len(strings.Fields(text))
	score := 0
	switch {
	case words > 80:
		score += 3
	case words > 30:
		score += 2
	case words > 12:
		score += 1
	}
	if strings.Contains(lower, "```") {
		score += 2
	}
	if strings.Count(lower, "?") > 1 {
		score++
	}
	for _, marker := range deepMarkers {
		if containsMarker(lower, marker) {
			score += 2
			break
		}
	}
	if words <= 16 {
		for _, marker := range lightMarkers {
			if containsMarker(lower, marker) {
				score -= 2
				break
			}
		}
	}
	switch {
	case score >= 3:
		return EffortDeep
	case score <= 0:
		return EffortLight
	default:
		return EffortStandard
	}
}

// ResolveEffort turns auto into a concrete tier using the prompt.
func ResolveEffort(requested Effort, prompt string) Effort {
	if requested == "" || requested == EffortAuto {
		return ClassifyPrompt(prompt)
	}
	return requested
}

// containsMarker matches phrases as substrings and single words as whole words,
// so "hi" does not match "this" and "plan" does not match "planet".
func containsMarker(text, marker string) bool {
	if strings.Contains(marker, " ") {
		return strings.Contains(text, marker)
	}
	for _, word := range strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if word == marker {
			return true
		}
	}
	return false
}
