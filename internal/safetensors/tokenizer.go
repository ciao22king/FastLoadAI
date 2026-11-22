package safetensors

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Tokenizer is the GGUF-compatible vocabulary extracted from tokenizer.json.
type Tokenizer struct {
	Tokens       []string
	Types        []int32
	Scores       []float32
	Merges       []string
	BosID        uint32
	EosID        uint32
	PadID        uint32
	AddBOS       bool
	AddEOS       bool
	ChatTemplate string
	Pre          string
}

type tokenizerFile struct {
	Model struct {
		Type   string            `json:"type"`
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

type tokenizerConfig struct {
	ChatTemplate string          `json:"chat_template"`
	AddBOSToken  bool            `json:"add_bos_token"`
	AddEOSToken  bool            `json:"add_eos_token"`
	BOSToken     json.RawMessage `json:"bos_token"`
	EOSToken     json.RawMessage `json:"eos_token"`
	PadToken     json.RawMessage `json:"pad_token"`
}

func loadTokenizer(dir string, config Config) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	var file tokenizerFile
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	if !strings.EqualFold(file.Model.Type, "BPE") {
		return nil, fmt.Errorf("unsupported tokenizer type %q: only byte-level BPE is supported", file.Model.Type)
	}
	vocabSize := config.VocabSize
	maxID := -1
	for _, id := range file.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, token := range file.AddedTokens {
		if token.ID > maxID {
			maxID = token.ID
		}
	}
	if vocabSize <= maxID {
		vocabSize = maxID + 1
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("tokenizer has no vocabulary")
	}
	tokenizer := &Tokenizer{
		Tokens: make([]string, vocabSize),
		Types:  make([]int32, vocabSize),
		Scores: make([]float32, vocabSize),
		Pre:    tokenizerPre(config.ModelType),
	}
	for id := range tokenizer.Tokens {
		tokenizer.Tokens[id] = fmt.Sprintf("<|reserved_special_token_%d|>", id)
		tokenizer.Types[id] = 5
	}
	for token, id := range file.Model.Vocab {
		if id < 0 || id >= vocabSize {
			return nil, fmt.Errorf("token %q has out-of-range id %d", token, id)
		}
		tokenizer.Tokens[id] = token
		tokenizer.Types[id] = 1
	}
	for _, added := range file.AddedTokens {
		if added.ID < 0 || added.ID >= vocabSize {
			return nil, fmt.Errorf("added token %q has out-of-range id %d", added.Content, added.ID)
		}
		tokenizer.Tokens[added.ID] = added.Content
		if added.Special || isControlToken(added.Content) {
			tokenizer.Types[added.ID] = 3
		} else {
			tokenizer.Types[added.ID] = 4
		}
	}
	for _, merge := range file.Model.Merges {
		var text string
		if err := json.Unmarshal(merge, &text); err == nil {
			tokenizer.Merges = append(tokenizer.Merges, text)
			continue
		}
		var parts []string
		if err := json.Unmarshal(merge, &parts); err == nil && len(parts) == 2 {
			tokenizer.Merges = append(tokenizer.Merges, parts[0]+" "+parts[1])
		}
	}
	configBytes, _ := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	var tokConfig tokenizerConfig
	if len(configBytes) > 0 {
		if err := json.Unmarshal(configBytes, &tokConfig); err != nil {
			return nil, fmt.Errorf("parse tokenizer_config.json: %w", err)
		}
	}
	tokenizer.ChatTemplate = tokConfig.ChatTemplate
	tokenizer.AddBOS = tokConfig.AddBOSToken
	tokenizer.AddEOS = tokConfig.AddEOSToken
	tokenizer.BosID = uint32(config.VocabSize)
	if config.ModelType == "qwen2" || config.ModelType == "qwen3" {
		tokenizer.BosID = 151643
	}
	tokenizer.EosID = tokenizer.BosID
	tokenizer.PadID = tokenizer.BosID
	if id, ok := tokenizer.idForRaw(tokConfig.BOSToken); ok {
		tokenizer.BosID = uint32(id)
	}
	if id, ok := tokenizer.idForRaw(tokConfig.EOSToken); ok {
		tokenizer.EosID = uint32(id)
	}
	if id, ok := tokenizer.idForRaw(tokConfig.PadToken); ok {
		tokenizer.PadID = uint32(id)
	}
	if config.ModelType == "qwen2" {
		tokenizer.BosID, tokenizer.EosID, tokenizer.PadID = 151643, 151645, 151643
	}
	return tokenizer, nil
}

func (t *Tokenizer) idForRaw(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return t.idForToken(text)
	}
	var object struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &object); err == nil && object.Content != "" {
		return t.idForToken(object.Content)
	}
	return 0, false
}

func (t *Tokenizer) idForToken(token string) (int, bool) {
	for id, candidate := range t.Tokens {
		if candidate == token {
			return id, true
		}
	}
	return 0, false
}

func isControlToken(token string) bool {
	if strings.HasPrefix(token, "<|") || token == "</s>" || token == "<s>" {
		return true
	}
	return strings.HasPrefix(token, "<|endoftext|>")
}

func tokenizerPre(modelType string) string {
	switch strings.ToLower(modelType) {
	case "qwen2", "qwen3":
		return "qwen2"
	case "llama":
		return "llama-bpe"
	default:
		return ""
	}
}
