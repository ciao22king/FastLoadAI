// Package safetensors reads Hugging Face safetensors checkpoints and writes a
// self-contained GGUF file, so FastLoadAI can run a model even when no GGUF is
// available yet.
package safetensors

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Config is the part of a Hugging Face config.json that the converter needs.
type Config struct {
	Architectures         []string `json:"architectures"`
	ModelType             string   `json:"model_type"`
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	VocabSize             int      `json:"vocab_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	RMSNormEPS            float64  `json:"rms_norm_eps"`
	RopeTheta             float64  `json:"rope_theta"`
	TieWordEmbeddings     bool     `json:"tie_word_embeddings"`
	TorchDtype            string   `json:"torch_dtype"`
	NameOrPath            string   `json:"_name_or_path"`
}

// Tensor is one tensor inside a safetensors shard.
type Tensor struct {
	Name      string
	Dtype     string
	Shape     []int64
	Shard     *Shard
	DataStart int64
	DataEnd   int64
}

// ByteSize returns the number of bytes stored for the tensor.
func (t *Tensor) ByteSize() (int64, error) {
	elements := int64(1)
	for _, dimension := range t.Shape {
		if dimension < 0 {
			return 0, fmt.Errorf("tensor %s has a negative dimension", t.Name)
		}
		elements *= dimension
	}
	size, err := dtypeSize(t.Dtype)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", t.Name, err)
	}
	return elements * size, nil
}

// Shard is one opened safetensors file.
type Shard struct {
	Path string
	file *os.File
}

// Model is a loaded Hugging Face checkpoint directory.
type Model struct {
	Dir       string
	Config    Config
	Tensors   map[string]*Tensor
	Tokenizer *Tokenizer
	shards    []*Shard
}

// Close releases the safetensors file handles.
func (m *Model) Close() {
	for _, shard := range m.shards {
		if shard.file != nil {
			shard.file.Close()
		}
	}
}

// IsModelDir reports whether dir contains a config and at least one safetensors file.
func IsModelDir(dir string) bool {
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(matches) > 0
}

// Load reads config.json, tokenizer metadata and every safetensors shard.
func Load(dir string) (*Model, error) {
	configBytes, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("read config.json: %w", err)
	}
	var config Config
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return nil, fmt.Errorf("parse config.json: %w", err)
	}
	if config.ModelType == "" && len(config.Architectures) > 0 {
		config.ModelType = strings.ToLower(strings.TrimSuffix(config.Architectures[0], "ForCausalLM"))
	}
	if config.NumKeyValueHeads <= 0 {
		config.NumKeyValueHeads = config.NumAttentionHeads
	}
	if config.RopeTheta <= 0 {
		config.RopeTheta = 10000
	}
	if config.RMSNormEPS <= 0 {
		config.RMSNormEPS = 1e-6
	}
	model := &Model{Dir: dir, Config: config, Tensors: map[string]*Tensor{}}
	shards, err := shardFiles(dir)
	if err != nil {
		model.Close()
		return nil, err
	}
	for _, path := range shards {
		if err := model.loadShard(path); err != nil {
			model.Close()
			return nil, err
		}
	}
	tokenizer, err := loadTokenizer(dir, config)
	if err != nil {
		model.Close()
		return nil, err
	}
	model.Tokenizer = tokenizer
	if config.VocabSize <= 0 {
		config.VocabSize = len(tokenizer.Tokens)
	}
	model.Config = config
	return model, nil
}

type safetensorsEntry struct {
	Dtype       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

func (m *Model) loadShard(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", filepath.Base(path), err)
	}
	shard := &Shard{Path: path, file: file}
	m.shards = append(m.shards, shard)
	headerLength, err := readU64At(file, 0)
	if err != nil {
		return fmt.Errorf("read %s header length: %w", filepath.Base(path), err)
	}
	if headerLength == 0 || headerLength > 64<<20 {
		return fmt.Errorf("%s has an invalid header length %d", filepath.Base(path), headerLength)
	}
	header := make([]byte, headerLength)
	if _, err := file.ReadAt(header, 8); err != nil {
		return fmt.Errorf("read %s header: %w", filepath.Base(path), err)
	}
	var entries map[string]json.RawMessage
	if err := json.Unmarshal(header, &entries); err != nil {
		return fmt.Errorf("parse %s header: %w", filepath.Base(path), err)
	}
	base := int64(8) + int64(headerLength)
	info, err := file.Stat()
	if err != nil {
		return err
	}
	for name, raw := range entries {
		if name == "__metadata__" {
			continue
		}
		var entry safetensorsEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return fmt.Errorf("parse tensor %s: %w", name, err)
		}
		start := base + entry.DataOffsets[0]
		end := base + entry.DataOffsets[1]
		if start < base || end < start || end > info.Size() {
			return fmt.Errorf("tensor %s has out-of-range offsets", name)
		}
		tensor := &Tensor{
			Name: name, Dtype: strings.ToUpper(entry.Dtype), Shape: entry.Shape,
			Shard: shard, DataStart: start, DataEnd: end,
		}
		expected, err := tensor.ByteSize()
		if err != nil {
			return err
		}
		if expected != end-start {
			return fmt.Errorf("tensor %s has %d bytes, expected %d", name, end-start, expected)
		}
		m.Tensors[name] = tensor
	}
	return nil
}

func shardFiles(dir string) ([]string, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if raw, err := os.ReadFile(indexPath); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &index); err != nil {
			return nil, fmt.Errorf("parse model.safetensors.index.json: %w", err)
		}
		seen := map[string]bool{}
		var files []string
		for _, name := range index.WeightMap {
			if !seen[name] {
				seen[name] = true
				files = append(files, filepath.Join(dir, name))
			}
		}
		sort.Strings(files)
		if len(files) > 0 {
			return files, nil
		}
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(matches) == 0 {
		return nil, errors.New("no .safetensors files found")
	}
	sort.Strings(matches)
	return matches, nil
}

func readU64At(file *os.File, offset int64) (uint64, error) {
	var buffer [8]byte
	if _, err := file.ReadAt(buffer[:], offset); err != nil {
		return 0, err
	}
	return uint64(buffer[0]) | uint64(buffer[1])<<8 | uint64(buffer[2])<<16 | uint64(buffer[3])<<24 |
		uint64(buffer[4])<<32 | uint64(buffer[5])<<40 | uint64(buffer[6])<<48 | uint64(buffer[7])<<56, nil
}

func dtypeSize(dtype string) (int64, error) {
	switch strings.ToUpper(dtype) {
	case "F64":
		return 8, nil
	case "F32", "I32":
		return 4, nil
	case "F16", "BF16", "I16":
		return 2, nil
	case "I64":
		return 8, nil
	case "I8", "U8", "BOOL":
		return 1, nil
	default:
		return 0, fmt.Errorf("unsupported safetensors dtype %q", dtype)
	}
}
