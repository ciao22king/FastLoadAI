// Package adaptive sizes a GGUF model to the memory that is actually free.
package adaptive

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	ggufMagic = "GGUF"

	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

// ModelProfile is the part of the GGUF metadata needed to plan memory use.
type ModelProfile struct {
	Path          string `json:"path,omitempty"`
	Architecture  string `json:"architecture,omitempty"`
	Name          string `json:"name,omitempty"`
	Parameters    int64  `json:"parameters"`
	Layers        int    `json:"layers"`
	HiddenSize    int    `json:"hidden_size"`
	Heads         int    `json:"heads"`
	KVHeads       int    `json:"kv_heads"`
	KeyLength     int    `json:"key_length"`
	ValueLength   int    `json:"value_length"`
	ContextLength int    `json:"context_length"`
	Experts       int    `json:"experts,omitempty"`
	ExpertsUsed   int    `json:"experts_used,omitempty"`
	FileBytes     int64  `json:"file_bytes"`
	Quantization  string `json:"quantization,omitempty"`
	Source        string `json:"source,omitempty"`
}

// LoadProfile reads GGUF metadata and falls back to a file-size estimate when
// the file is not a complete GGUF (for example, one shard of a split model).
func LoadProfile(path string) ModelProfile {
	size := int64(0)
	if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() {
		size = info.Size()
	}
	profile, err := ReadGGUF(path)
	if err != nil {
		profile = EstimateProfile(path, size)
	}
	profile.Path = path
	profile.FileBytes = size
	profile.Quantization = QuantLabel(path)
	if profile.Parameters <= 0 && size > 0 {
		profile.Parameters = estimateParameters(size, profile.Quantization)
	}
	if profile.Layers <= 0 || profile.KVHeads <= 0 || profile.HiddenSize <= 0 {
		fallback := EstimateProfile(path, size)
		if profile.Layers <= 0 {
			profile.Layers = fallback.Layers
		}
		if profile.HiddenSize <= 0 {
			profile.HiddenSize = fallback.HiddenSize
		}
		if profile.Heads <= 0 {
			profile.Heads = fallback.Heads
		}
		if profile.KVHeads <= 0 {
			profile.KVHeads = fallback.KVHeads
		}
		if profile.KeyLength <= 0 {
			profile.KeyLength = fallback.KeyLength
		}
		if profile.ValueLength <= 0 {
			profile.ValueLength = fallback.ValueLength
		}
	}
	if profile.Source == "" {
		profile.Source = "estimate"
	}
	return profile
}

// ReadGGUF parses the metadata and tensor shapes of a GGUF file.
func ReadGGUF(path string) (ModelProfile, error) {
	file, err := os.Open(path)
	if err != nil {
		return ModelProfile{}, err
	}
	defer file.Close()
	reader := bufio.NewReaderSize(file, 1<<20)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(reader, magic); err != nil {
		return ModelProfile{}, err
	}
	if string(magic) != ggufMagic {
		return ModelProfile{}, errors.New("not a GGUF file")
	}
	if _, err := readU32(reader); err != nil {
		return ModelProfile{}, err
	}
	tensorCount, err := readU64(reader)
	if err != nil {
		return ModelProfile{}, err
	}
	metadataCount, err := readU64(reader)
	if err != nil {
		return ModelProfile{}, err
	}
	if metadataCount > 1<<20 || tensorCount > 1<<20 {
		return ModelProfile{}, errors.New("GGUF header is too large")
	}

	ints := map[string]int64{}
	strs := map[string]string{}
	for index := uint64(0); index < metadataCount; index++ {
		key, err := readString(reader)
		if err != nil {
			return ModelProfile{}, err
		}
		valueType, err := readU32(reader)
		if err != nil {
			return ModelProfile{}, err
		}
		switch valueType {
		case ggufUint8, ggufInt8, ggufUint16, ggufInt16, ggufUint32, ggufInt32,
			ggufUint64, ggufInt64:
			value, err := readInteger(reader, valueType)
			if err != nil {
				return ModelProfile{}, err
			}
			ints[key] = value
		case ggufString:
			value, err := readString(reader)
			if err != nil {
				return ModelProfile{}, err
			}
			strs[key] = value
		default:
			if err := skipValue(reader, valueType); err != nil {
				return ModelProfile{}, err
			}
		}
	}

	parameters := ints["general.parameter_count"]
	if parameters <= 0 {
		parameters = countTensorElements(reader, tensorCount)
	}
	architecture := strs["general.architecture"]
	layers := int(firstInt(ints, architecture, "block_count"))
	hidden := int(firstInt(ints, architecture, "embedding_length"))
	heads := int(firstInt(ints, architecture, "attention.head_count"))
	kvHeads := int(firstInt(ints, architecture, "attention.head_count_kv"))
	keyLength := int(firstInt(ints, architecture, "attention.key_length"))
	valueLength := int(firstInt(ints, architecture, "attention.value_length"))
	contextLength := int(firstInt(ints, architecture, "context_length"))
	experts := int(firstInt(ints, architecture, "expert_count"))
	expertsUsed := int(firstInt(ints, architecture, "expert_used_count"))
	if kvHeads <= 0 {
		kvHeads = heads
	}
	if keyLength <= 0 && heads > 0 {
		keyLength = hidden / heads
	}
	if valueLength <= 0 {
		valueLength = keyLength
	}
	return ModelProfile{
		Architecture:  architecture,
		Name:          strs["general.name"],
		Parameters:    parameters,
		Layers:        layers,
		HiddenSize:    hidden,
		Heads:         heads,
		KVHeads:       kvHeads,
		KeyLength:     keyLength,
		ValueLength:   valueLength,
		ContextLength: contextLength,
		Experts:       experts,
		ExpertsUsed:   expertsUsed,
		Source:        "gguf",
	}, nil
}

func countTensorElements(reader *bufio.Reader, count uint64) int64 {
	var total int64
	for index := uint64(0); index < count; index++ {
		if _, err := readString(reader); err != nil {
			return total
		}
		dimensions, err := readU32(reader)
		if err != nil || dimensions > 8 {
			return total
		}
		elements := int64(1)
		for dimension := uint32(0); dimension < dimensions; dimension++ {
			size, err := readU64(reader)
			if err != nil {
				return total
			}
			if size > 0 && elements > math.MaxInt64/int64(size) {
				return total
			}
			elements *= int64(size)
		}
		if _, err := readU32(reader); err != nil {
			return total
		}
		if _, err := readU64(reader); err != nil {
			return total
		}
		if total > math.MaxInt64-elements {
			return total
		}
		total += elements
	}
	return total
}

func firstInt(values map[string]int64, architecture, suffix string) int64 {
	if architecture != "" {
		if value, ok := values[architecture+"."+suffix]; ok {
			return value
		}
	}
	for key, value := range values {
		if strings.HasSuffix(key, "."+suffix) {
			return value
		}
	}
	return 0
}

func readInteger(reader io.Reader, valueType uint32) (int64, error) {
	switch valueType {
	case ggufUint8:
		value, err := readU8(reader)
		return int64(value), err
	case ggufInt8:
		value, err := readU8(reader)
		return int64(int8(value)), err
	case ggufUint16:
		value, err := readU16(reader)
		return int64(value), err
	case ggufInt16:
		value, err := readU16(reader)
		return int64(int16(value)), err
	case ggufUint32:
		value, err := readU32(reader)
		return int64(value), err
	case ggufInt32:
		value, err := readU32(reader)
		return int64(int32(value)), err
	case ggufUint64:
		value, err := readU64(reader)
		return int64(value), err
	case ggufInt64:
		value, err := readU64(reader)
		return int64(value), err
	}
	return 0, fmt.Errorf("unsupported GGUF integer type %d", valueType)
}

func skipValue(reader io.Reader, valueType uint32) error {
	switch valueType {
	case ggufUint8, ggufInt8, ggufBool:
		_, err := readU8(reader)
		return err
	case ggufUint16, ggufInt16:
		_, err := readU16(reader)
		return err
	case ggufUint32, ggufInt32, ggufFloat32:
		_, err := readU32(reader)
		return err
	case ggufUint64, ggufInt64, ggufFloat64:
		_, err := readU64(reader)
		return err
	case ggufString:
		_, err := readString(reader)
		return err
	case ggufArray:
		elementType, err := readU32(reader)
		if err != nil {
			return err
		}
		count, err := readU64(reader)
		if err != nil {
			return err
		}
		if count > 1<<24 {
			return errors.New("GGUF array is too large")
		}
		for index := uint64(0); index < count; index++ {
			if err := skipValue(reader, elementType); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("unsupported GGUF value type %d", valueType)
}

func readU8(reader io.Reader) (uint8, error) {
	var value uint8
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readU16(reader io.Reader) (uint16, error) {
	var value uint16
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readU32(reader io.Reader) (uint32, error) {
	var value uint32
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readU64(reader io.Reader) (uint64, error) {
	var value uint64
	err := binary.Read(reader, binary.LittleEndian, &value)
	return value, err
}

func readString(reader io.Reader) (string, error) {
	length, err := readU64(reader)
	if err != nil {
		return "", err
	}
	if length > 1<<26 {
		return "", errors.New("GGUF string is too long")
	}
	buffer := make([]byte, length)
	if _, err := io.ReadFull(reader, buffer); err != nil {
		return "", err
	}
	return string(buffer), nil
}

var quantRE = regexp.MustCompile(`(?i)(IQ\d+_[A-Z0-9_]+|Q\d+_K_[A-Z]+|Q\d+_[A-Z0-9]+|Q\d+_K|BF16|F16|F32)`)

// QuantLabel extracts a quantization name from a GGUF filename.
func QuantLabel(path string) string {
	match := quantRE.FindString(filepath.Base(path))
	return strings.ToUpper(match)
}

func estimateParameters(size int64, quant string) int64 {
	if size <= 0 {
		return 0
	}
	return int64(float64(size) / bytesPerParameter(quant))
}

func bytesPerParameter(quant string) float64 {
	switch strings.ToUpper(quant) {
	case "Q2_K":
		return 0.34
	case "Q3_K_S", "Q3_K_M", "Q3_K_L":
		return 0.43
	case "Q4_0", "Q4_1":
		return 0.56
	case "Q4_K_S", "Q4_K_M":
		return 0.61
	case "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M":
		return 0.72
	case "Q6_K":
		return 0.82
	case "Q8_0":
		return 1.06
	case "F16", "BF16":
		return 2.0
	case "F32":
		return 4.0
	}
	return 0.65
}

// EstimateProfile builds a conservative profile from the file size alone.
func EstimateProfile(path string, size int64) ModelProfile {
	quant := QuantLabel(path)
	parameters := estimateParameters(size, quant)
	layers, hidden, heads, kvHeads := 32, 4096, 32, 8
	switch {
	case parameters < 200_000_000:
		layers, hidden, heads, kvHeads = 24, 768, 12, 4
	case parameters < 500_000_000:
		layers, hidden, heads, kvHeads = 24, 896, 14, 2
	case parameters < 1_000_000_000:
		layers, hidden, heads, kvHeads = 28, 1536, 12, 2
	case parameters < 2_000_000_000:
		layers, hidden, heads, kvHeads = 24, 2048, 16, 2
	case parameters < 4_000_000_000:
		layers, hidden, heads, kvHeads = 28, 3072, 24, 8
	case parameters < 9_000_000_000:
		layers, hidden, heads, kvHeads = 32, 4096, 32, 8
	case parameters < 20_000_000_000:
		layers, hidden, heads, kvHeads = 40, 5120, 40, 8
	case parameters < 40_000_000_000:
		layers, hidden, heads, kvHeads = 48, 6144, 48, 8
	default:
		layers, hidden, heads, kvHeads = 64, 8192, 64, 8
	}
	headSize := 0
	if heads > 0 {
		headSize = hidden / heads
	}
	return ModelProfile{
		Path:          path,
		Name:          strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Parameters:    parameters,
		Layers:        layers,
		HiddenSize:    hidden,
		Heads:         heads,
		KVHeads:       kvHeads,
		KeyLength:     headSize,
		ValueLength:   headSize,
		ContextLength: 4096,
		FileBytes:     size,
		Quantization:  quant,
		Source:        "estimate",
	}
}
