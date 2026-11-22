package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type testTensor struct {
	shape []int64
	dtype string
	data  []byte
}

func TestConvertSyntheticQwen2(t *testing.T) {
	dir := t.TempDir()
	writeTestModel(t, dir)
	output := filepath.Join(t.TempDir(), "model.gguf")
	result, err := Convert(dir, output)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	if result.Arch != "qwen2" || result.Tensors != 14 {
		t.Fatalf("unexpected result: %+v", result)
	}
	meta, tensors, data := readGGUF(t, output)
	if meta["general.architecture"] != "qwen2" {
		t.Fatalf("architecture = %v", meta["general.architecture"])
	}
	if meta["qwen2.block_count"] != uint32(1) {
		t.Fatalf("block_count = %v", meta["qwen2.block_count"])
	}
	tokens := meta["tokenizer.ggml.tokens"].([]string)
	if len(tokens) != 16 || tokens[0] != "a" || tokens[2] != "<|endoftext|>" {
		t.Fatalf("tokens = %v", tokens)
	}
	if types := meta["tokenizer.ggml.token_type"].([]int32); types[2] != 3 {
		t.Fatalf("control token type = %d", types[2])
	}
	embedding, ok := tensors["token_embd.weight"]
	if !ok {
		t.Fatal("missing token_embd.weight")
	}
	if len(embedding.dims) != 2 || embedding.dims[0] != 8 || embedding.dims[1] != 16 {
		t.Fatalf("token_embd dims = %v, want [8 16]", embedding.dims)
	}
	if _, ok := tensors["output.weight"]; ok {
		t.Fatal("tied model should not duplicate output.weight")
	}
	norm, ok := tensors["blk.0.attn_norm.weight"]
	if !ok {
		t.Fatal("missing blk.0.attn_norm.weight")
	}
	if norm.kind != ggmlF32 {
		t.Fatalf("norm kind = %d, want f32", norm.kind)
	}
	value := math.Float32frombits(binary.LittleEndian.Uint32(data[norm.offset:]))
	if math.Abs(float64(value)-1.0) > 1e-6 {
		t.Fatalf("converted norm = %v, want 1", value)
	}
	q, ok := tensors["blk.0.attn_q.weight"]
	if !ok {
		t.Fatal("missing blk.0.attn_q.weight")
	}
	if len(q.dims) != 2 || q.dims[0] != 8 || q.dims[1] != 8 {
		t.Fatalf("q dims = %v", q.dims)
	}
	embeddingData := data[embedding.offset : embedding.offset+embedding.size]
	wantEmbedding := make([]byte, 0, 16*8*2)
	for index := 0; index < 16*8; index++ {
		wantEmbedding = append(wantEmbedding, encodeBF16(float32(index)/100)...)
	}
	if string(embeddingData) != string(wantEmbedding) {
		t.Fatal("token_embd data changed")
	}
}

func TestEnsureConvertedReusesFile(t *testing.T) {
	dir := t.TempDir()
	writeTestModel(t, dir)
	models := t.TempDir()
	first, err := EnsureConverted(dir, models)
	if err != nil {
		t.Fatalf("EnsureConverted: %v", err)
	}
	info, err := os.Stat(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := EnsureConverted(dir, models)
	if err != nil {
		t.Fatalf("EnsureConverted again: %v", err)
	}
	if first != second {
		t.Fatalf("paths differ: %s vs %s", first, second)
	}
	again, err := os.Stat(second)
	if err != nil {
		t.Fatal(err)
	}
	if !again.ModTime().Equal(info.ModTime()) {
		t.Fatal("cached GGUF was rewritten")
	}
}

func TestIsModelDir(t *testing.T) {
	if IsModelDir(t.TempDir()) {
		t.Fatal("empty directory reported as model")
	}
	dir := t.TempDir()
	writeTestModel(t, dir)
	if !IsModelDir(dir) {
		t.Fatal("synthetic model not detected")
	}
}

func writeTestModel(t *testing.T, dir string) {
	t.Helper()
	config := map[string]any{
		"architectures": []string{"Qwen2ForCausalLM"}, "model_type": "qwen2",
		"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
		"num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 16,
		"max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000,
		"tie_word_embeddings": true, "torch_dtype": "bfloat16",
	}
	writeJSON(t, filepath.Join(dir, "config.json"), config)
	writeJSON(t, filepath.Join(dir, "tokenizer.json"), map[string]any{
		"model": map[string]any{
			"type": "BPE", "vocab": map[string]int{"a": 0, "b": 1, "<|endoftext|>": 2},
			"merges": []string{"a b"},
		},
		"added_tokens": []map[string]any{{"id": 2, "content": "<|endoftext|>", "special": true}},
	})
	writeJSON(t, filepath.Join(dir, "tokenizer_config.json"), map[string]any{
		"chat_template": "{{ messages }}", "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
	})
	bf := func(count int, start float32) []byte {
		out := make([]byte, 0, count*2)
		for index := 0; index < count; index++ {
			out = append(out, encodeBF16(start+float32(index)/100)...)
		}
		return out
	}
	tensors := map[string]testTensor{}
	tensors["model.embed_tokens.weight"] = testTensor{[]int64{16, 8}, "BF16", bf(16*8, 0)}
	tensors["model.norm.weight"] = testTensor{[]int64{8}, "BF16", bf(8, 1)}
	layer := func(suffix string, shape []int64, start float32) {
		count := 1
		for _, dimension := range shape {
			count *= int(dimension)
		}
		tensors["model.layers.0."+suffix] = testTensor{shape, "BF16", bf(count, start)}
	}
	layer("input_layernorm.weight", []int64{8}, 1)
	layer("post_attention_layernorm.weight", []int64{8}, 1)
	layer("self_attn.q_proj.weight", []int64{8, 8}, 0)
	layer("self_attn.k_proj.weight", []int64{4, 8}, 0)
	layer("self_attn.v_proj.weight", []int64{4, 8}, 0)
	layer("self_attn.o_proj.weight", []int64{8, 8}, 0)
	layer("self_attn.q_proj.bias", []int64{8}, 0)
	layer("self_attn.k_proj.bias", []int64{4}, 0)
	layer("self_attn.v_proj.bias", []int64{4}, 0)
	layer("mlp.gate_proj.weight", []int64{16, 8}, 0)
	layer("mlp.up_proj.weight", []int64{16, 8}, 0)
	layer("mlp.down_proj.weight", []int64{8, 16}, 0)
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSafetensors(t *testing.T, path string, tensors map[string]testTensor) {
	t.Helper()
	header := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	var payload []byte
	for name, tensor := range tensors {
		start := len(payload)
		payload = append(payload, tensor.data...)
		header[name] = map[string]any{
			"dtype": tensor.dtype, "shape": tensor.shape,
			"data_offsets": []int{start, len(payload)},
		}
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var out []byte
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(encoded)))
	out = append(out, length[:]...)
	out = append(out, encoded...)
	out = append(out, payload...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

func encodeBF16(value float32) []byte {
	bits := math.Float32bits(value)
	return []byte{byte(bits >> 16), byte(bits >> 24)}
}

type ggufTensor struct {
	dims   []uint64
	kind   uint32
	offset uint64
	size   uint64
}

func readGGUF(t *testing.T, path string) (map[string]any, map[string]ggufTensor, []byte) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw[:4]) != "GGUF" {
		t.Fatal("bad magic")
	}
	offset := 4
	version := binary.LittleEndian.Uint32(raw[offset:])
	offset += 4
	if version != 3 {
		t.Fatalf("version = %d", version)
	}
	count := binary.LittleEndian.Uint64(raw[offset:])
	offset += 8
	metaCount := binary.LittleEndian.Uint64(raw[offset:])
	offset += 8
	meta := map[string]any{}
	for index := uint64(0); index < metaCount; index++ {
		key, next := testString(raw, offset)
		offset = next
		kind := binary.LittleEndian.Uint32(raw[offset:])
		offset += 4
		value, next := testValue(raw, offset, kind)
		offset = next
		meta[key] = value
	}
	tensors := map[string]ggufTensor{}
	for index := uint64(0); index < count; index++ {
		name, next := testString(raw, offset)
		offset = next
		dimensions := binary.LittleEndian.Uint32(raw[offset:])
		offset += 4
		tensor := ggufTensor{}
		for dimension := uint32(0); dimension < dimensions; dimension++ {
			tensor.dims = append(tensor.dims, binary.LittleEndian.Uint64(raw[offset:]))
			offset += 8
		}
		tensor.kind = binary.LittleEndian.Uint32(raw[offset:])
		offset += 4
		tensor.offset = binary.LittleEndian.Uint64(raw[offset:])
		offset += 8
		tensor.size = tensorSize(tensor.dims, tensor.kind)
		tensors[name] = tensor
	}
	dataStart := (offset + ggufAlign - 1) / ggufAlign * ggufAlign
	return meta, tensors, raw[dataStart:]
}

func testString(raw []byte, offset int) (string, int) {
	length := binary.LittleEndian.Uint64(raw[offset:])
	offset += 8
	return string(raw[offset : offset+int(length)]), offset + int(length)
}

func testValue(raw []byte, offset int, kind uint32) (any, int) {
	switch kind {
	case typeUint32:
		return binary.LittleEndian.Uint32(raw[offset:]), offset + 4
	case typeInt32:
		return int32(binary.LittleEndian.Uint32(raw[offset:])), offset + 4
	case typeFloat32:
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[offset:])), offset + 4
	case typeBool:
		return raw[offset] != 0, offset + 1
	case typeString:
		return testString(raw, offset)
	case typeArray:
		element := binary.LittleEndian.Uint32(raw[offset:])
		offset += 4
		count := binary.LittleEndian.Uint64(raw[offset:])
		offset += 8
		switch element {
		case typeString:
			values := make([]string, count)
			for index := range values {
				values[index], offset = testString(raw, offset)
			}
			return values, offset
		case typeInt32:
			values := make([]int32, count)
			for index := range values {
				values[index] = int32(binary.LittleEndian.Uint32(raw[offset:]))
				offset += 4
			}
			return values, offset
		case typeFloat32:
			values := make([]float32, count)
			for index := range values {
				values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[offset:]))
				offset += 4
			}
			return values, offset
		}
	}
	panic(fmt.Sprintf("unsupported metadata type %d", kind))
}

func tensorSize(dims []uint64, kind uint32) uint64 {
	count := uint64(1)
	for _, dimension := range dims {
		count *= dimension
	}
	switch kind {
	case ggmlF32:
		return count * 4
	case ggmlF16, ggmlBF16:
		return count * 2
	default:
		return count
	}
}
