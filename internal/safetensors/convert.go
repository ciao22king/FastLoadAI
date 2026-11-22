package safetensors

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Result describes a converted GGUF file.
type Result struct {
	Path       string
	Bytes      int64
	Tensors    int
	Parameters int64
	Arch       string
	FileType   uint32
}

// Convert turns a Hugging Face safetensors directory into one GGUF file.
func Convert(dir, output string) (Result, error) {
	model, err := Load(dir)
	if err != nil {
		return Result{}, err
	}
	defer model.Close()
	arch, err := ggufArchitecture(model.Config)
	if err != nil {
		return Result{}, err
	}
	tensors, parameters, err := outputTensors(model)
	if err != nil {
		return Result{}, err
	}
	metadata := modelMetadata(model, arch, tensors)
	if err := writeGGUF(output, metadata, tensors); err != nil {
		return Result{}, err
	}
	info, err := os.Stat(output)
	if err != nil {
		return Result{}, err
	}
	fileType := uint32(1)
	for _, tensor := range tensors {
		if tensor.kind == ggmlBF16 {
			fileType = 32
			break
		}
		if tensor.kind == ggmlF32 {
			fileType = 0
		}
	}
	return Result{
		Path: output, Bytes: info.Size(), Tensors: len(tensors),
		Parameters: parameters, Arch: arch, FileType: fileType,
	}, nil
}

// EnsureConverted converts dir into modelsDir once and returns the GGUF path.
func EnsureConverted(dir, modelsDir string) (string, error) {
	model, err := Load(dir)
	if err != nil {
		return "", err
	}
	dtype := nativeDtype(model.Config, model.Tensors)
	model.Close()
	if modelsDir == "" {
		modelsDir = "models"
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return "", err
	}
	name := filepath.Base(filepath.Clean(dir))
	if name == "." || name == string(filepath.Separator) {
		name = "model"
	}
	output := filepath.Join(modelsDir, name+"-"+dtype+".gguf")
	if info, err := os.Stat(output); err == nil && info.Mode().IsRegular() && info.Size() > 4 {
		header := make([]byte, 4)
		if file, err := os.Open(output); err == nil {
			_, readErr := io.ReadFull(file, header)
			file.Close()
			if readErr == nil && string(header) == ggufMagic {
				return output, nil
			}
		}
	}
	temporary := output + ".tmp"
	if _, err := Convert(dir, temporary); err != nil {
		os.Remove(temporary)
		return "", err
	}
	if err := os.Rename(temporary, output); err != nil {
		os.Remove(temporary)
		return "", err
	}
	return output, nil
}

func ggufArchitecture(config Config) (string, error) {
	switch strings.ToLower(config.ModelType) {
	case "qwen2":
		return "qwen2", nil
	case "llama":
		return "llama", nil
	default:
		return "", fmt.Errorf("unsupported architecture %q: this build converts Qwen2 and Llama safetensors", config.ModelType)
	}
}

func outputTensors(model *Model) ([]outputTensor, int64, error) {
	var result []outputTensor
	var parameters int64
	seen := map[string]bool{}
	for name, source := range model.Tensors {
		mapped, ok := mapTensorName(name)
		if !ok {
			if strings.Contains(name, "rotary_emb") || strings.Contains(name, "inv_freq") {
				continue
			}
			return nil, 0, fmt.Errorf("unsupported tensor %q", name)
		}
		if seen[mapped] {
			return nil, 0, fmt.Errorf("duplicate mapped tensor %q", mapped)
		}
		seen[mapped] = true
		kind, size, err := outputTensorShape(source)
		if err != nil {
			return nil, 0, err
		}
		result = append(result, outputTensor{
			name: mapped, source: source, dims: shapeToDims(source.Shape), kind: kind, size: size,
		})
		count := int64(1)
		for _, dimension := range source.Shape {
			count *= dimension
		}
		parameters += count
	}
	if !seen["token_embd.weight"] {
		return nil, 0, fmt.Errorf("missing token_embd.weight (model.embed_tokens.weight)")
	}
	if !seen["output.weight"] && !model.Config.TieWordEmbeddings {
		return nil, 0, fmt.Errorf("missing lm_head.weight and tie_word_embeddings is false")
	}
	sort.Slice(result, func(i, j int) bool { return result[i].name < result[j].name })
	return result, parameters, nil
}

func outputTensorShape(source *Tensor) (uint32, int64, error) {
	elements := int64(1)
	for _, dimension := range source.Shape {
		elements *= dimension
	}
	if elements < 0 {
		return 0, 0, fmt.Errorf("tensor %s has an invalid shape", source.Name)
	}
	if len(source.Shape) <= 1 {
		return ggmlF32, elements * 4, nil
	}
	switch source.Dtype {
	case "F32":
		return ggmlF32, elements * 4, nil
	case "F16":
		return ggmlF16, elements * 2, nil
	case "BF16":
		return ggmlBF16, elements * 2, nil
	default:
		return 0, 0, fmt.Errorf("unsupported 2D dtype %s in %s", source.Dtype, source.Name)
	}
}

func shapeToDims(shape []int64) []uint64 {
	dims := make([]uint64, len(shape))
	for index, dimension := range shape {
		dims[len(shape)-1-index] = uint64(dimension)
	}
	return dims
}

func mapTensorName(name string) (string, bool) {
	switch name {
	case "model.embed_tokens.weight":
		return "token_embd.weight", true
	case "model.norm.weight":
		return "output_norm.weight", true
	case "lm_head.weight":
		return "output.weight", true
	}
	if !strings.HasPrefix(name, "model.layers.") {
		return "", false
	}
	rest := strings.TrimPrefix(name, "model.layers.")
	dot := strings.IndexByte(rest, '.')
	if dot < 0 {
		return "", false
	}
	layer, err := strconv.Atoi(rest[:dot])
	if err != nil || layer < 0 {
		return "", false
	}
	suffix := rest[dot+1:]
	mapped, ok := mapLayerSuffix(suffix)
	if !ok {
		return "", false
	}
	return fmt.Sprintf("blk.%d.%s", layer, mapped), true
}

func mapLayerSuffix(suffix string) (string, bool) {
	switch suffix {
	case "input_layernorm.weight":
		return "attn_norm.weight", true
	case "post_attention_layernorm.weight":
		return "ffn_norm.weight", true
	case "self_attn.q_proj.weight":
		return "attn_q.weight", true
	case "self_attn.q_proj.bias":
		return "attn_q.bias", true
	case "self_attn.k_proj.weight":
		return "attn_k.weight", true
	case "self_attn.k_proj.bias":
		return "attn_k.bias", true
	case "self_attn.v_proj.weight":
		return "attn_v.weight", true
	case "self_attn.v_proj.bias":
		return "attn_v.bias", true
	case "self_attn.o_proj.weight":
		return "attn_output.weight", true
	case "mlp.gate_proj.weight":
		return "ffn_gate.weight", true
	case "mlp.up_proj.weight":
		return "ffn_up.weight", true
	case "mlp.down_proj.weight":
		return "ffn_down.weight", true
	default:
		return "", false
	}
}

func modelMetadata(model *Model, arch string, tensors []outputTensor) []metadataValue {
	config := model.Config
	name := config.NameOrPath
	if name == "" {
		name = filepath.Base(filepath.Clean(model.Dir))
	}
	fileType := uint32(1)
	for _, tensor := range tensors {
		if tensor.kind == ggmlBF16 {
			fileType = 32
			break
		}
		if tensor.kind == ggmlF32 {
			fileType = 0
		}
	}
	headDim := 0
	if config.NumAttentionHeads > 0 {
		headDim = config.HiddenSize / config.NumAttentionHeads
	}
	metadata := []metadataValue{
		{"general.architecture", typeString, arch},
		{"general.name", typeString, name},
		{"general.file_type", typeUint32, fileType},
		{"general.alignment", typeUint32, uint32(ggufAlign)},
		{arch + ".context_length", typeUint32, uint32(config.MaxPositionEmbeddings)},
		{arch + ".embedding_length", typeUint32, uint32(config.HiddenSize)},
		{arch + ".block_count", typeUint32, uint32(config.NumHiddenLayers)},
		{arch + ".feed_forward_length", typeUint32, uint32(config.IntermediateSize)},
		{arch + ".attention.head_count", typeUint32, uint32(config.NumAttentionHeads)},
		{arch + ".attention.head_count_kv", typeUint32, uint32(config.NumKeyValueHeads)},
		{arch + ".attention.layer_norm_rms_epsilon", typeFloat32, float32(config.RMSNormEPS)},
		{arch + ".rope.freq_base", typeFloat32, float32(config.RopeTheta)},
		{arch + ".vocab_size", typeUint32, uint32(len(model.Tokenizer.Tokens))},
	}
	if headDim > 0 {
		metadata = append(metadata,
			metadataValue{arch + ".attention.key_length", typeUint32, uint32(headDim)},
			metadataValue{arch + ".attention.value_length", typeUint32, uint32(headDim)},
			metadataValue{arch + ".rope.dimension_count", typeUint32, uint32(headDim)},
		)
	}
	tokenizer := model.Tokenizer
	metadata = append(metadata,
		metadataValue{"tokenizer.ggml.model", typeString, "gpt2"},
		metadataValue{"tokenizer.ggml.tokens", typeArray, tokenizer.Tokens},
		metadataValue{"tokenizer.ggml.scores", typeArray, tokenizer.Scores},
		metadataValue{"tokenizer.ggml.token_type", typeArray, tokenizer.Types},
		metadataValue{"tokenizer.ggml.merges", typeArray, tokenizer.Merges},
		metadataValue{"tokenizer.ggml.bos_token_id", typeUint32, tokenizer.BosID},
		metadataValue{"tokenizer.ggml.eos_token_id", typeUint32, tokenizer.EosID},
		metadataValue{"tokenizer.ggml.padding_token_id", typeUint32, tokenizer.PadID},
		metadataValue{"tokenizer.ggml.add_bos_token", typeBool, tokenizer.AddBOS},
		metadataValue{"tokenizer.ggml.add_eos_token", typeBool, tokenizer.AddEOS},
		metadataValue{"tokenizer.ggml.add_space_prefix", typeBool, false},
	)
	if tokenizer.Pre != "" {
		metadata = append(metadata, metadataValue{"tokenizer.ggml.pre", typeString, tokenizer.Pre})
	}
	// llama.cpp ships the Qwen2 template with Qwen's default system prompt;
	// embedding the raw tokenizer template would drop that default.
	if tokenizer.ChatTemplate != "" && arch != "qwen2" {
		metadata = append(metadata, metadataValue{"tokenizer.ggml.chat_template", typeString, tokenizer.ChatTemplate})
	}
	return metadata
}

// NativeDtype reports the storage type of a safetensors checkpoint.
func NativeDtype(dir string) (string, error) {
	model, err := Load(dir)
	if err != nil {
		return "", err
	}
	defer model.Close()
	return nativeDtype(model.Config, model.Tensors), nil
}

func nativeDtype(config Config, tensors map[string]*Tensor) string {
	for _, tensor := range tensors {
		switch tensor.Dtype {
		case "BF16":
			return "BF16"
		case "F16":
			return "F16"
		case "F32":
			return "F32"
		}
	}
	switch strings.ToLower(config.TorchDtype) {
	case "bfloat16":
		return "BF16"
	case "float16":
		return "F16"
	case "float32":
		return "F32"
	default:
		return "F16"
	}
}
