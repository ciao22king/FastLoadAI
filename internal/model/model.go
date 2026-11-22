// Package model contains the hardware and runtime data shared by FastLoadAI.
package model

// Message is one chat turn sent to llama-server.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Specs is the detected hardware profile.
type Specs struct {
	CPUName        string  `json:"cpu_name,omitempty"`
	CPUThreads     int     `json:"cpu_threads"`
	PhysicalCores  int     `json:"physical_cores,omitempty"`
	RAMGB          float64 `json:"ram_gb"`
	GPUName        string  `json:"gpu_name,omitempty"`
	GPUVramGB      float64 `json:"gpu_vram_gb"`
	AvailableRAMGB float64 `json:"available_ram_gb"`
	DiskFreeGB     float64 `json:"disk_free_gb"`
	UnifiedMemory  bool    `json:"unified_memory"`
	IsLaptop       bool    `json:"is_laptop"`

	// Hybrid CPU topology (Intel P-core/E-core, Apple performance/efficiency).
	Hybrid             bool   `json:"hybrid,omitempty"`
	PerformanceCPUs    string `json:"performance_cpus,omitempty"`
	EfficiencyCPUs     string `json:"efficiency_cpus,omitempty"`
	PerformanceMask    string `json:"performance_mask,omitempty"`
	PerformanceThreads int    `json:"performance_threads,omitempty"`
	EfficiencyThreads  int    `json:"efficiency_threads,omitempty"`
}

// Config is the persisted runtime plan for llama.cpp.
type Config struct {
	CPUThreads          int     `json:"cpu_threads"`
	ContextSize         int     `json:"context_size"`
	NGPULayers          int     `json:"n_gpu_layers"`
	Mmap                bool    `json:"mmap"`
	LowMemory           bool    `json:"low_memory"`
	ModelFit            string  `json:"model_fit"`
	ModelSizeGB         float64 `json:"model_size_gb"`
	UsableModelMemoryGB float64 `json:"usable_model_memory_gb"`
	Hardware            Specs   `json:"hardware"`

	ModelPath        string `json:"model_path,omitempty"`
	ModelRepo        string `json:"model_repo,omitempty"`
	Revision         string `json:"revision,omitempty"`
	Quantization     string `json:"quantization,omitempty"`
	QuantizationFile string `json:"quantization_file,omitempty"`
	ModelSizeBytes   int64  `json:"model_size_bytes,omitempty"`
	LlamaCPPMoe      string `json:"llama_cpp_moe,omitempty"`

	// Optimization settings. Zero values mean "let FastLoadAI choose".
	OptimizationProfile string  `json:"optimization_profile,omitempty"`
	MoE                 bool    `json:"moe,omitempty"`
	FlashAttention      string  `json:"flash_attention,omitempty"`
	KVCacheTypeK        string  `json:"kv_cache_type_k,omitempty"`
	KVCacheTypeV        string  `json:"kv_cache_type_v,omitempty"`
	KVUnified           *bool   `json:"kv_unified,omitempty"`
	CacheRAMMiB         *int    `json:"cache_ram_mib,omitempty"`
	SlotSavePath        string  `json:"slot_save_path,omitempty"`
	LoadMode            string  `json:"load_mode,omitempty"`
	LazyMode            string  `json:"lazy_mode,omitempty"`
	NoKVOffload         bool    `json:"no_kv_offload,omitempty"`
	NoOpOffload         bool    `json:"no_op_offload,omitempty"`
	NoContBatching      bool    `json:"no_cont_batching,omitempty"`
	Parallel            int     `json:"parallel,omitempty"`
	BatchSize           int     `json:"batch_size,omitempty"`
	UBatchSize          int     `json:"ubatch_size,omitempty"`
	ThreadsBatch        int     `json:"threads_batch,omitempty"`
	NumA                string  `json:"numa,omitempty"`
	SplitMode           string  `json:"split_mode,omitempty"`
	TensorSplit         string  `json:"tensor_split,omitempty"`
	CPUMoE              bool    `json:"cpu_moe,omitempty"`
	NCPUMoE             int     `json:"n_cpu_moe,omitempty"`
	MoECacheMiB         int     `json:"moe_cache_mib,omitempty"`
	MoECacheCPUShare    float64 `json:"moe_cache_cpu_share,omitempty"`
	OverrideTensor      string  `json:"override_tensor,omitempty"`
	SpecType            string  `json:"spec_type,omitempty"`
	DraftModel          string  `json:"draft_model,omitempty"`
	DraftMax            int     `json:"draft_max,omitempty"`
	DraftMin            int     `json:"draft_min,omitempty"`
	DraftPMin           float64 `json:"draft_p_min,omitempty"`
	DraftGPULayers      int     `json:"draft_gpu_layers,omitempty"`
	CacheReuse          int     `json:"cache_reuse,omitempty"`
	ContextShift        bool    `json:"context_shift,omitempty"`
	SWAFull             bool    `json:"swa_full,omitempty"`
	SWACheckpoints      *int    `json:"swa_checkpoints,omitempty"`
	CheckpointMinStep   *int    `json:"checkpoint_min_step,omitempty"`
	RopeScaling         string  `json:"rope_scaling,omitempty"`
	RopeScale           float64 `json:"rope_scale,omitempty"`
	YarnOrigCtx         int     `json:"yarn_orig_ctx,omitempty"`
	CPUMask             string  `json:"cpu_mask,omitempty"`
	CPURange            string  `json:"cpu_range,omitempty"`
	NoRepack            bool    `json:"no_repack,omitempty"`
	NoWarmup            bool    `json:"no_warmup,omitempty"`
	Priority            int     `json:"priority,omitempty"`
	Poll                int     `json:"poll,omitempty"`

	// Adaptive parameter budgeting. The planner reads GGUF metadata, measures
	// free RAM/VRAM and chooses a residency plan that should not exhaust memory.
	Adaptive           bool   `json:"adaptive,omitempty"`
	Effort             string `json:"effort,omitempty"`
	ModelParameters    int64  `json:"model_parameters,omitempty"`
	ModelLayers        int    `json:"model_layers,omitempty"`
	ParameterBudget    int64  `json:"parameter_budget,omitempty"`
	ActiveParameters   int64  `json:"active_parameters,omitempty"`
	ResidentParameters int64  `json:"resident_parameters,omitempty"`
	KVBytesPerToken    int64  `json:"kv_bytes_per_token,omitempty"`
	Streaming          bool   `json:"streaming,omitempty"`
	AdaptiveRisk       string `json:"adaptive_risk,omitempty"`
	AdaptiveReason     string `json:"adaptive_reason,omitempty"`
}
