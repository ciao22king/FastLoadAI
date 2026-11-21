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
}
