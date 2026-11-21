// Package hardware discovers the local machine and recommends llama.cpp settings.
package hardware

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func readText(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func meminfoGB(key string) float64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	prefix := key + ":"
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kib, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			return 0
		}
		return kib / (1024 * 1024)
	}
	return 0
}

func cgroupMemoryLimitGB() float64 {
	for _, path := range []string{
		"/sys/fs/cgroup/memory.max",
		"/sys/fs/cgroup/memory/memory.limit_in_bytes",
	} {
		raw := readText(path)
		if raw == "" || raw == "max" {
			continue
		}
		value, err := strconv.ParseInt(raw, 10, 64)
		if err == nil && value > 0 && value < 1<<60 {
			return float64(value) / (1024 * 1024 * 1024)
		}
	}
	return 0
}

func cgroupMemoryUsageGB() float64 {
	for _, path := range []string{
		"/sys/fs/cgroup/memory.current",
		"/sys/fs/cgroup/memory/memory.usage_in_bytes",
	} {
		raw := readText(path)
		value, err := strconv.ParseInt(raw, 10, 64)
		if err == nil {
			return float64(value) / (1024 * 1024 * 1024)
		}
	}
	return 0
}

func cpuCountFromQuota(quota, period int64) int {
	if quota > 0 && period > 0 {
		return max(1, int(math.Round(float64(quota)/float64(period))))
	}
	return 0
}

func cgroupCPUCount() int {
	parts := strings.Fields(readText("/sys/fs/cgroup/cpu.max"))
	if len(parts) == 2 && parts[0] != "max" {
		quota, qerr := strconv.ParseInt(parts[0], 10, 64)
		period, perr := strconv.ParseInt(parts[1], 10, 64)
		if qerr == nil && perr == nil {
			if count := cpuCountFromQuota(quota, period); count > 0 {
				return count
			}
		}
	}
	quota, err := strconv.ParseInt(readText("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), 10, 64)
	if err != nil {
		return 0
	}
	periodText := readText("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
	if periodText == "" {
		periodText = "100000"
	}
	period, err := strconv.ParseInt(periodText, 10, 64)
	if err != nil {
		return 0
	}
	return cpuCountFromQuota(quota, period)
}

func totalRAMGB() float64 {
	var total float64
	switch runtime.GOOS {
	case "linux":
		total = meminfoGB("MemTotal")
	case "darwin":
		total = commandFloat("sysctl", "-n", "hw.memsize") / (1024 * 1024 * 1024)
	default:
		total = meminfoGB("MemTotal")
	}
	if limit := cgroupMemoryLimitGB(); limit > 0 && (total <= 0 || limit < total) {
		return limit
	}
	return total
}

func macOSAvailableRAMGB() float64 {
	total := commandFloat("sysctl", "-n", "hw.memsize") / (1024 * 1024 * 1024)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "vm_stat").Output()
	if err != nil {
		return total
	}
	pageSize := 4096.0
	lines := strings.Split(string(out), "\n")
	if len(lines) > 0 {
		if start := strings.Index(lines[0], "page size of "); start >= 0 {
			fields := strings.Fields(lines[0][start+len("page size of "):])
			if len(fields) > 0 {
				if value, err := strconv.ParseFloat(fields[0], 64); err == nil {
					pageSize = value
				}
			}
		}
	}
	var pages float64
	for _, line := range lines {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "Pages free", "Pages inactive", "Pages speculative":
			fields := strings.Fields(value)
			if len(fields) > 0 {
				if count, err := strconv.ParseFloat(fields[0], 64); err == nil {
					pages += count
				}
			}
		}
	}
	if pages > 0 {
		return math.Min(total, pages*pageSize/(1024*1024*1024))
	}
	return total
}

func availableRAMGB(total float64) float64 {
	available := meminfoGB("MemAvailable")
	if available == 0 {
		if runtime.GOOS == "darwin" {
			available = macOSAvailableRAMGB()
		} else {
			available = total
		}
	}
	if limit := cgroupMemoryLimitGB(); limit > 0 {
		remaining := math.Max(0, limit-cgroupMemoryUsageGB())
		if available > 0 {
			return math.Min(available, remaining)
		}
		return remaining
	}
	return available
}

func isLaptop() bool {
	if runtime.GOOS == "linux" {
		matches, _ := filepath.Glob("/sys/class/power_supply/BAT*")
		return len(matches) > 0
	}
	if runtime.GOOS == "darwin" {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "pmset", "-g", "batt").Output()
		return err == nil && strings.Contains(string(out), "InternalBattery")
	}
	return false
}

func cpuName() string {
	switch runtime.GOOS {
	case "linux":
		for _, line := range strings.Split(readText("/proc/cpuinfo"), "\n") {
			key, value, ok := strings.Cut(line, ":")
			if ok && strings.TrimSpace(key) == "model name" {
				return strings.TrimSpace(value)
			}
		}
	case "darwin":
		return commandString("sysctl", "-n", "machdep.cpu.brand_string")
	}
	return ""
}

func physicalCores() int {
	switch runtime.GOOS {
	case "linux":
		data, err := os.ReadFile("/proc/cpuinfo")
		if err == nil {
			pairs := map[string]bool{}
			processors := 0
			for _, block := range strings.Split(string(data), "\n\n") {
				physical, core := "", ""
				for _, line := range strings.Split(block, "\n") {
					key, value, ok := strings.Cut(line, ":")
					if !ok {
						continue
					}
					switch strings.TrimSpace(key) {
					case "processor":
						processors++
					case "physical id":
						physical = strings.TrimSpace(value)
					case "core id":
						core = strings.TrimSpace(value)
					}
				}
				if core != "" {
					pairs[physical+"/"+core] = true
				}
			}
			if len(pairs) > 0 {
				return len(pairs)
			}
			if processors > 0 {
				return processors
			}
		}
	case "darwin":
		if cores := int(commandFloat("sysctl", "-n", "hw.physicalcpu")); cores > 0 {
			return cores
		}
	}
	return runtime.NumCPU()
}

func commandString(name string, args ...string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func nvidiaGPU() (string, float64) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(
		ctx,
		"nvidia-smi",
		"--query-gpu=name,memory.total",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return "", 0
	}
	first := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)[0]
	index := strings.LastIndex(first, ",")
	if index < 0 {
		return "", 0
	}
	name := strings.TrimSpace(first[:index])
	memoryMB, err := strconv.ParseFloat(strings.TrimSpace(first[index+1:]), 64)
	if err != nil {
		return "", 0
	}
	return name, memoryMB / 1024
}

func appleGPU(ramGB float64) (string, float64, bool) {
	if runtime.GOOS != "darwin" || (runtime.GOARCH != "arm64" && runtime.GOARCH != "aarch64") {
		return "", 0, false
	}
	return "Apple Silicon (Metal)", roundTo(ramGB*0.70, 2), true
}

func commandFloat(name string, args ...string) float64 {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return 0
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		return 0
	}
	return value
}

// Detect probes CPU, memory, disk, GPU and container limits concurrently.
func Detect() model.Specs {
	var (
		wg               sync.WaitGroup
		total, available float64
		disk             float64
		laptop           bool
		gpuName          string
		gpuVRAM          float64
		cpu              string
		physical         int
	)
	total = totalRAMGB()
	wg.Add(6)
	go func() { defer wg.Done(); cpu = cpuName() }()
	go func() { defer wg.Done(); physical = physicalCores() }()
	go func() { defer wg.Done(); disk = diskFreeGB() }()
	go func() { defer wg.Done(); laptop = isLaptop() }()
	go func() { defer wg.Done(); gpuName, gpuVRAM = nvidiaGPU() }()
	go func() { defer wg.Done(); available = availableRAMGB(total) }()
	wg.Wait()

	unified := false
	if gpuName == "" {
		gpuName, gpuVRAM, unified = appleGPU(total)
	}
	threads := max(1, runtime.NumCPU())
	if cpus := cgroupCPUCount(); cpus > 0 {
		threads = min(threads, cpus)
	}
	if physical <= 0 || physical > threads {
		physical = threads
	}
	return model.Specs{
		CPUName:        cpu,
		CPUThreads:     threads,
		PhysicalCores:  physical,
		RAMGB:          roundTo(total, 2),
		GPUName:        gpuName,
		GPUVramGB:      roundTo(gpuVRAM, 2),
		AvailableRAMGB: roundTo(available, 2),
		DiskFreeGB:     roundTo(disk, 2),
		UnifiedMemory:  unified,
		IsLaptop:       laptop,
	}
}

// UsableModelMemoryGB returns a conservative weights budget.
func UsableModelMemoryGB(specs model.Specs) float64 {
	available := specs.AvailableRAMGB
	if available == 0 {
		available = specs.RAMGB
	}
	floor := 1.5
	if available < 2.5 {
		floor = 0.5
	}
	reserve := math.Min(math.Max(floor, available*0.25), available*0.6)
	ramBudget := math.Max(0, available-reserve) * 0.80
	vramBudget := 0.0
	if !specs.UnifiedMemory {
		vramBudget = specs.GPUVramGB * 0.60
	}
	return roundTo(ramBudget+vramBudget, 2)
}

// ModelFit classifies a model size as comfortable, tight, too large or unknown.
func ModelFit(specs model.Specs, modelSizeGB float64) string {
	if modelSizeGB < 0.001 {
		return "unknown"
	}
	budget := UsableModelMemoryGB(specs)
	if budget <= 0 {
		return "unknown"
	}
	if modelSizeGB <= budget*0.75 {
		return "comfortable"
	}
	if modelSizeGB <= budget {
		return "tight"
	}
	return "too_large"
}

// RecommendedContextSize picks a safe context window for the machine.
func RecommendedContextSize(specs model.Specs, modelSizeGB float64) int {
	available := specs.AvailableRAMGB
	if available == 0 {
		available = specs.RAMGB
	}
	contextSize := 1024
	switch {
	case available >= 24:
		contextSize = 8192
	case available >= 12:
		contextSize = 4096
	case available >= 8:
		contextSize = 2048
	}
	budget := UsableModelMemoryGB(specs)
	if modelSizeGB > budget {
		if available < 6 {
			return 512
		}
		return 1024
	}
	if modelSizeGB >= 0.01 && budget-modelSizeGB >= 4.0 && contextSize < 8192 {
		contextSize *= 2
	}
	return contextSize
}

// RecommendedThreads uses every available logical CPU, capped for llama.cpp.
func RecommendedThreads(specs model.Specs) int {
	threads := specs.CPUThreads
	if threads < 1 {
		return 1
	}
	if threads > 32 {
		return 32
	}
	return threads
}

// RecommendSettings builds the runtime config for a model and machine.
func RecommendSettings(specs model.Specs, modelPath string, modelSizeBytes int64) model.Config {
	modelSizeGB := 0.0
	if modelSizeBytes > 0 {
		modelSizeGB = float64(modelSizeBytes) / (1024 * 1024 * 1024)
	} else if modelPath != "" {
		if info, err := os.Stat(modelPath); err == nil && info.Mode().IsRegular() {
			modelSizeGB = float64(info.Size()) / (1024 * 1024 * 1024)
		}
	}
	contextSize := RecommendedContextSize(specs, modelSizeGB)
	budget := UsableModelMemoryGB(specs)
	fit := ModelFit(specs, modelSizeGB)
	gpuLayers := 0
	if specs.GPUName != "" && modelSizeGB > 0 {
		if specs.UnifiedMemory {
			if modelSizeGB <= budget {
				gpuLayers = 99
			}
		} else if modelSizeGB <= specs.GPUVramGB*0.60 {
			gpuLayers = 99
		}
	}
	available := specs.AvailableRAMGB
	if available == 0 {
		available = specs.RAMGB
	}
	return model.Config{
		CPUThreads:          RecommendedThreads(specs),
		ContextSize:         contextSize,
		NGPULayers:          gpuLayers,
		Mmap:                true,
		LowMemory:           available < 6.0 || fit == "too_large",
		ModelFit:            fit,
		ModelSizeGB:         roundTo(modelSizeGB, 3),
		UsableModelMemoryGB: budget,
		Hardware:            specs,
	}
}

func roundTo(value float64, places int) float64 {
	factor := math.Pow(10, float64(places))
	return math.Round(value*factor) / factor
}
