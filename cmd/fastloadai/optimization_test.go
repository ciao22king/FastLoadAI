package main

import (
	"flag"
	"testing"
)

func TestOptimizationFlagsValidate(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	if err := set.Parse([]string{
		"--profile", "speed", "--flash-attn", "on", "--kv-cache", "q8_0",
		"--load-mode", "mmap+mlock", "--split-mode", "row", "--numa", "distribute",
		"--priority", "1", "--poll", "20", "--moe-cache-cpu-share", "0.5",
	}); err != nil {
		t.Fatal(err)
	}
	if err := optimize.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
}

func TestOptimizationFlagsRejectBadValues(t *testing.T) {
	tests := [][]string{
		{"--profile", "turbo"},
		{"--flash-attn", "sometimes"},
		{"--kv-cache", "q2_k"},
		{"--priority", "9"},
		{"--poll", "101"},
		{"--moe-cache-cpu-share", "2"},
		{"--mlock", "--no-mlock"},
		{"--kv-cache-k", "q2_k"},
		{"--kv-cache-v", "bad"},
		{"--cache-ram", "-2"},
		{"--swa-checkpoints", "-2"},
		{"--checkpoint-min-step", "-2"},
		{"--swa-checkpoints", "-1"},
		{"--checkpoint-min-step", "-1"},
		{"--rope-scaling", "dynamic"},
		{"--rope-scale", "-1"},
		{"--yarn-orig-ctx", "-1"},
		{"--cpu-mask", "xyz"},
		{"--cpu-mask", "0"},
		{"--cpu-range", "7-2"},
		{"--cpu-range", "3"},
		{"--cpu-mask", "1", "--cpu-range", "0-1"},
		{"--kv-unified", "--no-kv-unified"},
		{"--no-repack", "--repack"},
		{"--no-warmup", "--warmup"},
	}
	for _, args := range tests {
		set := flag.NewFlagSet("test", flag.ContinueOnError)
		optimize := registerOptimizationFlags(set)
		if err := set.Parse(args); err != nil {
			t.Fatal(err)
		}
		if err := optimize.validate(); err == nil {
			t.Fatalf("args %v should fail validation", args)
		}
	}
}

func TestOptimizationOptions(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	if err := set.Parse([]string{
		"--profile", "memory", "--no-mmap", "--mlock", "--n-cpu-moe", "4",
		"--speculative", "--cache-reuse", "0", "--priority", "2", "--poll", "0",
	}); err != nil {
		t.Fatal(err)
	}
	opts := optimize.options()
	if opts.Profile != "memory" || !opts.NoMmap || opts.Mlock == nil || !*opts.Mlock {
		t.Fatalf("unexpected options: %+v", opts)
	}
	if opts.NCPUMoE != 4 || !opts.Speculative || opts.CacheReuse != 0 {
		t.Fatalf("unexpected options: %+v", opts)
	}
	if opts.Priority != 2 || opts.Poll != 0 {
		t.Fatalf("unexpected scheduling options: %+v", opts)
	}
	if !optimize.touched() {
		t.Fatal("optimization flags should count as touched")
	}
}

func TestOptimizationTouchedIgnoresOtherFlags(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	set.Bool("no-color", false, "")
	if err := set.Parse([]string{"--no-color"}); err != nil {
		t.Fatal(err)
	}
	if optimize.touched() {
		t.Fatal("non-optimization flags should not count as touched")
	}
}

func TestOptimizationNewFlags(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	if err := set.Parse([]string{
		"--kv-cache-k", "q8_0", "--kv-cache-v", "q4_0", "--kv-unified",
		"--cache-ram", "0", "--slot-save-path", "/tmp/slots",
		"--swa-checkpoints", "0", "--checkpoint-min-step", "0",
		"--rope-scaling", "yarn", "--rope-scale", "2", "--yarn-orig-ctx", "4096",
		"--cpu-range", "0-7", "--no-repack", "--no-warmup",
	}); err != nil {
		t.Fatal(err)
	}
	if err := optimize.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	opts := optimize.options()
	if opts.KVCacheTypeK != "q8_0" || opts.KVCacheTypeV != "q4_0" {
		t.Fatalf("split KV options = %q/%q", opts.KVCacheTypeK, opts.KVCacheTypeV)
	}
	if opts.KVUnified == nil || !*opts.KVUnified {
		t.Fatalf("KV unified = %v", opts.KVUnified)
	}
	if opts.CacheRAMMiB == nil || *opts.CacheRAMMiB != 0 {
		t.Fatalf("cache RAM = %v", opts.CacheRAMMiB)
	}
	if opts.SlotSavePath != "/tmp/slots" || opts.SWACheckpoints != 0 || opts.CheckpointMinStep != 0 {
		t.Fatalf("unexpected cache options: %+v", opts)
	}
	if opts.RopeScaling != "yarn" || opts.RopeScale != 2 || opts.YarnOrigCtx != 4096 {
		t.Fatalf("unexpected RoPE options: %+v", opts)
	}
	if opts.CPURange != "0-7" || opts.NoRepack == nil || !*opts.NoRepack || opts.NoWarmup == nil || !*opts.NoWarmup {
		t.Fatalf("unexpected affinity/startup options: %+v", opts)
	}
}

func TestOptimizationPositiveToggles(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	if err := set.Parse([]string{"--no-kv-unified", "--repack", "--warmup", "--cache-ram", "-1"}); err != nil {
		t.Fatal(err)
	}
	if err := optimize.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	opts := optimize.options()
	if opts.KVUnified == nil || *opts.KVUnified {
		t.Fatalf("KV unified = %v, want false", opts.KVUnified)
	}
	if opts.NoRepack == nil || *opts.NoRepack || opts.NoWarmup == nil || *opts.NoWarmup {
		t.Fatalf("positive toggles failed: %+v", opts)
	}
	if opts.CacheRAMMiB == nil || *opts.CacheRAMMiB != -1 {
		t.Fatalf("cache RAM = %v, want -1", opts.CacheRAMMiB)
	}
}

func TestOptimizationAdaptiveFlags(t *testing.T) {
	set := flag.NewFlagSet("test", flag.ContinueOnError)
	optimize := registerOptimizationFlags(set)
	if err := set.Parse([]string{"--effort", "deep", "--adaptive=false"}); err != nil {
		t.Fatal(err)
	}
	if err := optimize.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	opts := optimize.options()
	if opts.Effort != "deep" || opts.Adaptive {
		t.Fatalf("unexpected adaptive options: %+v", opts)
	}
	if !optimize.touched() {
		t.Fatal("adaptive flags should count as touched")
	}

	bad := flag.NewFlagSet("test", flag.ContinueOnError)
	badOptimize := registerOptimizationFlags(bad)
	if err := bad.Parse([]string{"--effort", "maximum"}); err != nil {
		t.Fatal(err)
	}
	if err := badOptimize.validate(); err == nil {
		t.Fatal("unknown effort should fail validation")
	}
}
