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
