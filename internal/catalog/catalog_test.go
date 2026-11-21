package catalog

import (
	"testing"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

func TestMatch(t *testing.T) {
	matched, ok := Match("qwen2.5-0.5b")
	if !ok || matched.Repo != "Qwen/Qwen2.5-0.5B-Instruct-GGUF" {
		t.Fatalf("match = %+v ok=%v", matched, ok)
	}
	if _, ok := Match("does-not-exist"); ok {
		t.Fatal("unexpected match")
	}
}

func TestRowsSortedAndRecommended(t *testing.T) {
	specs := model.Specs{CPUThreads: 8, RAMGB: 16, AvailableRAMGB: 8}
	rows := Rows(specs)
	if len(rows) != len(Catalog) {
		t.Fatalf("got %d rows, want %d", len(rows), len(Catalog))
	}
	recommended := 0
	for index, row := range rows {
		if row.Recommended {
			recommended++
		}
		if index > 0 && rows[index-1].SizeGB > row.SizeGB {
			t.Fatalf("rows are not sorted at %d", index)
		}
	}
	if recommended != 1 {
		t.Fatalf("recommended = %d, want 1", recommended)
	}
}
