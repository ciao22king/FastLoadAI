//go:build !windows

package hardware

import "syscall"

func diskFreeGB() float64 {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(".", &stat); err != nil {
		return 0
	}
	return float64(stat.Bavail) * float64(stat.Bsize) / (1024 * 1024 * 1024)
}
