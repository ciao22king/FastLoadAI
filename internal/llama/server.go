// Package llama starts and stops llama.cpp's OpenAI-compatible server.
package llama

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ciao22king/FastLoadAI/internal/model"
)

// Server owns one llama-server process and its temporary log.
type Server struct {
	Cmd     *exec.Cmd
	BaseURL string
	LogPath string

	logFile *os.File
	waitCh  chan error
	mu      sync.Mutex
	waitErr error
}

// AvailablePort asks the kernel for a free loopback port.
func AvailablePort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

// BuildCommand returns the llama-server arguments for a config.
func BuildCommand(executable, modelPath string, config model.Config, port int) []string {
	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--threads", strconv.Itoa(config.CPUThreads),
		"--ctx-size", strconv.Itoa(config.ContextSize),
		"--n-gpu-layers", strconv.Itoa(config.NGPULayers),
	}
	if !config.Mmap {
		args = append(args, "--no-mmap")
	}
	if config.LowMemory {
		args = append(args, "--batch-size", "128", "--ubatch-size", "128")
	}
	return args
}

// Start launches llama-server and waits for the process to boot.
func Start(executable, modelPath string, config model.Config) (*Server, error) {
	port, err := AvailablePort()
	if err != nil {
		return nil, err
	}
	logFile, err := os.CreateTemp("", "fastloadai-llama-*.log")
	if err != nil {
		return nil, err
	}
	command := exec.Command(executable, BuildCommand(executable, modelPath, config, port)...)
	command.Stdout = logFile
	command.Stderr = logFile
	if err := command.Start(); err != nil {
		logFile.Close()
		os.Remove(logFile.Name())
		return nil, err
	}
	server := &Server{
		Cmd:     command,
		BaseURL: fmt.Sprintf("http://127.0.0.1:%d", port),
		LogPath: logFile.Name(),
		logFile: logFile,
		waitCh:  make(chan error, 1),
	}
	go func() {
		err := command.Wait()
		server.mu.Lock()
		server.waitErr = err
		server.mu.Unlock()
		server.waitCh <- err
		close(server.waitCh)
	}()
	return server, nil
}

// WaitReady polls /health until the model is loaded or the process exits.
func (s *Server) WaitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: time.Second}
	for time.Now().Before(deadline) {
		select {
		case err := <-s.waitCh:
			if err == nil {
				return fmt.Errorf("llama-server exited while loading the model")
			}
			return fmt.Errorf("llama-server exited while loading the model: %w", err)
		default:
		}
		request, _ := http.NewRequestWithContext(ctx, http.MethodGet, s.BaseURL+"/health", nil)
		response, err := client.Do(request)
		if err == nil {
			response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
	return fmt.Errorf("timed out waiting for llama-server to become ready")
}

// Stop terminates the server, escalating to Kill after five seconds.
func (s *Server) Stop() {
	if s == nil || s.Cmd == nil || s.Cmd.Process == nil {
		return
	}
	select {
	case <-s.waitCh:
		s.cleanup()
		return
	default:
	}
	_ = s.Cmd.Process.Signal(os.Interrupt)
	select {
	case <-s.waitCh:
	case <-time.After(5 * time.Second):
		_ = s.Cmd.Process.Kill()
		<-s.waitCh
	}
	s.cleanup()
}

func (s *Server) cleanup() {
	if s.logFile != nil {
		_ = s.logFile.Close()
		s.logFile = nil
	}
}

// LogTail returns the last few kilobytes written by llama-server.
func (s *Server) LogTail() string {
	data, err := os.ReadFile(s.LogPath)
	if err != nil {
		return ""
	}
	if len(data) > 3000 {
		data = data[len(data)-3000:]
	}
	return strings.TrimSpace(string(data))
}

// IsMemoryError reports the usual llama.cpp allocation failures.
func IsMemoryError(details string) bool {
	lowered := strings.ToLower(details)
	for _, marker := range []string{
		"failed to allocate",
		"out of memory",
		"not enough memory",
		"insufficient memory",
		"bad_alloc",
	} {
		if strings.Contains(lowered, marker) {
			return true
		}
	}
	return false
}
