// Package hf talks to the Hugging Face Hub and downloads GGUF files.
package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// DefaultBaseURL is the public Hugging Face Hub.
const DefaultBaseURL = "https://huggingface.co"

// Sibling is one file in a Hugging Face repository.
type Sibling struct {
	Filename string `json:"rfilename"`
	Size     *int64 `json:"size"`
}

// ModelInfo is the subset of the Hub model API used by FastLoadAI.
type ModelInfo struct {
	Siblings []Sibling `json:"siblings"`
}

// Group is a single GGUF file or all shards of one GGUF model.
type Group struct {
	Name      string
	Files     []string
	FileSizes map[string]int64
	Size      int64
	SizeKnown bool
}

// Client is a small Hugging Face API and download client.
type Client struct {
	BaseURL string
	Token   string
	HTTP    *http.Client
}

// NewClient creates a client with a token and a sane timeout.
func NewClient(token string) *Client {
	return &Client{
		BaseURL: DefaultBaseURL,
		Token:   token,
		HTTP:    &http.Client{Timeout: 0},
	}
}

// HTTPError is returned for non-2xx Hub responses.
type HTTPError struct {
	Status int
	Body   string
}

func (e *HTTPError) Error() string {
	body := strings.TrimSpace(e.Body)
	if body == "" {
		return fmt.Sprintf("HTTP %d", e.Status)
	}
	return fmt.Sprintf("HTTP %d: %s", e.Status, body)
}

func escapeSegments(value string) string {
	parts := strings.Split(value, "/")
	for index, part := range parts {
		parts[index] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

func (c *Client) request(ctx context.Context, method, target string, body io.Reader) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return nil, err
	}
	request.Header.Set("User-Agent", "fastloadai/0.5")
	if c.Token != "" {
		request.Header.Set("Authorization", "Bearer "+c.Token)
	}
	return c.HTTP.Do(request)
}

// ModelInfo fetches file metadata, including sizes, for a repository.
func (c *Client) ModelInfo(ctx context.Context, repo, revision string) (ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	target := c.BaseURL + "/api/models/" + escapeSegments(repo) + "?blobs=true"
	if revision != "" {
		target += "&revision=" + url.QueryEscape(revision)
	}
	response, err := c.request(ctx, http.MethodGet, target, nil)
	if err != nil {
		return ModelInfo{}, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		return ModelInfo{}, &HTTPError{Status: response.StatusCode, Body: string(body)}
	}
	var info ModelInfo
	if err := json.NewDecoder(response.Body).Decode(&info); err != nil {
		return ModelInfo{}, err
	}
	return info, nil
}

var shardRE = regexp.MustCompile(`(?i)^(.*?)-(\d{5})-of-(\d{5})(\.gguf)$`)

// GGUFGroups groups sharded GGUF files and records known sizes.
func GGUFGroups(siblings []Sibling) []Group {
	groups := map[string]*Group{}
	order := []string{}
	for _, sibling := range siblings {
		name := sibling.Filename
		if !strings.HasSuffix(strings.ToLower(name), ".gguf") {
			continue
		}
		groupName := name
		if match := shardRE.FindStringSubmatch(name); len(match) == 5 {
			groupName = match[1] + match[4]
		}
		group, ok := groups[groupName]
		if !ok {
			group = &Group{Name: groupName, SizeKnown: true, FileSizes: map[string]int64{}}
			groups[groupName] = group
			order = append(order, groupName)
		}
		group.Files = append(group.Files, name)
		if sibling.Size == nil {
			group.SizeKnown = false
		} else {
			group.Size += *sibling.Size
			group.FileSizes[name] = *sibling.Size
		}
	}
	result := make([]Group, 0, len(order))
	for _, name := range order {
		group := groups[name]
		sort.Strings(group.Files)
		result = append(result, *group)
	}
	return result
}

var quantRE = regexp.MustCompile(`(?i)(?:^|[^A-Z0-9])((?:IQ2|Q2|IQ3|Q3|IQ4|Q4|Q5|Q6|Q8|F16|BF16|F32)(?:_[A-Z0-9]+)*)`)

var quantRanks = []struct {
	token string
	rank  int
}{
	{"Q2", 20}, {"IQ2", 22}, {"Q3", 30}, {"IQ3", 32}, {"Q4", 40},
	{"IQ4", 42}, {"Q5", 50}, {"Q6", 60}, {"Q8", 80}, {"F16", 90},
	{"BF16", 90}, {"F32", 100},
}

var preferTarget = map[string]int{"speed": 40, "balanced": 45, "quality": 80}

// QuantLabel extracts a short label such as Q4_K_M from a filename.
func QuantLabel(filename string) string {
	match := quantRE.FindStringSubmatch(strings.ToUpper(filename))
	if len(match) > 1 {
		return match[1]
	}
	base := path.Base(filename)
	return strings.TrimSuffix(base, path.Ext(base))
}

// QuantRank returns a numeric quality rank for a GGUF filename.
func QuantRank(filename string) int {
	label := QuantLabel(filename)
	upper := strings.ToUpper(label)
	for _, entry := range quantRanks {
		if upper == entry.token || strings.HasPrefix(upper, entry.token+"_") {
			return entry.rank
		}
	}
	return 0
}

var kQuantRE = regexp.MustCompile(`_K(?:[^A-Z0-9]|$)`)

func quantFinesse(filename string) int {
	upper := strings.ToUpper(filename)
	switch {
	case strings.Contains(upper, "_K_M"):
		return 3
	case strings.Contains(upper, "_K_S"):
		return 2
	case kQuantRE.MatchString(upper):
		return 1
	default:
		return 0
	}
}

// SelectGGUF chooses the best quantization that fits the memory budget.
func SelectGGUF(groups []Group, budgetGB float64, requestedFile, prefer string) (Group, error) {
	if requestedFile != "" {
		for _, group := range groups {
			if group.Name == requestedFile {
				return group, nil
			}
			for _, file := range group.Files {
				if file == requestedFile {
					return group, nil
				}
			}
		}
		return Group{}, fmt.Errorf("GGUF file not found in repository: %s", requestedFile)
	}
	if len(groups) == 0 {
		return Group{}, errors.New("the Hugging Face repository does not contain any GGUF files")
	}

	var fitting []Group
	for _, group := range groups {
		if group.SizeKnown && float64(group.Size) <= budgetGB*(1024*1024*1024) {
			fitting = append(fitting, group)
		}
	}
	target := preferTarget[prefer]
	if target == 0 {
		target = preferTarget["balanced"]
	}
	if len(fitting) > 0 {
		sort.SliceStable(fitting, func(i, j int) bool {
			left := abs(QuantRank(fitting[i].Name) - target)
			right := abs(QuantRank(fitting[j].Name) - target)
			if left != right {
				return left < right
			}
			if quantFinesse(fitting[i].Name) != quantFinesse(fitting[j].Name) {
				return quantFinesse(fitting[i].Name) > quantFinesse(fitting[j].Name)
			}
			return fitting[i].Size < fitting[j].Size
		})
		return fitting[0], nil
	}

	var known []Group
	for _, group := range groups {
		if group.SizeKnown {
			known = append(known, group)
		}
	}
	if len(known) > 0 {
		sort.SliceStable(known, func(i, j int) bool {
			if known[i].Size != known[j].Size {
				return known[i].Size < known[j].Size
			}
			return QuantRank(known[i].Name) > QuantRank(known[j].Name)
		})
		return known[0], nil
	}

	var quants []Group
	for _, group := range groups {
		if QuantRank(group.Name) > 0 {
			quants = append(quants, group)
		}
	}
	if len(quants) == 0 {
		quants = groups
	}
	sort.SliceStable(quants, func(i, j int) bool {
		return QuantRank(quants[i].Name) < QuantRank(quants[j].Name)
	})
	return quants[0], nil
}

func abs(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

// GroupSizeGB returns the group size in gigabytes.
func GroupSizeGB(group Group) float64 {
	return float64(group.Size) / (1024 * 1024 * 1024)
}

const maxDownloadWorkers = 16

// ClampWorkers resolves an explicit or automatic worker count for a file list.
func ClampWorkers(workers, files int) int {
	if files <= 0 {
		return 0
	}
	if workers <= 0 {
		workers = AutoWorkers(files)
	}
	if workers > maxDownloadWorkers {
		workers = maxDownloadWorkers
	}
	if workers > files {
		workers = files
	}
	return workers
}

// AutoWorkers picks a parallel download width for a number of files.
func AutoWorkers(files int) int {
	if files <= 0 {
		return 0
	}
	workers := runtime.NumCPU() * 2
	if workers < 4 {
		workers = 4
	}
	if workers > maxDownloadWorkers {
		workers = maxDownloadWorkers
	}
	if workers > files {
		workers = files
	}
	return workers
}

// DownloadGroup downloads all files of a group concurrently and returns paths.
func (c *Client) DownloadGroup(ctx context.Context, repo, revision, localDir string, group Group, workers int) ([]string, error) {
	if len(group.Files) == 0 {
		return nil, errors.New("no files to download")
	}
	workers = ClampWorkers(workers, len(group.Files))
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	type job struct {
		index int
		name  string
	}
	jobs := make(chan job)
	errCh := make(chan error, 1)
	paths := make([]string, len(group.Files))
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobs {
				target := filepath.Join(localDir, filepath.FromSlash(item.name))
				if err := c.downloadOne(ctx, repo, revision, item.name, target, group.FileSizes[item.name]); err != nil {
					select {
					case errCh <- err:
						cancel()
					default:
					}
					return
				}
				paths[item.index] = target
			}
		}()
	}
	go func() {
		defer close(jobs)
		for index, name := range group.Files {
			select {
			case jobs <- job{index: index, name: name}:
			case <-ctx.Done():
				return
			}
		}
	}()
	wg.Wait()
	select {
	case err := <-errCh:
		return nil, err
	default:
	}
	for _, item := range paths {
		if item == "" {
			return nil, ctx.Err()
		}
	}
	return paths, nil
}

func (c *Client) downloadOne(ctx context.Context, repo, revision, filename, target string, expectedSize int64) error {
	if info, err := os.Stat(target); err == nil && info.Size() > 0 {
		if expectedSize <= 0 || info.Size() == expectedSize {
			return nil
		}
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	if revision == "" {
		revision = "main"
	}
	targetURL := c.BaseURL + "/" + escapeSegments(repo) + "/resolve/" +
		url.PathEscape(revision) + "/" + escapeSegments(filename) + "?download=true"
	response, err := c.request(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		return &HTTPError{Status: response.StatusCode, Body: string(body)}
	}
	dir := filepath.Dir(target)
	temp, err := os.CreateTemp(dir, filepath.Base(target)+".part-*")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer os.Remove(tempName)
	if _, err := io.Copy(temp, response.Body); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	return os.Rename(tempName, target)
}
