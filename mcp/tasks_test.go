// Copyright 2026 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/nekohasekai/go-sdk/jsonrpc"
)

type doubleInput struct {
	N int `json:"n"`
}

type doubleOutput struct {
	Double int `json:"double"`
}

func newTaskServer(opts *TaskOptions) *Server {
	if opts == nil {
		opts = &TaskOptions{PollInterval: func(*CallToolRequest) time.Duration { return 10 * time.Millisecond }}
	}
	return NewServer(testImpl, &ServerOptions{Tasks: opts})
}

// addDoubleTool adds a typed tool that runs as a task until release is
// closed, and reports each task ID on ids.
func addDoubleTool(s *Server, release <-chan struct{}, ids chan<- string) {
	AddTool(s, &Tool{Name: "double"}, func(ctx context.Context, req *CallToolRequest, in doubleInput) (*CallToolResult, doubleOutput, error) {
		return StartTaskFor(ctx, req, func(ctx context.Context, task *ServerTask) (*CallToolResult, doubleOutput, error) {
			ids <- task.ID()
			task.SetStatusMessage("doubling")
			select {
			case <-release:
			case <-ctx.Done():
				return nil, doubleOutput{}, ctx.Err()
			}
			if in.N < 0 {
				return nil, doubleOutput{}, errors.New("negative input")
			}
			return nil, doubleOutput{in.N * 2}, nil
		})
	})
}

func textOf(t *testing.T, res *CallToolResult) string {
	t.Helper()
	if len(res.Content) != 1 {
		t.Fatalf("got %d content blocks, want 1", len(res.Content))
	}
	text, ok := res.Content[0].(*TextContent)
	if !ok {
		t.Fatalf("content is %T, want *TextContent", res.Content[0])
	}
	return text.Text
}

func wireCode(t *testing.T, err error) int64 {
	t.Helper()
	var wireErr *jsonrpc.Error
	if !errors.As(err, &wireErr) {
		t.Fatalf("error %v is not a JSON-RPC error", err)
	}
	return wireErr.Code
}

func TestTaskPolledToResult(t *testing.T) {
	for _, version := range []string{protocolVersion20260728, protocolVersion20251125} {
		t.Run(version, func(t *testing.T) {
			release := make(chan struct{})
			close(release)
			ids := make(chan string, 2)
			srv := newTaskServer(nil)
			addDoubleTool(srv, release, ids)
			cs := connectAt(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{}}, version)

			res, err := cs.CallTool(t.Context(), &CallToolParams{Name: "double", Arguments: doubleInput{21}})
			if err != nil {
				t.Fatal(err)
			}
			if id, task := <-ids, version >= protocolVersion20260728; (id != "") != task {
				t.Errorf("tool call ran as task %q, want a task: %v", id, task)
			}
			if got, want := textOf(t, res), `{"double":42}`; got != want {
				t.Errorf("result %s, want %s", got, want)
			}

			res, err = cs.CallTool(t.Context(), &CallToolParams{Name: "double", Arguments: doubleInput{-1}})
			<-ids
			if err != nil {
				t.Fatal(err)
			}
			if !res.IsError || textOf(t, res) != "negative input" {
				t.Errorf("got %+v, want a tool error result", res)
			}
		})
	}
}

func connectAt(t *testing.T, s *Server, opts *ClientOptions, version string) *ClientSession {
	if version == protocolVersion20251125 {
		return mustConnectOldProtocol(t, s, opts)
	}
	return mustConnect(t, s, opts)
}

func TestTaskSynchronousWithoutExtension(t *testing.T) {
	release := make(chan struct{})
	close(release)
	ids := make(chan string, 1)
	srv := newTaskServer(nil)
	addDoubleTool(srv, release, ids)
	cs := mustConnect(t, srv, nil)

	res, err := cs.CallTool(t.Context(), &CallToolParams{Name: "double", Arguments: doubleInput{2}})
	if err != nil {
		t.Fatal(err)
	}
	if id := <-ids; id != "" {
		t.Errorf("tool call ran as task %q, want synchronously", id)
	}
	if res.Task() != nil || textOf(t, res) != `{"double":4}` {
		t.Errorf("got %+v, want the direct result", res)
	}
}

func TestTaskManualPolling(t *testing.T) {
	release := make(chan struct{})
	ids := make(chan string, 1)
	srv := newTaskServer(&TaskOptions{PollInterval: func(*CallToolRequest) time.Duration { return 250 * time.Millisecond }, Retention: time.Minute})
	addDoubleTool(srv, release, ids)
	cs := mustConnect(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{DisablePolling: true}})
	ctx := t.Context()

	res, err := cs.CallTool(ctx, &CallToolParams{Name: "double", Arguments: doubleInput{5}})
	if err != nil {
		t.Fatal(err)
	}
	created := res.Task()
	if created == nil {
		t.Fatalf("got %+v, want a task", res)
	}
	if id := <-ids; created.TaskID != id {
		t.Errorf("task ID %q, want %q", created.TaskID, id)
	}
	if created.Status != TaskStatusWorking || created.PollIntervalMs != 250 || created.TTLMs != nil {
		t.Errorf("created task %+v, want working with poll interval 250 and unlimited TTL", created)
	}

	close(release)
	var task *Task
	for {
		got, err := cs.GetTask(ctx, &GetTaskParams{TaskID: created.TaskID})
		if err != nil {
			t.Fatal(err)
		}
		if task = &got.Task; task.Status.terminal() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if task.Status != TaskStatusCompleted || task.Result == nil || textOf(t, task.Result) != `{"double":10}` {
		t.Fatalf("got %+v, want completed with the result", task)
	}
	if task.StatusMessage != "doubling" {
		t.Errorf("status message %q, want %q", task.StatusMessage, "doubling")
	}
	if task.TTLMs == nil || *task.TTLMs < 60000 {
		t.Errorf("TTL %v, want at least the retention", task.TTLMs)
	}
	if err := cs.CancelTask(ctx, &CancelTaskParams{TaskID: task.TaskID}); err != nil {
		t.Errorf("cancelling a completed task: %v", err)
	}
	if got, err := cs.GetTask(ctx, &GetTaskParams{TaskID: task.TaskID}); err != nil || got.Status != TaskStatusCompleted {
		t.Errorf("after cancel: %+v, %v; want still completed", got, err)
	}
}

func TestTaskInput(t *testing.T) {
	nameSchema := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"name": {Type: "string"}},
	}
	srv := newTaskServer(nil)
	srv.AddTool(&Tool{Name: "greet", InputSchema: &jsonschema.Schema{Type: "object"}}, func(ctx context.Context, req *CallToolRequest) (*CallToolResult, error) {
		return StartTask(ctx, req, func(ctx context.Context, task *ServerTask) (*CallToolResult, error) {
			responses, err := task.RequestInput(ctx, InputRequestMap{
				"name": &ElicitParams{Message: "Name?", RequestedSchema: nameSchema},
			})
			if err != nil {
				return nil, err
			}
			if task.ID() != "" {
				if _, err := task.RequestInput(ctx, InputRequestMap{"name": &ElicitParams{Message: "Again?"}}); err == nil {
					return nil, errors.New("reusing an input request key succeeded")
				}
			}
			name := responses["name"].(*ElicitResult).Content["name"]
			return &CallToolResult{Content: []Content{&TextContent{Text: "hello " + name.(string)}}}, nil
		})
	})

	for _, test := range []struct {
		version string
		tasks   *ClientTaskOptions
		code    int64
	}{
		{protocolVersion20260728, &ClientTaskOptions{}, 0},
		{protocolVersion20251125, &ClientTaskOptions{}, 0},
		{protocolVersion20251125, nil, 0},
		{protocolVersion20260728, nil, CodeMissingRequiredClientCapabilities},
	} {
		t.Run(test.version+"/"+map[bool]string{true: "tasks", false: "sync"}[test.tasks != nil], func(t *testing.T) {
			var calls atomic.Int32
			cs := connectAt(t, srv, &ClientOptions{
				Tasks: test.tasks,
				ElicitationHandler: func(context.Context, *ElicitRequest) (*ElicitResult, error) {
					calls.Add(1)
					return &ElicitResult{Action: "accept", Content: map[string]any{"name": "Luca"}}, nil
				},
			}, test.version)
			res, err := cs.CallTool(t.Context(), &CallToolParams{Name: "greet"})
			if test.code != 0 {
				if err == nil || wireCode(t, err) != test.code {
					t.Fatalf("got %v, want error code %d", err, test.code)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := textOf(t, res); got != "hello Luca" {
				t.Errorf("result %q, want %q", got, "hello Luca")
			}
			if n := calls.Load(); n != 1 {
				t.Errorf("elicitation handler called %d times, want 1", n)
			}
		})
	}
}

func TestTaskCancel(t *testing.T) {
	ids := make(chan string, 1)
	stopped := make(chan error, 1)
	srv := newTaskServer(nil)
	srv.AddTool(&Tool{Name: "wait", InputSchema: &jsonschema.Schema{Type: "object"}}, func(ctx context.Context, req *CallToolRequest) (*CallToolResult, error) {
		return StartTask(ctx, req, func(ctx context.Context, task *ServerTask) (*CallToolResult, error) {
			ids <- task.ID()
			<-ctx.Done()
			stopped <- ctx.Err()
			return &CallToolResult{Content: []Content{&TextContent{Text: "late"}}}, nil
		})
	})
	// Cancel once the client polls, which is after it has received the task.
	ctx, cancel := context.WithCancel(t.Context())
	c := NewClient(testImpl, &ClientOptions{Tasks: &ClientTaskOptions{}})
	c.AddSendingMiddleware(func(next MethodHandler) MethodHandler {
		return func(ctx context.Context, method string, req Request) (Result, error) {
			if method == methodGetTask {
				cancel()
			}
			return next(ctx, method, req)
		}
	})
	st, ct := NewInMemoryTransports()
	ss, err := srv.Connect(t.Context(), st, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ss.Close()
	cs, err := c.Connect(t.Context(), ct, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()

	if _, err := cs.CallTool(ctx, &CallToolParams{Name: "wait"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want %v", err, context.Canceled)
	}
	id := <-ids
	select {
	case err := <-stopped:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("task context ended with %v, want %v", err, context.Canceled)
		}
	case <-time.After(time.Second):
		t.Fatal("task context was not cancelled")
	}
	got, err := cs.GetTask(t.Context(), &GetTaskParams{TaskID: id})
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != TaskStatusCancelled || got.Result != nil {
		t.Errorf("got %+v, want cancelled without a result", got.Task)
	}
}

func TestTaskFailed(t *testing.T) {
	srv := newTaskServer(nil)
	srv.AddTool(&Tool{Name: "fail", InputSchema: &jsonschema.Schema{Type: "object"}}, func(ctx context.Context, req *CallToolRequest) (*CallToolResult, error) {
		return StartTask(ctx, req, func(context.Context, *ServerTask) (*CallToolResult, error) {
			return nil, &jsonrpc.Error{Code: -32001, Message: "quota exceeded"}
		})
	})
	cs := mustConnect(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{}})

	_, err := cs.CallTool(t.Context(), &CallToolParams{Name: "fail"})
	if err == nil || wireCode(t, err) != -32001 {
		t.Fatalf("got %v, want error code -32001", err)
	}
}

func TestTaskProtocolErrors(t *testing.T) {
	ctx := t.Context()
	release := make(chan struct{})
	defer close(release)
	ids := make(chan string, 1)
	srv := newTaskServer(nil)
	addDoubleTool(srv, release, ids)
	withTasks := mustConnect(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{DisablePolling: true}})
	res, err := withTasks.CallTool(ctx, &CallToolParams{Name: "double", Arguments: doubleInput{1}})
	if err != nil {
		t.Fatal(err)
	}
	id := res.Task().TaskID

	if _, err := withTasks.GetTask(ctx, &GetTaskParams{TaskID: "unknown"}); err == nil || wireCode(t, err) != jsonrpc.CodeInvalidParams {
		t.Errorf("unknown task: got %v, want invalid params", err)
	}

	without := mustConnect(t, srv, nil)
	for name, call := range map[string]func() error{
		"get": func() error {
			_, err := without.GetTask(ctx, &GetTaskParams{TaskID: id})
			return err
		},
		"update": func() error {
			return without.UpdateTask(ctx, &UpdateTaskParams{TaskID: id, InputResponses: InputResponseMap{}})
		},
		"cancel": func() error { return without.CancelTask(ctx, &CancelTaskParams{TaskID: id}) },
		"listen": func() error { return without.ListenTasks(ctx, id) },
	} {
		err := call()
		if err == nil || wireCode(t, err) != CodeMissingRequiredClientCapabilities {
			t.Errorf("%s without the extension: got %v, want error code %d", name, err, CodeMissingRequiredClientCapabilities)
			continue
		}
		var wireErr *jsonrpc.Error
		errors.As(err, &wireErr)
		var data MissingRequiredClientCapabilityData
		if err := json.Unmarshal(wireErr.Data, &data); err != nil || data.RequiredCapabilities == nil || data.RequiredCapabilities.Extensions[TasksExtensionID] == nil {
			t.Errorf("%s without the extension: data %s does not require the extension", name, wireErr.Data)
		}
	}

	disabled := mustConnect(t, NewServer(testImpl, nil), &ClientOptions{Tasks: &ClientTaskOptions{}})
	if _, err := disabled.GetTask(ctx, &GetTaskParams{TaskID: id}); err == nil || wireCode(t, err) != jsonrpc.CodeMethodNotFound {
		t.Errorf("server without tasks: got %v, want method not found", err)
	}
}

func TestTaskNotifications(t *testing.T) {
	release := make(chan struct{})
	ids := make(chan string, 1)
	srv := newTaskServer(nil)
	addDoubleTool(srv, release, ids)
	statuses := make(chan Task, 16)
	cs := mustConnect(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{
		DisablePolling: true,
		StatusHandler: func(_ context.Context, req *TaskStatusRequest) {
			statuses <- req.Params.Task
		},
	}})

	res, err := cs.CallTool(t.Context(), &CallToolParams{Name: "double", Arguments: doubleInput{4}})
	if err != nil {
		t.Fatal(err)
	}
	id := <-ids
	listenCtx, stopListening := context.WithCancel(t.Context())
	defer stopListening()
	listening := make(chan error, 1)
	go func() { listening <- cs.ListenTasks(listenCtx, id, "unknown") }()

	next := func() Task {
		t.Helper()
		select {
		case task := <-statuses:
			if task.TaskID != res.Task().TaskID {
				t.Fatalf("notification for task %q, want %q", task.TaskID, res.Task().TaskID)
			}
			return task
		case err := <-listening:
			t.Fatalf("listen ended: %v", err)
		case <-time.After(time.Second):
			t.Fatal("no task notification")
		}
		panic("unreachable")
	}
	if task := next(); task.Status != TaskStatusWorking {
		t.Errorf("first notification %+v, want the working state", task)
	}
	close(release)
	task := next()
	for !task.Status.terminal() {
		task = next()
	}
	if task.Status != TaskStatusCompleted || textOf(t, task.Result) != `{"double":8}` {
		t.Errorf("last notification %+v, want completed with the result", task)
	}
}

func TestTaskStreamable(t *testing.T) {
	release := make(chan struct{})
	ids := make(chan string, 1)
	remoteAddrs := make(chan string, 1)
	srv := newTaskServer(&TaskOptions{PollInterval: func(req *CallToolRequest) time.Duration {
		remoteAddrs <- req.Extra.RemoteAddr
		return 10 * time.Millisecond
	}})
	addDoubleTool(srv, release, ids)
	handler := NewStreamableHTTPHandler(func(*http.Request) *Server { return srv }, &StreamableHTTPOptions{Stateless: true})
	var mu sync.Mutex
	names := map[string]string{}
	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if method := r.Header.Get(methodHeader); method == methodGetTask {
			mu.Lock()
			names[method] = r.Header.Get(nameHeader)
			mu.Unlock()
		}
		handler.ServeHTTP(w, r)
	}))
	defer httpServer.Close()

	cs, err := NewClient(testImpl, &ClientOptions{Tasks: &ClientTaskOptions{}}).Connect(t.Context(), &StreamableClientTransport{Endpoint: httpServer.URL}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer cs.Close()
	go func() {
		<-time.After(50 * time.Millisecond)
		close(release)
	}()
	res, err := cs.CallTool(t.Context(), &CallToolParams{Name: "double", Arguments: doubleInput{3}})
	if err != nil {
		t.Fatal(err)
	}
	if got := textOf(t, res); got != `{"double":6}` {
		t.Errorf("result %s, want %s", got, `{"double":6}`)
	}
	if remoteAddr := <-remoteAddrs; !netip.MustParseAddrPort(remoteAddr).Addr().IsLoopback() {
		t.Errorf("poll interval chosen for remote address %q, want a loopback address", remoteAddr)
	}
	id := <-ids
	mu.Lock()
	defer mu.Unlock()
	if names[methodGetTask] != id {
		t.Errorf("tasks/get carried Mcp-Name %q, want the task ID %q", names[methodGetTask], id)
	}
}

func TestTaskCreateAfter(t *testing.T) {
	const wait = 200 * time.Millisecond
	quick := make(chan struct{})
	close(quick)
	ids := make(chan string, 1)
	srv := newTaskServer(&TaskOptions{CreateAfter: wait})
	addDoubleTool(srv, quick, ids)
	srv.AddTool(&Tool{Name: "slow", InputSchema: &jsonschema.Schema{Type: "object"}}, func(ctx context.Context, req *CallToolRequest) (*CallToolResult, error) {
		return StartTask(ctx, req, func(ctx context.Context, task *ServerTask) (*CallToolResult, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
	})
	srv.AddTool(&Tool{Name: "ask", InputSchema: &jsonschema.Schema{Type: "object"}}, func(ctx context.Context, req *CallToolRequest) (*CallToolResult, error) {
		return StartTask(ctx, req, func(ctx context.Context, task *ServerTask) (*CallToolResult, error) {
			_, err := task.RequestInput(ctx, InputRequestMap{"ok": &ElicitParams{Message: "OK?"}})
			return nil, err
		})
	})
	cs := mustConnect(t, srv, &ClientOptions{Tasks: &ClientTaskOptions{DisablePolling: true}})
	ctx := t.Context()

	res, err := cs.CallTool(ctx, &CallToolParams{Name: "double", Arguments: doubleInput{7}})
	if err != nil {
		t.Fatal(err)
	}
	<-ids
	if res.Task() != nil || textOf(t, res) != `{"double":14}` {
		t.Errorf("quick call: got %+v, want the direct result", res)
	}

	for _, test := range []struct {
		tool   string
		status TaskStatus
	}{
		{"slow", TaskStatusWorking},
		{"ask", TaskStatusInputRequired},
	} {
		started := time.Now()
		res, err := cs.CallTool(ctx, &CallToolParams{Name: test.tool})
		if err != nil {
			t.Fatal(err)
		}
		if elapsed := time.Since(started); elapsed < wait {
			t.Errorf("%s: answered after %v, want after %v", test.tool, elapsed, wait)
		}
		task := res.Task()
		if task == nil || task.Status != test.status {
			t.Errorf("%s: got %+v, want a task with status %s", test.tool, res, test.status)
			continue
		}
		if _, err := cs.GetTask(ctx, &GetTaskParams{TaskID: task.TaskID}); err != nil {
			t.Errorf("%s: %v", test.tool, err)
		}
		if err := cs.CancelTask(ctx, &CancelTaskParams{TaskID: task.TaskID}); err != nil {
			t.Error(err)
		}
	}
}
