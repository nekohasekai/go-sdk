// Copyright 2026 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package mcp

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"sync"
	"time"

	"github.com/nekohasekai/go-sdk/jsonrpc"
)

// TasksExtensionID identifies the tasks extension, which lets a server answer
// "tools/call" with a task that the client polls for the final result.
//
// See https://github.com/modelcontextprotocol/ext-tasks.
const TasksExtensionID = "io.modelcontextprotocol/tasks"

const (
	defaultTaskPollInterval = time.Second
	defaultTaskRetention    = 10 * time.Minute
)

// TaskStatus is the status of a [Task].
type TaskStatus string

const (
	// TaskStatusWorking means the request is being processed.
	TaskStatusWorking TaskStatus = "working"
	// TaskStatusInputRequired means the server waits for the client to
	// fulfill [Task.InputRequests].
	TaskStatusInputRequired TaskStatus = "input_required"
	// TaskStatusCompleted means the request finished with [Task.Result],
	// including a tool result that reports an error.
	TaskStatusCompleted TaskStatus = "completed"
	// TaskStatusFailed means the request failed with the JSON-RPC error
	// [Task.Error].
	TaskStatusFailed TaskStatus = "failed"
	// TaskStatusCancelled means the request was cancelled before it finished.
	TaskStatusCancelled TaskStatus = "cancelled"
)

func (s TaskStatus) terminal() bool {
	return s == TaskStatusCompleted || s == TaskStatusFailed || s == TaskStatusCancelled
}

// A Task reports the state of a request that a server processes
// asynchronously, as defined by the tasks extension ([TasksExtensionID]).
type Task struct {
	// TaskID identifies the task in "tasks/get", "tasks/update" and
	// "tasks/cancel".
	TaskID string     `json:"taskId"`
	Status TaskStatus `json:"status"`
	// StatusMessage optionally describes the current status.
	StatusMessage string    `json:"statusMessage,omitempty"`
	CreatedAt     time.Time `json:"createdAt"`
	LastUpdatedAt time.Time `json:"lastUpdatedAt"`
	// TTLMs is the time to live from creation in milliseconds, after which
	// the server may discard the task. Nil means unlimited.
	TTLMs *int `json:"ttlMs"`
	// PollIntervalMs is the polling interval in milliseconds that the server
	// suggests.
	PollIntervalMs int `json:"pollIntervalMs,omitempty"`

	// InputRequests holds the outstanding requests that the client fulfills
	// with "tasks/update", when Status is [TaskStatusInputRequired]. A key is
	// never reused within a task.
	InputRequests InputRequestMap `json:"inputRequests,omitempty"`
	// Result is the final result, when Status is [TaskStatusCompleted].
	Result *CallToolResult `json:"result,omitempty"`
	// Error is the error that the request failed with, when Status is
	// [TaskStatusFailed].
	Error *jsonrpc.Error `json:"error,omitempty"`
}

// GetTaskParams are the parameters for "tasks/get".
type GetTaskParams struct {
	Meta   `json:"_meta,omitempty"`
	TaskID string `json:"taskId"`
}

func (x *GetTaskParams) isParams()              {}
func (x *GetTaskParams) isNil() bool            { return x == nil }
func (x *GetTaskParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *GetTaskParams) SetProgressToken(t any) { setProgressToken(x, t) }

// GetTaskResult is the result of "tasks/get": the current state of the task.
type GetTaskResult struct {
	completeResultWithType
	Meta `json:"_meta,omitempty"`
	Task
}

func (*GetTaskResult) isResult()     {}
func (x *GetTaskResult) isNil() bool { return x == nil }

// UpdateTaskParams are the parameters for "tasks/update".
type UpdateTaskParams struct {
	Meta   `json:"_meta,omitempty"`
	TaskID string `json:"taskId"`
	// InputResponses maps keys of [Task.InputRequests] to their responses.
	InputResponses InputResponseMap `json:"inputResponses"`
}

func (x *UpdateTaskParams) isParams()              {}
func (x *UpdateTaskParams) isNil() bool            { return x == nil }
func (x *UpdateTaskParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *UpdateTaskParams) SetProgressToken(t any) { setProgressToken(x, t) }

// CancelTaskParams are the parameters for "tasks/cancel".
type CancelTaskParams struct {
	Meta   `json:"_meta,omitempty"`
	TaskID string `json:"taskId"`
}

func (x *CancelTaskParams) isParams()              {}
func (x *CancelTaskParams) isNil() bool            { return x == nil }
func (x *CancelTaskParams) GetProgressToken() any  { return getProgressToken(x) }
func (x *CancelTaskParams) SetProgressToken(t any) { setProgressToken(x, t) }

// taskAckResult is the empty result of "tasks/update" and "tasks/cancel".
type taskAckResult struct {
	completeResultWithType
	Meta `json:"_meta,omitempty"`
}

func (*taskAckResult) isResult()     {}
func (x *taskAckResult) isNil() bool { return x == nil }

// TaskStatusNotificationParams are the parameters for "notifications/tasks":
// the state of the task, as "tasks/get" reports it.
type TaskStatusNotificationParams struct {
	Meta `json:"_meta,omitempty"`
	Task
}

func (x *TaskStatusNotificationParams) isParams()   {}
func (x *TaskStatusNotificationParams) isNil() bool { return x == nil }

var errMissingTasksCapability = &jsonrpc.Error{
	Code:    CodeMissingRequiredClientCapabilities,
	Message: "missing required client capability",
	Data:    json.RawMessage(`{"requiredCapabilities":{"extensions":{"` + TasksExtensionID + `":{}}}}`),
}

func declaresTasks(caps *ClientCapabilities) bool {
	if caps == nil {
		return false
	}
	_, ok := caps.Extensions[TasksExtensionID]
	return ok
}

func requestUserID(extra *RequestExtra) string {
	if extra == nil || extra.TokenInfo == nil {
		return ""
	}
	return extra.TokenInfo.UserID
}

// TaskOptions configures the tasks extension on a server. See
// [ServerOptions.Tasks].
type TaskOptions struct {
	// PollInterval is the polling interval suggested to clients.
	// If zero, it defaults to one second.
	PollInterval time.Duration
	// Retention is how long a task stays retrievable after it completes,
	// fails or is cancelled. If zero, it defaults to ten minutes. Until then,
	// a task reports an unlimited time to live.
	Retention time.Duration
}

// A ServerTask is the server side of a tool call started with [StartTask] or
// [StartTaskFor].
type ServerTask struct {
	// server is nil when the call runs synchronously, and session is set only
	// then.
	server  *Server
	session *ServerSession
	userID  string
	cancel  context.CancelFunc

	// notifyMu orders each change with the notifications that report it.
	notifyMu sync.Mutex

	mu          sync.Mutex
	task        Task
	result      *CallToolResult
	err         *jsonrpc.Error
	pending     map[string]pendingInput
	used        map[string]bool
	subscribers map[*ServerSession]jsonrpc.ID
}

type pendingInput struct {
	request InputRequest
	waiter  *inputWaiter
}

type inputWaiter struct {
	responses InputResponseMap
	remaining int
	done      chan struct{}
}

// ID returns the task ID, or "" when the call runs synchronously.
func (t *ServerTask) ID() string { return t.task.TaskID }

// SetStatusMessage sets the message that describes the current status to the
// client. It has no effect when the call runs synchronously.
func (t *ServerTask) SetStatusMessage(message string) {
	if t.server == nil {
		return
	}
	t.update(func(time.Time) bool {
		if t.task.Status.terminal() || t.task.StatusMessage == message {
			return false
		}
		t.task.StatusMessage = message
		return true
	})
}

// RequestInput asks the client to fulfill requests, and waits until it has
// responded to all of them or ctx is done.
//
// While a task waits, its status is input_required and the client receives
// the requests through "tasks/get" or "notifications/tasks". Keys must be
// unique over the lifetime of the task.
//
// When the call runs synchronously, the requests are sent to the client
// directly if the session uses a protocol version before 2026-07-28, which
// still allows server-initiated requests. Otherwise RequestInput fails with a
// [CodeMissingRequiredClientCapabilities] error naming the tasks extension.
func (t *ServerTask) RequestInput(ctx context.Context, requests InputRequestMap) (InputResponseMap, error) {
	if t.server == nil {
		if clientSupportsMultiRoundTrip(t.session) {
			return nil, errMissingTasksCapability
		}
		return fulfillServerInputRequests(ctx, t.session, requests)
	}
	if len(requests) == 0 {
		return InputResponseMap{}, nil
	}
	w := &inputWaiter{
		responses: make(InputResponseMap, len(requests)),
		remaining: len(requests),
		done:      make(chan struct{}),
	}
	var err error
	t.update(func(time.Time) bool {
		if t.task.Status.terminal() {
			err = fmt.Errorf("task %q is %s", t.task.TaskID, t.task.Status)
			return false
		}
		for key := range requests {
			if t.used[key] {
				err = fmt.Errorf("input request key %q was already used in task %q", key, t.task.TaskID)
				return false
			}
		}
		for key, request := range requests {
			t.used[key] = true
			t.pending[key] = pendingInput{request, w}
		}
		t.task.Status = TaskStatusInputRequired
		return true
	})
	if err != nil {
		return nil, err
	}
	select {
	case <-w.done:
		return w.responses, nil
	case <-ctx.Done():
		t.update(func(time.Time) bool {
			withdrawn := false
			for key := range requests {
				if p, ok := t.pending[key]; ok && p.waiter == w {
					delete(t.pending, key)
					withdrawn = true
				}
			}
			if withdrawn && len(t.pending) == 0 && t.task.Status == TaskStatusInputRequired {
				t.task.Status = TaskStatusWorking
			}
			return withdrawn
		})
		return nil, ctx.Err()
	}
}

// update applies mutate to the task, and if it reports a change, records the
// time and notifies the subscribed listen streams.
func (t *ServerTask) update(mutate func(now time.Time) bool) {
	t.notifyMu.Lock()
	defer t.notifyMu.Unlock()
	now := time.Now().UTC().Truncate(time.Millisecond)
	t.mu.Lock()
	if !mutate(now) {
		t.mu.Unlock()
		return
	}
	t.task.LastUpdatedAt = now
	snapshot := t.snapshotLocked()
	subscribers := maps.Clone(t.subscribers)
	t.mu.Unlock()
	t.server.notifySubscribedSessions(subscribers, notificationTasks, func() Params {
		return &TaskStatusNotificationParams{Task: snapshot}
	})
}

func (t *ServerTask) snapshotLocked() Task {
	task := t.task
	switch task.Status {
	case TaskStatusInputRequired:
		task.InputRequests = make(InputRequestMap, len(t.pending))
		for key, p := range t.pending {
			task.InputRequests[key] = p.request
		}
	case TaskStatusCompleted:
		task.Result = t.result
	case TaskStatusFailed:
		task.Error = t.err
	}
	return task
}

func (t *ServerTask) complete(res *CallToolResult, err error) {
	if err == nil {
		if res == nil {
			res = &CallToolResult{}
		}
		if asksForInput(res) {
			err = &jsonrpc.Error{
				Code:    jsonrpc.CodeInternalError,
				Message: "server bug: a task result cannot ask for input, use ServerTask.RequestInput",
			}
		} else if res.Content == nil {
			res.Content = []Content{}
		}
		res.setResultType(resultTypeComplete)
	}
	t.update(func(now time.Time) bool {
		if t.task.Status.terminal() {
			return false
		}
		if err != nil {
			wireErr, ok := err.(*jsonrpc.Error)
			if !ok {
				wireErr = &jsonrpc.Error{Code: jsonrpc.CodeInternalError, Message: err.Error()}
				var wrapped *jsonrpc.Error
				if errors.As(err, &wrapped) {
					wireErr.Code = wrapped.Code
				}
			}
			t.task.Status = TaskStatusFailed
			t.task.StatusMessage = wireErr.Message
			t.err = wireErr
		} else {
			t.task.Status = TaskStatusCompleted
			t.result = res
		}
		t.endLocked(now)
		return true
	})
}

func (t *ServerTask) endLocked(now time.Time) {
	retention := t.server.opts.Tasks.Retention
	ttl := int((now.Sub(t.task.CreatedAt) + retention).Milliseconds())
	t.task.TTLMs = &ttl
	clear(t.pending)
	t.cancel()
	s, id := t.server, t.task.TaskID
	time.AfterFunc(retention, func() {
		s.tasksMu.Lock()
		delete(s.tasks, id)
		s.tasksMu.Unlock()
	})
}

// newServerTask returns a task that runs synchronously within req, unless the
// server and the client both enable tasks: then the task is registered and
// returned with a context for running it in the background.
func newServerTask(ctx context.Context, req *CallToolRequest) (*ServerTask, context.Context) {
	s := req.Session.server
	if s.opts.Tasks == nil || !declaresTasks(req.ClientCapabilities()) {
		return &ServerTask{session: req.Session}, ctx
	}
	ctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	now := time.Now().UTC().Truncate(time.Millisecond)
	t := &ServerTask{
		server: s,
		userID: requestUserID(req.Extra),
		cancel: cancel,
		task: Task{
			TaskID:         rand.Text(),
			Status:         TaskStatusWorking,
			CreatedAt:      now,
			LastUpdatedAt:  now,
			PollIntervalMs: int(s.opts.Tasks.PollInterval.Milliseconds()),
		},
		pending:     make(map[string]pendingInput),
		used:        make(map[string]bool),
		subscribers: make(map[*ServerSession]jsonrpc.ID),
	}
	s.tasksMu.Lock()
	s.tasks[t.task.TaskID] = t
	s.tasksMu.Unlock()
	return t, ctx
}

// StartTask runs a tool call as a task when the server enables
// [ServerOptions.Tasks] and the client declares the tasks extension on req.
// The returned result is then the task handle for the tool handler to
// return, and run continues in the background with a context that is
// independent of the request and is cancelled when the client cancels the
// task.
//
// Otherwise, StartTask calls run synchronously and returns its results.
//
// run reports its outcome as a [ToolHandler] does: its result completes the
// task, and its error fails the task with a JSON-RPC error.
func StartTask(ctx context.Context, req *CallToolRequest, run func(context.Context, *ServerTask) (*CallToolResult, error)) (*CallToolResult, error) {
	t, ctx := newServerTask(ctx, req)
	if t.server == nil {
		return run(ctx, t)
	}
	created := t.task
	go func() { t.complete(run(ctx, t)) }()
	return &CallToolResult{task: &created}, nil
}

type toolResultKey struct{}

// StartTaskFor is [StartTask] for a [ToolHandlerFor]. It must be called by a
// handler added with [AddTool] with the same Out type, and processes the
// results of run as that handler's results are processed.
func StartTaskFor[Out any](ctx context.Context, req *CallToolRequest, run func(context.Context, *ServerTask) (*CallToolResult, Out, error)) (*CallToolResult, Out, error) {
	toolResult, ok := ctx.Value(toolResultKey{}).(func(*CallToolResult, Out, error) (*CallToolResult, error))
	if !ok {
		panic(fmt.Sprintf("StartTaskFor[%v] called outside a ToolHandlerFor with that output type", reflect.TypeFor[Out]()))
	}
	t, ctx := newServerTask(ctx, req)
	if t.server == nil {
		return run(ctx, t)
	}
	created := t.task
	go func() { t.complete(toolResult(run(ctx, t))) }()
	var zero Out
	return &CallToolResult{task: &created}, zero, nil
}

func lookupTask[P Params](s *Server, req *ServerRequest[P], id string) (*ServerTask, error) {
	if s.opts.Tasks == nil {
		return nil, &jsonrpc.Error{Code: jsonrpc.CodeMethodNotFound, Message: "tasks are not enabled"}
	}
	if !declaresTasks(req.ClientCapabilities()) {
		return nil, errMissingTasksCapability
	}
	s.tasksMu.Lock()
	t := s.tasks[id]
	s.tasksMu.Unlock()
	if t == nil || t.userID != requestUserID(req.Extra) {
		return nil, &jsonrpc.Error{Code: jsonrpc.CodeInvalidParams, Message: fmt.Sprintf("task %q not found", id)}
	}
	return t, nil
}

func (s *Server) getTask(_ context.Context, req *GetTaskRequest) (*GetTaskResult, error) {
	t, err := lookupTask(s, req, req.Params.TaskID)
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return &GetTaskResult{Task: t.snapshotLocked()}, nil
}

func (s *Server) updateTask(_ context.Context, req *UpdateTaskRequest) (*taskAckResult, error) {
	t, err := lookupTask(s, req, req.Params.TaskID)
	if err != nil {
		return nil, err
	}
	t.update(func(time.Time) bool {
		answered := false
		for key, response := range req.Params.InputResponses {
			p, ok := t.pending[key]
			if !ok {
				continue
			}
			delete(t.pending, key)
			p.waiter.responses[key] = response
			p.waiter.remaining--
			if p.waiter.remaining == 0 {
				close(p.waiter.done)
			}
			answered = true
		}
		if answered && len(t.pending) == 0 && t.task.Status == TaskStatusInputRequired {
			t.task.Status = TaskStatusWorking
		}
		return answered
	})
	return &taskAckResult{}, nil
}

func (s *Server) cancelTask(_ context.Context, req *CancelTaskRequest) (*taskAckResult, error) {
	t, err := lookupTask(s, req, req.Params.TaskID)
	if err != nil {
		return nil, err
	}
	t.update(func(now time.Time) bool {
		if t.task.Status.terminal() {
			return false
		}
		t.task.Status = TaskStatusCancelled
		t.endLocked(now)
		return true
	})
	return &taskAckResult{}, nil
}

// ClientTaskOptions configures the tasks extension on a client. See
// [ClientOptions.Tasks].
type ClientTaskOptions struct {
	// DisablePolling makes [ClientSession.CallTool] return the task handle
	// that a server sends in place of the result, as reported by
	// [CallToolResult.Task]. By default, CallTool polls the task, fulfills its
	// input requests with the client's handlers, and returns its final
	// result; when the context of CallTool is done, the task is cancelled.
	DisablePolling bool
	// StatusHandler, if non-nil, is called for each "notifications/tasks"
	// that the server sends on a stream opened by [ClientSession.ListenTasks].
	StatusHandler func(context.Context, *TaskStatusRequest)
}

// GetTask makes a "tasks/get" request to the server.
func (cs *ClientSession) GetTask(ctx context.Context, params *GetTaskParams) (*GetTaskResult, error) {
	if cs.usesNewProtocol() {
		params = injectRequestMeta(cs, params)
	}
	return handleSend[*GetTaskResult](ctx, methodGetTask, newClientRequest(cs, orZero[Params](params)))
}

// UpdateTask makes a "tasks/update" request to the server, which responds to
// the task's input requests.
func (cs *ClientSession) UpdateTask(ctx context.Context, params *UpdateTaskParams) error {
	if cs.usesNewProtocol() {
		params = injectRequestMeta(cs, params)
	}
	_, err := handleSend[*taskAckResult](ctx, methodUpdateTask, newClientRequest(cs, orZero[Params](params)))
	return err
}

// CancelTask makes a "tasks/cancel" request to the server. The server
// acknowledges it before the task stops, and the task may still end with
// another status.
func (cs *ClientSession) CancelTask(ctx context.Context, params *CancelTaskParams) error {
	if cs.usesNewProtocol() {
		params = injectRequestMeta(cs, params)
	}
	_, err := handleSend[*taskAckResult](ctx, methodCancelTask, newClientRequest(cs, orZero[Params](params)))
	return err
}

// ListenTasks opens a "subscriptions/listen" stream for the "notifications/tasks"
// of the given tasks, which are delivered to [ClientTaskOptions.StatusHandler].
// The server first reports the current state of each task it accepts.
//
// ListenTasks blocks until ctx is done or the server ends the stream, which
// it does at once when it accepts none of the tasks.
func (cs *ClientSession) ListenTasks(ctx context.Context, taskIDs ...string) error {
	return cs.subscriptionsListen(ctx, &SubscriptionsListenParams{
		Notifications: &NotificationSubscriptions{TaskIDs: taskIDs},
	})
}

func (c *Client) callTaskStatusHandler(ctx context.Context, req *TaskStatusRequest) (Result, error) {
	if c.opts.Tasks != nil && c.opts.Tasks.StatusHandler != nil {
		c.opts.Tasks.StatusHandler(ctx, req)
	}
	return nil, nil
}

func clientTaskMiddleware() Middleware {
	return func(next MethodHandler) MethodHandler {
		return func(ctx context.Context, method string, req Request) (Result, error) {
			res, err := next(ctx, method, req)
			if err != nil || method != methodCallTool {
				return res, err
			}
			created, ok := res.(*CallToolResult)
			if !ok || created.task == nil {
				return res, nil
			}
			cs, ok := req.GetSession().(*ClientSession)
			if !ok {
				return res, nil
			}
			return awaitTask(ctx, cs, created.task)
		}
	}
}

// awaitTask polls task until it ends, fulfilling each of its input requests
// once.
func awaitTask(ctx context.Context, cs *ClientSession, task *Task) (*CallToolResult, error) {
	id := task.TaskID
	cancel := func() {
		ctx, stop := context.WithTimeout(context.WithoutCancel(ctx), notifyTimeout)
		defer stop()
		_ = cs.CancelTask(ctx, &CancelTaskParams{TaskID: id})
	}
	answered := make(map[string]bool)
	for {
		switch task.Status {
		case TaskStatusCompleted:
			if task.Result == nil {
				return nil, fmt.Errorf("task %q completed without a result", id)
			}
			return task.Result, nil
		case TaskStatusFailed:
			if task.Error != nil {
				return nil, task.Error
			}
			return nil, fmt.Errorf("task %q failed: %s", id, task.StatusMessage)
		case TaskStatusCancelled:
			return nil, fmt.Errorf("task %q was cancelled: %s", id, task.StatusMessage)
		case TaskStatusInputRequired:
			requests := make(InputRequestMap)
			for key, request := range task.InputRequests {
				if !answered[key] {
					requests[key] = request
				}
			}
			if len(requests) > 0 {
				responses, err := fulfillInputRequests(ctx, cs, requests)
				if err == nil {
					err = cs.UpdateTask(ctx, &UpdateTaskParams{TaskID: id, InputResponses: responses})
				}
				if err != nil {
					cancel()
					return nil, err
				}
				for key := range requests {
					answered[key] = true
				}
			}
		}
		interval := time.Duration(task.PollIntervalMs) * time.Millisecond
		if interval <= 0 {
			interval = defaultTaskPollInterval
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			cancel()
			return nil, ctx.Err()
		case <-timer.C:
		}
		res, err := cs.GetTask(ctx, &GetTaskParams{TaskID: id})
		if ctx.Err() != nil {
			cancel()
			return nil, ctx.Err()
		}
		if err != nil {
			return nil, err
		}
		task = &res.Task
	}
}
