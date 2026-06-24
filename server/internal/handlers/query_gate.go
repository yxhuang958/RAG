package handlers

import (
	"context"
	"net/http"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"

	"rag-server/internal/config"
)

type queryGate struct {
	workers     chan struct{}
	queue       chan struct{}
	waitTimeout time.Duration
	degradeAt   int64

	active   int64
	queued   int64
	rejected int64
	degraded int64
}

type queryAdmission struct {
	degraded bool
	release  func()
}

var queryLimiter = newQueryGate()

func newQueryGate() *queryGate {
	defaultWorkers := runtime.GOMAXPROCS(0) * 8
	if defaultWorkers < 32 {
		defaultWorkers = 32
	}

	maxWorkers := envInt("BACKEND_QUERY_MAX_CONCURRENCY", defaultWorkers)
	maxQueue := envInt("BACKEND_QUERY_QUEUE_SIZE", maxWorkers*4)
	waitMs := envInt("BACKEND_QUERY_QUEUE_TIMEOUT_MS", 250)
	degradeAt := int64(envInt("BACKEND_QUERY_DEGRADE_AT", maxWorkers))

	return &queryGate{
		workers:     make(chan struct{}, maxWorkers),
		queue:       make(chan struct{}, maxQueue),
		waitTimeout: time.Duration(waitMs) * time.Millisecond,
		degradeAt:   degradeAt,
	}
}

func (g *queryGate) acquire(ctx context.Context) (queryAdmission, int) {
	select {
	case g.queue <- struct{}{}:
		atomic.AddInt64(&g.queued, 1)
	default:
		atomic.AddInt64(&g.rejected, 1)
		return queryAdmission{}, http.StatusServiceUnavailable
	}

	degraded := g.shouldDegrade()
	timer := time.NewTimer(g.waitTimeout)
	defer timer.Stop()

	select {
	case g.workers <- struct{}{}:
		<-g.queue
		atomic.AddInt64(&g.queued, -1)
		active := atomic.AddInt64(&g.active, 1)
		if degraded || active >= g.degradeAt {
			degraded = true
			atomic.AddInt64(&g.degraded, 1)
		}
		return queryAdmission{
			degraded: degraded,
			release: func() {
				atomic.AddInt64(&g.active, -1)
				<-g.workers
			},
		}, http.StatusOK
	case <-timer.C:
		<-g.queue
		atomic.AddInt64(&g.queued, -1)
		atomic.AddInt64(&g.rejected, 1)
		return queryAdmission{}, http.StatusTooManyRequests
	case <-ctx.Done():
		<-g.queue
		atomic.AddInt64(&g.queued, -1)
		return queryAdmission{}, 499
	}
}

func (g *queryGate) shouldDegrade() bool {
	if g.degradeAt <= 0 {
		return false
	}
	return atomic.LoadInt64(&g.active)+atomic.LoadInt64(&g.queued) >= g.degradeAt
}

func (g *queryGate) snapshot() map[string]any {
	return map[string]any{
		"active":        atomic.LoadInt64(&g.active),
		"queued":        atomic.LoadInt64(&g.queued),
		"rejected":      atomic.LoadInt64(&g.rejected),
		"degraded":      atomic.LoadInt64(&g.degraded),
		"max_active":    cap(g.workers),
		"max_queue":     cap(g.queue),
		"degrade_at":    g.degradeAt,
		"queue_timeout": g.waitTimeout.String(),
	}
}

func envFloat(key string, fallback float64) float64 {
	value, err := strconv.ParseFloat(config.GetEnv(key, ""), 64)
	if err != nil || value < 0 {
		return fallback
	}
	return value
}
