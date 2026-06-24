package handlers

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"rag-server/internal/config"
)

type queryRequest struct {
	Question     string `json:"question"`
	TopK         int    `json:"top_k"`
	Rerank       bool   `json:"rerank"`
	SessionID    string `json:"session_id"`
	PipelineMode string `json:"pipeline_mode"`
}

type pipelineQueryRequest struct {
	Messages   any    `json:"messages"`
	TopK       int    `json:"top_k"`
	UserID     string `json:"user_id"`
	Rerank     bool   `json:"rerank"`
	SecondSort bool   `json:"second_sort"`
}

type pipelineTextIngestRequest struct {
	UserID   string `json:"user_id"`
	Text     string `json:"text"`
	Filename string `json:"filename"`
	DocID    string `json:"doc_id"`
}

var pipelineHTTPClient = &http.Client{}

// Query 转发检索+问答请求到云端 Pipeline 服务
func Query(c *gin.Context) {
	admission, status := queryLimiter.acquire(c.Request.Context())
	if status != http.StatusOK {
		if status == http.StatusTooManyRequests {
			c.JSON(status, gin.H{"error": "query queue timeout", "load": queryLimiter.snapshot()})
			return
		}
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "query queue full", "load": queryLimiter.snapshot()})
		return
	}
	defer admission.release()

	user, ok := CurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	var queryPayload queryRequest
	if err := c.ShouldBindJSON(&queryPayload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid query payload"})
		return
	}
	queryPayload.Question = strings.TrimSpace(queryPayload.Question)
	if queryPayload.Question == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "question is required"})
		return
	}
	if queryPayload.TopK <= 0 {
		queryPayload.TopK, _ = strconv.Atoi(config.GetEnv("RETRIEVE_TOPK", "30"))
	}
	if admission.degraded {
		degradedTopK := envInt("BACKEND_QUERY_DEGRADED_TOPK", 10)
		if degradedTopK > 0 && queryPayload.TopK > degradedTopK {
			queryPayload.TopK = degradedTopK
		}
		queryPayload.Rerank = false
		c.Header("X-Backend-Degraded", "true")
	}
	if strings.TrimSpace(queryPayload.SessionID) == "" {
		queryPayload.SessionID = newSessionID()
	} else if err := chatStore.LoadUser(user.UserID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load chat history"})
		return
	}

	// 1. 从环境变量获取云端查询服务的 URL
	pipelineURL := selectPipelineURL(queryPayload.PipelineMode)
	queryURL := pipelineURL + "/query"

	// 2. 会话历史在业务后端完成隔离，Pipeline 只接收文本上下文
	recentMessages := envInt("SESSION_RECENT_MESSAGES", 8)
	if admission.degraded {
		recentMessages = envInt("BACKEND_QUERY_DEGRADED_RECENT_MESSAGES", 2)
	}
	pipelineBody, err := json.Marshal(pipelineQueryRequest{
		Messages: chatStore.PipelineMessages(
			user.UserID,
			queryPayload.SessionID,
			queryPayload.Question,
			recentMessages,
		),
		TopK:       queryPayload.TopK,
		UserID:     user.UserID,
		Rerank:     queryPayload.Rerank,
		SecondSort: !admission.degraded,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to build query payload"})
		return
	}

	// 3. 配置超时时间，防止云端服务响应过慢导致阻塞
	defaultTimeout := config.GetEnv("PIPELINE_TIMEOUT", "60")
	timeoutSeconds, err := strconv.Atoi(defaultTimeout)
	if err != nil || timeoutSeconds <= 0 {
		timeoutSeconds = 60
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// 4. 构造转发给云端服务的 POST 请求
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, queryURL, bytes.NewReader(pipelineBody))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create query request"})
		return
	}
	request.Header.Set("Content-Type", "application/json")

	// 5. 如果配置了 API Key，则添加到请求头进行鉴权
	apiKey := config.GetEnv("PIPELINE_API_KEY", "")
	if apiKey != "" {
		header := config.GetEnv("PIPELINE_API_KEY_HEADER", "Authorization")
		prefix := config.GetEnv("PIPELINE_API_KEY_PREFIX", "Bearer ")
		request.Header.Set(header, prefix+apiKey)
	}

	// 6. 发起 HTTP 请求
	response, err := pipelineHTTPClient.Do(request)
	if err != nil {
		// 处理超时或其他连接错误
		if ctx.Err() == context.DeadlineExceeded {
			c.JSON(http.StatusGatewayTimeout, gin.H{"error": "query timed out"})
			return
		}
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to call query service"})
		return
	}
	defer response.Body.Close()

	// 7. 读取云端服务返回的响应内容
	respBody, err := io.ReadAll(response.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read query response"})
		return
	}

	// 如果云端返回状态码不是 2xx，透传错误信息
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "query failed",
			"details": strings.TrimSpace(string(respBody)),
		})
		return
	}

	// 8. 将响应结果解析为 JSON 并返回给前端
	var responsePayload map[string]any
	if err := json.Unmarshal(respBody, &responsePayload); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "invalid query response",
			"details": strings.TrimSpace(string(respBody)),
		})
		return
	}

	answer, _ := responsePayload["answer"].(string)
	if _, err := chatStore.AppendExchange(user.UserID, queryPayload.SessionID, queryPayload.Question, answer); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save chat history"})
		return
	}
	session, compacted, err := chatStore.CompactSession(user.UserID, queryPayload.SessionID, envInt("SESSION_COMPACT_KEEP_MESSAGES", 12))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to compact chat history"})
		return
	}
	if compacted {
		go ingestSessionMemory(user.UserID, session.ID, session.Summary)
	}

	responsePayload["session_id"] = queryPayload.SessionID
	responsePayload["history"] = chatStore.PipelineDisplayHistory(user.UserID, queryPayload.SessionID)
	responsePayload["session_summary"] = chatStore.SessionSummary(user.UserID, queryPayload.SessionID)
	if admission.degraded {
		responsePayload["degraded"] = true
	}

	c.JSON(http.StatusOK, responsePayload)
}

func ingestSessionMemory(userID, sessionID, summary string) {
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return
	}

	pipelineURL := config.GetEnv("PIPELINE_URL", "http://localhost:9000")
	ingestURL := pipelineURL + "/ingest_text"
	body, err := json.Marshal(pipelineTextIngestRequest{
		UserID:   userID,
		Text:     "Long-running conversation memory for session " + sessionID + ":\n" + summary,
		Filename: "session_memory_" + sessionID + ".txt",
		DocID:    userID + "-session-memory-" + sessionID,
	})
	if err != nil {
		return
	}

	timeoutSeconds := envInt("SESSION_MEMORY_INGEST_TIMEOUT", 60)
	request, err := http.NewRequest(http.MethodPost, ingestURL, bytes.NewReader(body))
	if err != nil {
		return
	}
	request.Header.Set("Content-Type", "application/json")

	apiKey := config.GetEnv("PIPELINE_API_KEY", "")
	if apiKey != "" {
		header := config.GetEnv("PIPELINE_API_KEY_HEADER", "Authorization")
		prefix := config.GetEnv("PIPELINE_API_KEY_PREFIX", "Bearer ")
		request.Header.Set(header, prefix+apiKey)
	}

	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	response, err := client.Do(request)
	if err != nil {
		return
	}
	defer response.Body.Close()
	io.Copy(io.Discard, response.Body)
}

func selectPipelineURL(mode string) string {
	mode = strings.TrimSpace(strings.ToLower(mode))
	if mode == "mock" && envBool("ENABLE_MOCK_PIPELINE", false) {
		return config.GetEnv("MOCK_PIPELINE_URL", "http://127.0.0.1:9100")
	}
	return config.GetEnv("PIPELINE_URL", "http://localhost:9000")
}

func envInt(key string, fallback int) int {
	value, err := strconv.Atoi(config.GetEnv(key, strconv.Itoa(fallback)))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func envBool(key string, fallback bool) bool {
	value := strings.TrimSpace(strings.ToLower(config.GetEnv(key, "")))
	if value == "" {
		return fallback
	}
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func newSessionID() string {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		return "session_" + strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return "session_" + hex.EncodeToString(bytes)
}
