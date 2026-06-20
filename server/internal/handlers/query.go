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
	Question  string `json:"question"`
	TopK      int    `json:"top_k"`
	Rerank    bool   `json:"rerank"`
	SessionID string `json:"session_id"`
}

type pipelineQueryRequest struct {
	Messages any    `json:"messages"`
	TopK     int    `json:"top_k"`
	UserID   string `json:"user_id"`
}

// Query 转发检索+问答请求到云端 Pipeline 服务
func Query(c *gin.Context) {
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
		queryPayload.TopK = 6
	}
	if strings.TrimSpace(queryPayload.SessionID) == "" {
		queryPayload.SessionID = newSessionID()
	} else if err := chatStore.LoadUser(user.UserID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load chat history"})
		return
	}

	// 1. 从环境变量获取云端查询服务的 URL
	pipelineURL := config.GetEnv("PIPELINE_URL", "http://localhost:9000")
	queryURL := pipelineURL + "/query"

	// 2. 会话历史在业务后端完成隔离，Pipeline 只接收文本上下文
	pipelineBody, err := json.Marshal(pipelineQueryRequest{
		Messages: chatStore.PipelineMessages(user.UserID, queryPayload.SessionID, queryPayload.Question),
		TopK:     queryPayload.TopK,
		UserID:   user.UserID,
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
	client := &http.Client{}
	response, err := client.Do(request)
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

	responsePayload["session_id"] = queryPayload.SessionID
	responsePayload["history"] = chatStore.PipelineHistory(user.UserID, queryPayload.SessionID)

	c.JSON(http.StatusOK, responsePayload)
}

func newSessionID() string {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		return "session_" + strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return "session_" + hex.EncodeToString(bytes)
}
