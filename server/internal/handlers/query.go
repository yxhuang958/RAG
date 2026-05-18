package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"rag-server/internal/config"
)

// Query 转发检索+问答请求到云端 CANN 服务
func Query(c *gin.Context) {
	// 1. 从环境变量获取云端查询服务的 URL
	queryURL := config.GetEnv("CANN_QUERY_URL", "")
	if queryURL == "" {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "CANN_QUERY_URL not configured"})
		return
	}

	// 2. 读取前端发送过来的请求体 (包含问题 question 等)
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read request"})
		return
	}

	// 校验请求体是否为空
	if len(strings.TrimSpace(string(body))) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "empty request body"})
		return
	}

	// 3. 配置超时时间，防止云端服务响应过慢导致阻塞
	defaultTimeout := config.GetEnv("CANN_QUERY_TIMEOUT_SECONDS", "60")
	timeoutSeconds, err := strconv.Atoi(defaultTimeout)
	if err != nil || timeoutSeconds <= 0 {
		timeoutSeconds = 60
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// 4. 构造转发给云端服务的 POST 请求
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, queryURL, strings.NewReader(string(body)))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create query request"})
		return
	}
	request.Header.Set("Content-Type", "application/json")

	// 5. 如果配置了 API Key，则添加到请求头进行鉴权
	apiKey := config.GetEnv("CANN_API_KEY", "")
	if apiKey != "" {
		header := config.GetEnv("CANN_API_KEY_HEADER", "Authorization")
		prefix := config.GetEnv("CANN_API_KEY_PREFIX", "Bearer ")
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
	var payload map[string]any
	if err := json.Unmarshal(respBody, &payload); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "invalid query response",
			"details": strings.TrimSpace(string(respBody)),
		})
		return
	}

	c.JSON(http.StatusOK, payload)
}
