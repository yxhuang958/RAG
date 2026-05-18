package handlers

import (
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"rag-server/internal/config"
)

// Upload 处理 PDF 文件上传，并转发到云端 CANN ingest 服务
func Upload(c *gin.Context) {
	// 读取 multipart 表单中的文件
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "missing file"})
		return
	}

	// 简单校验：仅允许 PDF
	if !strings.HasSuffix(strings.ToLower(file.Filename), ".pdf") {
		c.JSON(http.StatusBadRequest, gin.H{"error": "only PDF files are supported"})
		return
	}

	// 打开文件流
	opened, err := file.Open()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to open file"})
		return
	}
	defer opened.Close()

	// 云端 CANN ingest 接口地址
	ingestURL := config.GetEnv("CANN_INGEST_URL", "")
	if ingestURL == "" {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "CANN_INGEST_URL not configured"})
		return
	}

	// 云端调用超时设置
	defaultTimeout := config.GetEnv("CANN_INGEST_TIMEOUT_SECONDS", "120")
	timeoutSeconds, err := strconv.Atoi(defaultTimeout)
	if err != nil || timeoutSeconds <= 0 {
		timeoutSeconds = 120
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// 用 Pipe 组装 multipart 请求体，避免本地落盘
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)

	go func() {
		defer writer.Close()
		defer multipartWriter.Close()

		part, err := multipartWriter.CreateFormFile("file", file.Filename) // 创建 multipart 文件字段，用于server端接收识别
		if err != nil {
			_ = writer.CloseWithError(err)
			return
		}

		if _, err := io.Copy(part, opened); err != nil { // 写入文件内容，流式传输
			_ = writer.CloseWithError(err)
			return
		}
	}()

	// 构建转发请求
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, ingestURL, reader)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create ingest request"})
		return
	}
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())

	// 可选鉴权头
	apiKey := config.GetEnv("CANN_API_KEY", "")
	if apiKey != "" {
		header := config.GetEnv("CANN_API_KEY_HEADER", "Authorization")
		prefix := config.GetEnv("CANN_API_KEY_PREFIX", "Bearer ")
		request.Header.Set(header, prefix+apiKey)
	}

	client := &http.Client{}
	response, err := client.Do(request)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			c.JSON(http.StatusGatewayTimeout, gin.H{"error": "ingest timed out"})
			return
		}
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to call ingest service"})
		return
	}
	defer response.Body.Close()

	// 读取云端返回
	body, err := io.ReadAll(response.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to read ingest response"})
		return
	}

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "ingest failed",
			"details": strings.TrimSpace(string(body)),
		})
		return
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "invalid ingest response",
			"details": strings.TrimSpace(string(body)),
		})
		return
	}

	c.JSON(http.StatusOK, payload)  // 直接将云端返回的 JSON payload 转发给前端, const result = await response.json();
}
