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
	"os"

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

	// -- Save debug: success 20260531
	// os.MkdirAll("uploads", 0755)
	// out, err := os.Create("uploads/" + file.Filename)
	// if err == nil {
	// 	defer out.Close()
	// 	io.Copy(out, opened)
	// 	out.Sync()
	// }
	// return
	// -- End of save debug

	/*
		[ opened (原始PDF) ] 
			|
			v  (io.Copy)
		[ part (Writer句柄) ]  <-- 标记它是 "file" 字段
			|
			v
		[ multipart.Writer (打包机) ]
			|
			v
		[ pipe.writer (传送带入口) ]  <==== [ 内存管道 ] ====> [ pipe.reader (传送带出口) ]
																		|
																		v
															[ http.Request (发送器) ]
		
		multipart.Writer 确实把所有的 Multipart 数据协议格式（包括分隔符 Boundary、
			字段名 name="file"等）全部写进了 writer 也就是流向了 reader。
		这导致 reader 吐出来的字节流，本身就是一个完美的、合乎规范的 HTTP POST 请求体 (Request Body)。
		但是这个请求体是没有 HTTP Headers 的，Headers 需要我们在 http.Request 里手动设置，
			特别是 Content-Type，要和 multipart.Writer 的 FormDataContentType() 保持一致。
	*/

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
