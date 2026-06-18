package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"rag-server/internal/config"
)

type uploadJobState struct {
	Status    string         `json:"status"`
	Filename  string         `json:"filename,omitempty"`
	UserID    string         `json:"-"`
	Error     string         `json:"error,omitempty"`
	Result    map[string]any `json:"result,omitempty"`
	UpdatedAt time.Time      `json:"updated_at"`
}

var uploadJobs = struct {
	sync.Mutex
	state uploadJobState
}{
	state: uploadJobState{Status: "idle", UpdatedAt: time.Now()},
}

// Upload 处理 PDF 文件上传，并转发到云端 ingest 服务
func Upload(c *gin.Context) {
	user, ok := CurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

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

	if !startUploadJob(user.UserID, file.Filename) {
		c.JSON(http.StatusConflict, gin.H{"error": "another document is still being processed"})
		return
	}

	// 打开文件流
	opened, err := file.Open()
	if err != nil {
		failUploadJob("failed to open file")
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to open file"})
		return
	}
	defer opened.Close()

	content, err := io.ReadAll(opened)
	if err != nil {
		failUploadJob("failed to read file")
		c.JSON(http.StatusBadRequest, gin.H{"error": "failed to read file"})
		return
	}

	go runIngestJob(user.UserID, file.Filename, content)

	c.JSON(http.StatusAccepted, gin.H{
		"status":   "processing",
		"filename": file.Filename,
	})
}

func UploadStatus(c *gin.Context) {
	if _, ok := CurrentUser(c); !ok {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	uploadJobs.Lock()
	state := uploadJobs.state
	uploadJobs.Unlock()

	c.JSON(http.StatusOK, state)
}

func startUploadJob(userID, filename string) bool {
	uploadJobs.Lock()
	defer uploadJobs.Unlock()

	if uploadJobs.state.Status == "processing" {
		return false
	}

	uploadJobs.state = uploadJobState{
		Status:    "processing",
		Filename:  filename,
		UserID:    userID,
		UpdatedAt: time.Now(),
	}
	return true
}

func finishUploadJob(result map[string]any) {
	uploadJobs.Lock()
	defer uploadJobs.Unlock()

	uploadJobs.state.Status = "completed"
	uploadJobs.state.Error = ""
	uploadJobs.state.Result = result
	uploadJobs.state.UpdatedAt = time.Now()
}

func failUploadJob(message string) {
	uploadJobs.Lock()
	defer uploadJobs.Unlock()

	uploadJobs.state.Status = "failed"
	uploadJobs.state.Error = message
	uploadJobs.state.UpdatedAt = time.Now()
}

func runIngestJob(userID, filename string, content []byte) {
	// 云端 Pipeline ingest 接口地址
	pipelineURL := config.GetEnv("PIPELINE_URL", "http://localhost:9000")
	ingestURL := pipelineURL + "/ingest"

	// 云端调用超时设置
	defaultTimeout := config.GetEnv("PIPELINE_TIMEOUT", "60")
	timeoutSeconds, err := strconv.Atoi(defaultTimeout)
	if err != nil || timeoutSeconds <= 0 {
		timeoutSeconds = 120
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// 用 Pipe 组装 multipart 请求体，避免本地落盘
	reader, writer := io.Pipe()
	multipartWriter := multipart.NewWriter(writer)

	go func() {
		defer writer.Close()
		defer multipartWriter.Close()

		// 写入用户 ID 字段
		_ = multipartWriter.WriteField("user_id", userID)

		part, err := multipartWriter.CreateFormFile("file", filename) // 创建 multipart 文件字段，用于server端接收识别
		if err != nil {
			_ = writer.CloseWithError(err)
			return
		}

		if _, err := io.Copy(part, bytes.NewReader(content)); err != nil { // 写入文件内容，流式传输
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
		failUploadJob("failed to create ingest request")
		return
	}
	request.Header.Set("Content-Type", multipartWriter.FormDataContentType())

	// 可选鉴权头
	apiKey := config.GetEnv("PIPELINE_API_KEY", "")
	if apiKey != "" {
		header := config.GetEnv("PIPELINE_API_KEY_HEADER", "Authorization")
		prefix := config.GetEnv("PIPELINE_API_KEY_PREFIX", "Bearer ")
		request.Header.Set(header, prefix+apiKey)
	}

	client := &http.Client{}
	response, err := client.Do(request)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			failUploadJob("ingest timed out")
			return
		}
		failUploadJob("failed to call ingest service")
		return
	}
	defer response.Body.Close()

	// 读取云端返回
	body, err := io.ReadAll(response.Body)
	if err != nil {
		failUploadJob("failed to read ingest response")
		return
	}

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		failUploadJob("ingest failed: " + strings.TrimSpace(string(body)))
		return
	}

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		failUploadJob("invalid ingest response: " + strings.TrimSpace(string(body)))
		return
	}

	finishUploadJob(payload)
}
