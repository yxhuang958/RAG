package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"rag-server/internal/auth"
	"rag-server/internal/chat"
	"rag-server/internal/config"
)

var chatStore = chat.NewStore(config.GetEnv("CHAT_STORE_PATH", "data/chat_sessions.json"))

type loginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type userContext struct {
	UserID   string
	Username string
}

func Login(c *gin.Context) {
	var payload loginRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid login payload"})
		return
	}

	username := strings.TrimSpace(payload.Username)
	if username == "" || strings.TrimSpace(payload.Password) == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
		return
	}

	userID := "user_" + username
	isNewUser, err := chatStore.AuthenticateOrRegister(userID, username, payload.Password)
	if err != nil {
		if errors.Is(err, chat.ErrInvalidCredentials) {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "invalid username or password"})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to authenticate user"})
		return
	}
	token, err := auth.Sign(userID, username, jwtSecret(), 24*time.Hour)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create token"})
		return
	}

	go func() {
		_ = chatStore.LoadUser(userID)
		warmupPipeline(userID)
	}()

	c.JSON(http.StatusOK, gin.H{
		"token":    token,
		"user_id":  userID,
		"username": username,
		"is_new":   isNewUser,
		"sessions": chatStore.ListSessions(userID),
	})
}

func Sessions(c *gin.Context) {
	user, ok := CurrentUser(c)
	if !ok {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		return
	}

	if err := chatStore.LoadUser(user.UserID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to load sessions"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"sessions": chatStore.ListSessions(user.UserID)})
}

func CurrentUser(c *gin.Context) (userContext, bool) {
	userID, ok := c.Get("user_id")
	if !ok {
		return userContext{}, false
	}
	username, _ := c.Get("username")
	userIDValue, ok := userID.(string)
	if !ok || userIDValue == "" {
		return userContext{}, false
	}

	usernameValue, _ := username.(string)
	return userContext{UserID: userIDValue, Username: usernameValue}, true
}

func jwtSecret() string {
	return config.GetEnv("JWT_SECRET", "dev-secret-change-me")
}

func warmupPipeline(userID string) {
	pipelineURL := config.GetEnv("PIPELINE_URL", "http://localhost:9000")
	warmupURL := pipelineURL + "/warmup"

	defaultTimeout := config.GetEnv("PIPELINE_WARMUP_TIMEOUT", "30")
	timeoutSeconds, err := strconv.Atoi(defaultTimeout)
	if err != nil || timeoutSeconds <= 0 {
		timeoutSeconds = 30
	}

	body, err := json.Marshal(gin.H{"user_id": userID})
	if err != nil {
		return
	}

	request, err := http.NewRequest(http.MethodPost, warmupURL, bytes.NewReader(body))
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
}
