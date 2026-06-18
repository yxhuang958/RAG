package handlers

import (
	"net/http"
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
	token, err := auth.Sign(userID, username, jwtSecret(), 24*time.Hour)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to create token"})
		return
	}

	go func() {
		_ = chatStore.LoadUser(userID)
	}()

	c.JSON(http.StatusOK, gin.H{
		"token":    token,
		"user_id":  userID,
		"username": username,
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
