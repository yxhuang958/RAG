package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"rag-server/internal/auth"
	"rag-server/internal/config"
)

func AuthRequired() gin.HandlerFunc {
	return func(c *gin.Context) {
		header := strings.TrimSpace(c.GetHeader("Authorization"))
		if !strings.HasPrefix(header, "Bearer ") {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "missing bearer token"})
			return
		}

		claims, err := auth.Parse(strings.TrimSpace(strings.TrimPrefix(header, "Bearer ")), jwtSecret())
		if err != nil {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "invalid token"})
			return
		}

		c.Set("user_id", claims.UserID)
		c.Set("username", claims.Username)
		c.Next()
	}
}

func jwtSecret() string {
	return config.GetEnv("JWT_SECRET", "dev-secret-change-me")
}
