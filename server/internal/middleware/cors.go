package middleware

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// CORS 中间件处理跨域请求
// 返回的这个函数会在 每个请求 到达 handler 之前执行
func CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*") // 通配符（所有源都可以）
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent) // 204 No Content, 表示预检请求成功但没有内容返回
			return
		}

		c.Next()
	}
}
