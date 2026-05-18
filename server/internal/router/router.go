package router

import (
	"github.com/gin-gonic/gin"
	"rag-server/internal/handlers"
	"rag-server/internal/middleware"
)

// Setup 设置路由
func Setup() *gin.Engine {
	router := gin.Default()

	// 应用中间件
	router.Use(middleware.CORS())

	// 健康检查
	router.GET("/health", handlers.Health)

	// 文件上传
	router.POST("/api/upload", handlers.Upload)

	// 问答检索
	router.POST("/api/query", handlers.Query)

	return router
}
