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

	router.POST("/api/login", handlers.Login)

	api := router.Group("/api")
	api.Use(middleware.AuthRequired())
	{
		// 文件上传
		api.POST("/upload", handlers.Upload)
		api.GET("/upload/status", handlers.UploadStatus)

		// 问答检索
		api.POST("/query", handlers.Query)

		// 会话历史
		api.GET("/sessions", handlers.Sessions)
	}

	return router
}
