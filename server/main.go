package main

import (
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/joho/godotenv"
)

func main() {
    _ = godotenv.Load("../.env")

    router := gin.Default()
    router.Use(corsMiddleware())

    router.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{"status": "ok"})
    })

    router.POST("/api/upload", handleUpload)

    uploadDir := getEnv("UPLOAD_DIR", "uploads")
    if err := os.MkdirAll(uploadDir, 0o755); err != nil {
        panic(err)
    }

    port := getEnv("SERVER_PORT", "8080")
    if err := router.Run(":" + port); err != nil {
        panic(err)
    }
}

func handleUpload(c *gin.Context) {
    file, err := c.FormFile("file")
    if err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "missing file"})
        return
    }

    if !strings.HasSuffix(strings.ToLower(file.Filename), ".pdf") {
        c.JSON(http.StatusBadRequest, gin.H{"error": "only PDF files are supported"})
        return
    }

    uploadDir := getEnv("UPLOAD_DIR", "uploads")
    filename := time.Now().Format("20060102-150405") + "-" + filepath.Base(file.Filename)
    savePath := filepath.Join(uploadDir, filename)

    if err := c.SaveUploadedFile(file, savePath); err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to save file"})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "message":  "uploaded",
        "filename": filename,
        "path":     savePath,
    })
}

func corsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
        c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")

        if c.Request.Method == http.MethodOptions {
            c.AbortWithStatus(http.StatusNoContent)
            return
        }

        c.Next()
    }
}

func getEnv(key, fallback string) string {
    if value := strings.TrimSpace(os.Getenv(key)); value != "" {
        return value
    }
    return fallback
}
