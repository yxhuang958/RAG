package config

import (
	"os"
	"strings"

	"github.com/joho/godotenv"
)

func init() {
	_ = godotenv.Load("../.env", ".env")
}

// GetEnv 获取环境变量，如果不存在则返回默认值
func GetEnv(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
