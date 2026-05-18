package config

import (
	"os"
	"strings"
)

// GetEnv 获取环境变量，如果不存在则返回默认值
func GetEnv(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
