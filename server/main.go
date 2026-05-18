package main

// go run main.go

import (
	"github.com/joho/godotenv"
	"rag-server/internal/config"
	"rag-server/internal/router"
)

func main() {
	_ = godotenv.Load("../.env")

	app := router.Setup()

	port := config.GetEnv("SERVER_PORT", "8080")
	if err := app.Run(":" + port); err != nil {
		panic(err)
	}
}
