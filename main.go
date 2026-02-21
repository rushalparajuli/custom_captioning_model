package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	pythonServiceURL = "http://localhost:8000/caption"
	maxRequestSize   = 10 << 20 // 10MB
	requestTimeout   = 30 * time.Second
)

type ImageRequest struct {
	ImageBase64 string `json:"image_base64" binding:"required"`
}

type CaptionResponse struct {
	Caption string `json:"caption"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

// Custom CORS middleware
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "http://localhost:3000")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func main() {
	r := gin.Default()

	// Use custom CORS middleware
	r.Use(CORSMiddleware())

	// Limit request body size
	r.Use(func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxRequestSize)
		c.Next()
	})

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "healthy"})
	})

	r.POST("/process", func(c *gin.Context) {
		var req ImageRequest
		
		// Bind and validate JSON
		if err := c.ShouldBindJSON(&req); err != nil {
			log.Printf("Invalid request: %v", err)
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error: "Invalid request format",
			})
			return
		}

		// Basic validation
		if len(req.ImageBase64) == 0 {
			c.JSON(http.StatusBadRequest, ErrorResponse{
				Error: "image_base64 is required",
			})
			return
		}

		log.Printf("Processing image request (size: %d bytes)", len(req.ImageBase64))

		// Call Python FastAPI service with timeout
		caption, err := getCaptionFromPython(req)
		if err != nil {
			log.Printf("Caption service error: %v", err)
			c.JSON(http.StatusInternalServerError, ErrorResponse{
				Error: "Failed to generate caption",
			})
			return
		}

		// Return caption
		if caption == "" {
			caption = "No caption generated"
		}
		
		log.Printf("Successfully generated caption: %s", caption)
		c.JSON(http.StatusOK, gin.H{"caption": caption})
	})

	log.Println("Starting server on :8080")
	if err := r.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func getCaptionFromPython(req ImageRequest) (string, error) {
	// Create request body using proper JSON marshaling
	requestBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(
		ctx,
		"POST",
		pythonServiceURL,
		bytes.NewBuffer(requestBody),
	)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	client := &http.Client{
		Timeout: requestTimeout,
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("python service returned %d: %s", resp.StatusCode, string(body))
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	// Parse response
	var captionResp CaptionResponse
	if err := json.Unmarshal(body, &captionResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %w", err)
	}

	return captionResp.Caption, nil
}
