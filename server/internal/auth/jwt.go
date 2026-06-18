package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

var base64URL = base64.RawURLEncoding

type Claims struct {
	UserID   string `json:"sub"`
	Username string `json:"username"`
	Expires  int64  `json:"exp"`
	IssuedAt int64  `json:"iat"`
}

type header struct {
	Algorithm string `json:"alg"`
	Type      string `json:"typ"`
}

func Sign(userID, username, secret string, ttl time.Duration) (string, error) {
	now := time.Now()
	claims := Claims{
		UserID:   userID,
		Username: username,
		IssuedAt: now.Unix(),
		Expires:  now.Add(ttl).Unix(),
	}

	headerBytes, err := json.Marshal(header{Algorithm: "HS256", Type: "JWT"})
	if err != nil {
		return "", err
	}
	claimsBytes, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}

	unsigned := base64URL.EncodeToString(headerBytes) + "." + base64URL.EncodeToString(claimsBytes)
	signature := sign(unsigned, secret)
	return unsigned + "." + signature, nil
}

func Parse(token, secret string) (Claims, error) {
	var claims Claims
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return claims, errors.New("invalid token")
	}

	unsigned := parts[0] + "." + parts[1]
	if !hmac.Equal([]byte(sign(unsigned, secret)), []byte(parts[2])) {
		return claims, errors.New("invalid token signature")
	}

	claimsBytes, err := base64URL.DecodeString(parts[1])
	if err != nil {
		return claims, err
	}
	if err := json.Unmarshal(claimsBytes, &claims); err != nil {
		return claims, err
	}
	if claims.UserID == "" || claims.Expires <= time.Now().Unix() {
		return claims, errors.New("expired token")
	}

	return claims, nil
}

func sign(unsigned, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(unsigned))
	return base64URL.EncodeToString(mac.Sum(nil))
}
