package chat

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"go.mongodb.org/mongo-driver/mongo/readpref"
	"golang.org/x/crypto/bcrypt"
)

var ErrInvalidCredentials = errors.New("invalid username or password")

type Message struct {
	Role      string `bson:"role" json:"role"`
	Content   string `bson:"content" json:"content"`
	CreatedAt int64  `bson:"created_at" json:"created_at"`
}

type Session struct {
	ID                 string    `bson:"session_id" json:"session_id"`
	UserID             string    `bson:"user_id" json:"user_id"`
	Messages           []Message `bson:"messages" json:"messages"`
	Summary            string    `bson:"summary,omitempty" json:"summary,omitempty"`
	SummarizedMessages int       `bson:"summarized_messages,omitempty" json:"summarized_messages,omitempty"`
	UpdatedAt          int64     `bson:"updated_at" json:"updated_at"`
}

type User struct {
	ID           string `bson:"user_id" json:"user_id"`
	Username     string `bson:"username" json:"username"`
	PasswordHash string `bson:"password_hash" json:"-"`
	CreatedAt    int64  `bson:"created_at" json:"created_at"`
	UpdatedAt    int64  `bson:"updated_at" json:"updated_at"`
	LastLogin    int64  `bson:"last_login" json:"last_login"`
}

type PipelineMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Store struct {
	path       string
	backend    string
	mu         sync.RWMutex
	sessions   map[string]Session
	users      map[string]User
	syncWrites bool
	disabled   bool
	persistCh  chan map[string]Session
	userCh     chan User
	mongo      *mongoStore

	errMu          sync.RWMutex
	lastPersistErr error
}

func NewStore(path string) *Store {
	store := &Store{
		path:       path,
		backend:    strings.TrimSpace(strings.ToLower(os.Getenv("CHAT_STORE_BACKEND"))),
		sessions:   make(map[string]Session),
		users:      make(map[string]User),
		syncWrites: envBool("CHAT_STORE_SYNC_WRITES", false),
		disabled:   envBool("CHAT_STORE_DISABLE_PERSISTENCE", false),
		persistCh:  make(chan map[string]Session, envPositiveInt("CHAT_STORE_PERSIST_QUEUE_SIZE", 8192)),
		userCh:     make(chan User, envPositiveInt("CHAT_STORE_USER_QUEUE_SIZE", 1024)),
	}

	if store.backend == "mongodb" {
		mongoStore, err := newMongoStore()
		if err != nil {
			log.Printf("chat store: mongodb unavailable, falling back to local memory/json: %v", err)
			store.setLastError(err)
		} else {
			store.mongo = mongoStore
		}
	}

	if !store.syncWrites && !store.disabled {
		go store.persistLoop()
	}
	if store.mongo != nil && !store.disabled {
		go store.userPersistLoop()
	}
	return store
}

func (s *Store) LoadUser(userID string) error {
	if s.disabled {
		return nil
	}
	if s.mongo != nil {
		sessions, err := s.mongo.loadUserSessions(userID)
		if err != nil {
			return err
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		for id, persisted := range sessions {
			current, exists := s.sessions[id]
			// Persistence is asynchronous. A sessions request can therefore read
			// an older MongoDB snapshot immediately after AppendExchange. Never
			// let that stale snapshot replace newer in-memory conversation state.
			if !exists || persisted.UpdatedAt > current.UpdatedAt {
				s.sessions[id] = persisted
			}
		}
		return nil
	}
	all, err := s.readAll()
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	for id, session := range all {
		if session.UserID == userID {
			s.sessions[id] = session
		}
	}
	return nil
}

func (s *Store) AuthenticateOrRegister(userID, username, password string) (bool, error) {
	userID = strings.TrimSpace(userID)
	username = strings.TrimSpace(username)
	password = strings.TrimSpace(password)
	if userID == "" || username == "" || password == "" {
		return false, ErrInvalidCredentials
	}

	if s.mongo != nil && !s.disabled {
		return s.mongo.authenticateOrRegister(userID, username, password)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if user, ok := s.users[userID]; ok {
		if bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password)) != nil {
			return false, ErrInvalidCredentials
		}
		now := time.Now().Unix()
		user.UpdatedAt = now
		user.LastLogin = now
		s.users[userID] = user
		return false, nil
	}

	passwordHash, err := hashPassword(password)
	if err != nil {
		return false, err
	}
	now := time.Now().Unix()
	s.users[userID] = User{
		ID:           userID,
		Username:     username,
		PasswordHash: passwordHash,
		CreatedAt:    now,
		UpdatedAt:    now,
		LastLogin:    now,
	}
	return true, nil
}

func (s *Store) GetHistory(userID, sessionID string) ([]Message, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	session, ok := s.sessions[sessionID]
	if !ok || session.UserID != userID {
		return nil, false
	}
	return append([]Message(nil), session.Messages...), true
}

func (s *Store) ListSessions(userID string) []Session {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sessions := make([]Session, 0)
	for _, session := range s.sessions {
		if session.UserID == userID {
			copySession := session
			copySession.Messages = append([]Message(nil), session.Messages...)
			sessions = append(sessions, copySession)
		}
	}
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt > sessions[j].UpdatedAt
	})
	return sessions
}

func (s *Store) AppendExchange(userID, sessionID, question, answer string) (Session, error) {
	now := time.Now().Unix()

	s.mu.Lock()
	session := s.sessions[sessionID]
	if session.ID == "" {
		session = Session{ID: sessionID, UserID: userID}
	}
	if session.UserID != userID {
		s.mu.Unlock()
		return Session{}, errors.New("session belongs to another user")
	}
	session.Messages = append(session.Messages,
		Message{Role: "user", Content: question, CreatedAt: now},
		Message{Role: "assistant", Content: answer, CreatedAt: now},
	)
	session.UpdatedAt = now
	s.sessions[sessionID] = session
	s.mu.Unlock()

	if err := s.persist(map[string]Session{sessionID: session}); err != nil {
		return Session{}, err
	}
	return session, nil
}

func (s *Store) CompactSession(userID, sessionID string, keepMessages int) (Session, bool, error) {
	if keepMessages < 2 {
		keepMessages = 2
	}

	now := time.Now().Unix()
	s.mu.Lock()
	session, ok := s.sessions[sessionID]
	if !ok || session.UserID != userID {
		s.mu.Unlock()
		return Session{}, false, errors.New("session not found")
	}
	if len(session.Messages) <= keepMessages {
		s.mu.Unlock()
		return session, false, nil
	}

	cutoff := len(session.Messages) - keepMessages
	toSummarize := session.Messages[:cutoff]
	session.Summary = mergeSummary(session.Summary, toSummarize)
	session.SummarizedMessages += len(toSummarize)
	session.Messages = append([]Message(nil), session.Messages[cutoff:]...)
	session.UpdatedAt = now
	s.sessions[sessionID] = session
	s.mu.Unlock()

	if err := s.persist(map[string]Session{sessionID: session}); err != nil {
		return Session{}, false, err
	}
	return session, true, nil
}

func (s *Store) PipelineHistory(userID, sessionID string, recentMessages int) []PipelineMessage {
	history, ok := s.GetHistory(userID, sessionID)
	if !ok {
		return []PipelineMessage{}
	}
	summary := s.SessionSummary(userID, sessionID)
	if recentMessages > 0 && len(history) > recentMessages {
		history = history[len(history)-recentMessages:]
	}

	messages := make([]PipelineMessage, 0, len(history)+1)
	if summary != "" {
		messages = append(messages, PipelineMessage{
			Role:    "system",
			Content: "Conversation memory summary:\n" + summary,
		})
	}
	for _, item := range history {
		messages = append(messages, PipelineMessage{Role: item.Role, Content: item.Content})
	}
	return messages
}

func (s *Store) PipelineMessages(userID, sessionID, question string, recentMessages int) []PipelineMessage {
	messages := s.PipelineHistory(userID, sessionID, recentMessages)
	return append(messages, PipelineMessage{Role: "user", Content: question})
}

func (s *Store) PipelineDisplayHistory(userID, sessionID string) []PipelineMessage {
	history, ok := s.GetHistory(userID, sessionID)
	if !ok {
		return []PipelineMessage{}
	}

	messages := make([]PipelineMessage, 0, len(history))
	for _, item := range history {
		messages = append(messages, PipelineMessage{Role: item.Role, Content: item.Content})
	}
	return messages
}

func (s *Store) SessionSummary(userID, sessionID string) string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	session, ok := s.sessions[sessionID]
	if !ok || session.UserID != userID {
		return ""
	}
	return session.Summary
}

func (s *Store) readAll() (map[string]Session, error) {
	content, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]Session{}, nil
		}
		return nil, err
	}
	if len(content) == 0 {
		return map[string]Session{}, nil
	}

	var sessions map[string]Session
	if err := json.Unmarshal(content, &sessions); err != nil {
		return nil, err
	}
	return sessions, nil
}

func (s *Store) writeSessions(sessions map[string]Session) error {
	if s.mongo != nil {
		return s.mongo.upsertSessions(sessions)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0755); err != nil {
		return err
	}

	all, err := s.readAll()
	if err != nil {
		return err
	}
	for id, session := range sessions {
		all[id] = session
	}

	content, err := json.Marshal(all)
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, content, 0644)
}

func (s *Store) userPersistLoop() {
	for user := range s.userCh {
		s.setLastError(s.mongo.upsertUser(user))
	}
}

func (s *Store) persist(sessions map[string]Session) error {
	if s.disabled {
		return nil
	}
	if s.syncWrites {
		return s.writeSessions(sessions)
	}

	select {
	case s.persistCh <- sessions:
	default:
		select {
		case <-s.persistCh:
		default:
		}
		select {
		case s.persistCh <- sessions:
		default:
			// Keep request latency bounded under extreme bursts. The in-memory
			// session is already current; a later persist event will flush it.
		}
	}
	return nil
}

func (s *Store) persistLoop() {
	var pending map[string]Session
	timer := time.NewTimer(time.Hour)
	if !timer.Stop() {
		<-timer.C
	}

	for {
		if pending == nil {
			pending = <-s.persistCh
			timer.Reset(persistInterval())
			continue
		}

		select {
		case next := <-s.persistCh:
			for id, session := range next {
				pending[id] = session
			}
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(persistInterval())
		case <-timer.C:
			s.setLastError(s.writeSessions(pending))
			pending = nil
		}
	}
}

func (s *Store) lastError() error {
	s.errMu.RLock()
	defer s.errMu.RUnlock()
	return s.lastPersistErr
}

func (s *Store) setLastError(err error) {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	s.lastPersistErr = err
}

func persistInterval() time.Duration {
	value, err := time.ParseDuration(os.Getenv("CHAT_STORE_ASYNC_FLUSH_INTERVAL"))
	if err != nil || value <= 0 {
		return 250 * time.Millisecond
	}
	return value
}

func envBool(key string, fallback bool) bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if value == "" {
		return fallback
	}
	return value == "1" || value == "true" || value == "yes" || value == "on"
}

func envPositiveInt(key string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(key)))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

type mongoStore struct {
	client     *mongo.Client
	users      *mongo.Collection
	sessions   *mongo.Collection
	timeout    time.Duration
	retries    int
	retryDelay time.Duration
}

func newMongoStore() (*mongoStore, error) {
	uri := firstEnv("MONGODB_URI", "MONGO_URI")
	if strings.TrimSpace(uri) == "" {
		return nil, errors.New("MONGODB_URI is required when CHAT_STORE_BACKEND=mongodb")
	}

	timeout := envDuration("MONGODB_TIMEOUT", 5*time.Second)
	connectTimeout := envDuration("MONGODB_CONNECT_TIMEOUT", 10*time.Second)
	serverSelectionTimeout := envDuration("MONGODB_SERVER_SELECTION_TIMEOUT", 5*time.Second)
	database := envString("MONGODB_DATABASE", "rag_app")
	usersCollection := envString("MONGODB_USERS_COLLECTION", "users")
	sessionsCollection := envString("MONGODB_SESSIONS_COLLECTION", "chat_sessions")

	ctx, cancel := context.WithTimeout(context.Background(), connectTimeout)
	defer cancel()

	clientOptions := options.Client().
		ApplyURI(uri).
		SetConnectTimeout(connectTimeout).
		SetServerSelectionTimeout(serverSelectionTimeout).
		SetMaxPoolSize(uint64(envPositiveInt("MONGODB_MAX_POOL_SIZE", 100))).
		SetMinPoolSize(uint64(envPositiveInt("MONGODB_MIN_POOL_SIZE", 0)))
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		return nil, err
	}

	pingCtx, pingCancel := context.WithTimeout(context.Background(), timeout)
	defer pingCancel()
	if err := client.Ping(pingCtx, readpref.Primary()); err != nil {
		_ = client.Disconnect(context.Background())
		return nil, err
	}

	db := client.Database(database)
	store := &mongoStore{
		client:     client,
		users:      db.Collection(usersCollection),
		sessions:   db.Collection(sessionsCollection),
		timeout:    timeout,
		retries:    envPositiveInt("MONGODB_RETRIES", 3),
		retryDelay: envDuration("MONGODB_RETRY_DELAY", 200*time.Millisecond),
	}
	if err := store.ensureIndexes(); err != nil {
		_ = client.Disconnect(context.Background())
		return nil, err
	}
	return store, nil
}

func (m *mongoStore) ensureIndexes() error {
	return m.withRetry(func(ctx context.Context) error {
		_, err := m.users.Indexes().CreateMany(ctx, []mongo.IndexModel{
			{
				Keys:    bson.D{{Key: "user_id", Value: 1}},
				Options: options.Index().SetUnique(true).SetName("user_id_unique"),
			},
			{
				Keys:    bson.D{{Key: "username", Value: 1}},
				Options: options.Index().SetUnique(true).SetName("username_unique"),
			},
		})
		if err != nil {
			return err
		}
		_, err = m.sessions.Indexes().CreateMany(ctx, []mongo.IndexModel{
			{
				Keys:    bson.D{{Key: "session_id", Value: 1}},
				Options: options.Index().SetUnique(true).SetName("session_id_unique"),
			},
			{
				Keys:    bson.D{{Key: "user_id", Value: 1}},
				Options: options.Index().SetName("user_id"),
			},
			{
				Keys:    bson.D{{Key: "user_id", Value: 1}, {Key: "updated_at", Value: -1}},
				Options: options.Index().SetName("user_updated_at"),
			},
		})
		return err
	})
}

func (m *mongoStore) loadUserSessions(userID string) (map[string]Session, error) {
	result := make(map[string]Session)
	err := m.withRetry(func(ctx context.Context) error {
		cursor, err := m.sessions.Find(ctx, bson.M{"user_id": userID})
		if err != nil {
			return err
		}
		defer cursor.Close(ctx)

		for cursor.Next(ctx) {
			var session Session
			if err := cursor.Decode(&session); err != nil {
				return err
			}
			if session.ID != "" {
				result[session.ID] = session
			}
		}
		return cursor.Err()
	})
	return result, err
}

func (m *mongoStore) upsertUser(user User) error {
	return m.withRetry(func(ctx context.Context) error {
		_, err := m.users.UpdateOne(
			ctx,
			bson.M{"user_id": user.ID},
			bson.M{
				"$set": bson.M{
					"username":      user.Username,
					"password_hash": user.PasswordHash,
					"updated_at":    user.UpdatedAt,
					"last_login":    user.LastLogin,
				},
				"$setOnInsert": bson.M{
					"_id":        primitive.NewObjectID(),
					"user_id":    user.ID,
					"created_at": user.CreatedAt,
				},
			},
			options.Update().SetUpsert(true),
		)
		return err
	})
}

func (m *mongoStore) authenticateOrRegister(userID, username, password string) (bool, error) {
	var existing User
	err := m.withRetry(func(ctx context.Context) error {
		return m.users.FindOne(ctx, bson.M{"user_id": userID}).Decode(&existing)
	})
	if err == nil {
		if bcrypt.CompareHashAndPassword([]byte(existing.PasswordHash), []byte(password)) != nil {
			return false, ErrInvalidCredentials
		}
		now := time.Now().Unix()
		return false, m.withRetry(func(ctx context.Context) error {
			_, err := m.users.UpdateOne(
				ctx,
				bson.M{"user_id": userID},
				bson.M{"$set": bson.M{"updated_at": now, "last_login": now}},
			)
			return err
		})
	}
	if !errors.Is(err, mongo.ErrNoDocuments) {
		return false, err
	}

	passwordHash, err := hashPassword(password)
	if err != nil {
		return false, err
	}
	now := time.Now().Unix()
	user := User{
		ID:           userID,
		Username:     username,
		PasswordHash: passwordHash,
		CreatedAt:    now,
		UpdatedAt:    now,
		LastLogin:    now,
	}
	err = m.withRetry(func(ctx context.Context) error {
		_, err := m.users.InsertOne(ctx, bson.M{
			"_id":           primitive.NewObjectID(),
			"user_id":       user.ID,
			"username":      user.Username,
			"password_hash": user.PasswordHash,
			"created_at":    user.CreatedAt,
			"updated_at":    user.UpdatedAt,
			"last_login":    user.LastLogin,
		})
		return err
	})
	if mongo.IsDuplicateKeyError(err) {
		return m.authenticateOrRegister(userID, username, password)
	}
	return true, err
}

func (m *mongoStore) upsertSessions(sessions map[string]Session) error {
	return m.withRetry(func(ctx context.Context) error {
		for _, session := range sessions {
			_, err := m.sessions.UpdateOne(
				ctx,
				bson.M{"session_id": session.ID},
				bson.M{
					"$set": bson.M{
						"session_id":          session.ID,
						"user_id":             session.UserID,
						"messages":            session.Messages,
						"summary":             session.Summary,
						"summarized_messages": session.SummarizedMessages,
						"updated_at":          session.UpdatedAt,
					},
					"$setOnInsert": bson.M{"_id": primitive.NewObjectID()},
				},
				options.Update().SetUpsert(true),
			)
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (m *mongoStore) withRetry(operation func(context.Context) error) error {
	var lastErr error
	for attempt := 1; attempt <= m.retries; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), m.timeout)
		err := operation(ctx)
		cancel()
		if err == nil {
			return nil
		}
		if errors.Is(err, mongo.ErrNoDocuments) {
			return err
		}
		lastErr = err
		if attempt < m.retries {
			time.Sleep(m.retryDelay * time.Duration(attempt))
		}
	}
	return lastErr
}

func firstEnv(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}

func envString(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	duration, err := time.ParseDuration(value)
	if err == nil && duration > 0 {
		return duration
	}
	seconds, err := strconv.Atoi(value)
	if err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	return fallback
}

func hashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func mergeSummary(existing string, messages []Message) string {
	parts := make([]string, 0, len(messages)+1)
	if existing != "" {
		parts = append(parts, existing)
	}
	parts = append(parts, summarizeMessages(messages))
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func summarizeMessages(messages []Message) string {
	const maxLineChars = 500
	lines := make([]string, 0, len(messages))
	for _, message := range messages {
		content := strings.Join(strings.Fields(message.Content), " ")
		if content == "" {
			continue
		}
		if len(content) > maxLineChars {
			content = content[:maxLineChars] + "..."
		}
		lines = append(lines, message.Role+": "+content)
	}
	return strings.Join(lines, "\n")
}
