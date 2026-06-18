package chat

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type Message struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	CreatedAt int64  `json:"created_at"`
}

type Session struct {
	ID        string    `json:"session_id"`
	UserID    string    `json:"user_id"`
	Messages  []Message `json:"messages"`
	UpdatedAt int64     `json:"updated_at"`
}

type PipelineMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Store struct {
	path     string
	mu       sync.RWMutex
	sessions map[string]Session
}

func NewStore(path string) *Store {
	return &Store{
		path:     path,
		sessions: make(map[string]Session),
	}
}

func (s *Store) LoadUser(userID string) error {
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

	snapshot := make(map[string]Session, len(s.sessions))
	for id, cached := range s.sessions {
		snapshot[id] = cached
	}
	s.mu.Unlock()

	if err := s.writeSessions(snapshot); err != nil {
		return Session{}, err
	}
	return session, nil
}

func (s *Store) PipelineHistory(userID, sessionID string) []PipelineMessage {
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

func (s *Store) PipelineMessages(userID, sessionID, question string) []PipelineMessage {
	messages := s.PipelineHistory(userID, sessionID)
	return append(messages, PipelineMessage{Role: "user", Content: question})
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

	content, err := json.MarshalIndent(all, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, content, 0644)
}
