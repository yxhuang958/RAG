import { useEffect, useMemo, useState } from 'react';
import './App.css';

const statusItems = [
    { label: 'Auth', value: 'JWT' },
    { label: 'History', value: 'Backend cache' },
    { label: 'Vector Store', value: 'Milvus' },
    { label: 'LLM', value: 'Cloud API' }
];

const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const formatContextLabel = (ctx, index) => {
    const chunk = ctx.chunk_index ?? index;
    const score = ctx.rerank_score ?? ctx.score;
    const scoreLabel = Number.isFinite(Number(score)) ? `score ${Number(score).toFixed(3)}` : 'score n/a';
    const docLabel = ctx.doc_id ? String(ctx.doc_id).split('-').slice(0, -1).join('-') || ctx.doc_id : 'document';
    const preview = (ctx.text || '').replace(/\s+/g, ' ').trim();

    return `#${chunk} · ${scoreLabel} · ${docLabel}${preview ? ` · ${preview}` : ''}`;
};

function App() {
    const [username, setUsername] = useState('demo');
    const [password, setPassword] = useState('demo');
    const [token, setToken] = useState('');
    const [userId, setUserId] = useState('');
    const [loginStatus, setLoginStatus] = useState(token ? 'Signed in' : 'Please sign in');
    const [sessions, setSessions] = useState([]);
    const [sessionId, setSessionId] = useState('');
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('Ready for ingestion...');
    const [isUploading, setIsUploading] = useState(false);
    const [shouldPollUpload, setShouldPollUpload] = useState(false);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [contexts, setContexts] = useState([]);
    const [history, setHistory] = useState([]);
    const [isQuerying, setIsQuerying] = useState(false);
    const useRerank = true;

    const authHeaders = useMemo(() => ({
        Authorization: `Bearer ${token}`
    }), [token]);

    useEffect(() => {
        if (!token) return;
        refreshSessions();
    }, [token]);

    useEffect(() => {
        if (!token) return;

        const loadUploadStatus = async () => {
            try {
                const response = await fetch(`${apiBaseUrl}/api/upload/status`, {
                    headers: authHeaders
                });
                const result = await response.json();
                if (!response.ok) return;

                if (result.status === 'processing') {
                    setIsUploading(true);
                    setShouldPollUpload(true);
                    setUploadStatus(`Uploading: ${result.filename || 'document'}`);
                } else if (result.status === 'failed') {
                    setUploadStatus(result.error || 'Upload failed.');
                }
            } catch (error) {
                setUploadStatus('Ready for ingestion...');
            }
        };

        loadUploadStatus();
    }, [authHeaders, token]);

    useEffect(() => {
        if (!token || !shouldPollUpload) return undefined;

        let cancelled = false;
        const pollUploadStatus = async () => {
            try {
                const response = await fetch(`${apiBaseUrl}/api/upload/status`, {
                    headers: authHeaders
                });
                const result = await response.json();
                if (cancelled || !response.ok) return;

                if (result.status === 'completed') {
                    const uploadedName = result.result?.filename || result.result?.doc_id || result.filename || selectedFile?.name;
                    setUploadStatus(`Uploaded: ${uploadedName}`);
                    setIsUploading(false);
                    setShouldPollUpload(false);
                } else if (result.status === 'failed') {
                    setUploadStatus(result.error || 'Upload failed.');
                    setIsUploading(false);
                    setShouldPollUpload(false);
                } else if (result.status === 'processing') {
                    setUploadStatus(`Uploading: ${result.filename || selectedFile?.name || 'document'}`);
                }
            } catch (error) {
                if (!cancelled) {
                    setUploadStatus('Checking upload status...');
                }
            }
        };

        pollUploadStatus();
        const interval = window.setInterval(pollUploadStatus, 2000);
        return () => {
            cancelled = true;
            window.clearInterval(interval);
        };
    }, [authHeaders, selectedFile, shouldPollUpload, token]);

    const handleLogin = async () => {
        setLoginStatus('Signing in...');

        try {
            const response = await fetch(`${apiBaseUrl}/api/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const result = await response.json();

            if (!response.ok) {
                setLoginStatus(result.error || 'Login failed.');
                return;
            }

            setToken(result.token);
            setUserId(result.user_id);
            setSessions(result.sessions || []);
            setLoginStatus(`Signed in as ${result.user_id}`);
        } catch (error) {
            setLoginStatus('Login failed. Please try again.');
        }
    };

    const handleLogout = () => {
        setToken('');
        setUserId('');
        setSessions([]);
        setSessionId('');
        setHistory([]);
        setAnswer('');
        localStorage.removeItem('rag_token');
        localStorage.removeItem('rag_user_id');
        localStorage.removeItem('rag_session_id');
        setLoginStatus('Signed out');
    };

    const refreshSessions = async (activeSessionId = sessionId) => {
        try {
            const response = await fetch(`${apiBaseUrl}/api/sessions`, {
                headers: authHeaders
            });
            const result = await response.json();
            if (response.ok) {
                setSessions(result.sessions || []);
                const active = result.sessions?.find((item) => item.session_id === activeSessionId);
                if (active) {
                    setHistory(active.messages || []);
                }
            }
        } catch (error) {
            setLoginStatus('Could not refresh sessions.');
        }
    };

    const handleFileChange = (event) => {
        if (isUploading) return;
        const file = event.target.files?.[0] || null;
        setSelectedFile(file);
        setUploadStatus(file ? `Selected: ${file.name}` : 'Ready for ingestion...');
    };

    const handleUpload = async () => {
        if (!token) {
            setUploadStatus('Please sign in first.');
            return;
        }
        if (!selectedFile) {
            setUploadStatus('Please select a PDF file first.');
            return;
        }

        setIsUploading(true);
        setShouldPollUpload(false);
        setUploadStatus('Uploading...');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(`${apiBaseUrl}/api/upload`, {
                method: 'POST',
                headers: authHeaders,
                body: formData
            });
            const result = await response.json();

            if (!response.ok) {
                setUploadStatus(result.error || 'Upload failed.');
                setIsUploading(false);
                setShouldPollUpload(false);
                return;
            }

            setUploadStatus(`Uploading: ${result.filename || selectedFile.name}`);
            setShouldPollUpload(true);
        } catch (error) {
            setUploadStatus('Upload failed. Please try again.');
            setIsUploading(false);
            setShouldPollUpload(false);
        }
    };

    const handleQuery = async () => {
        if (!token) {
            setAnswer('Please sign in first.');
            return;
        }
        if (!question.trim()) {
            setAnswer('Please enter a question.');
            return;
        }

        const submittedQuestion = question.trim();
        const optimisticUserMessage = {
            role: 'user',
            content: submittedQuestion,
            created_at: Date.now()
        };
        setIsQuerying(true);
        setQuestion('');
        setAnswer('Querying...');
        setContexts([]);
        setHistory((current) => [...current, optimisticUserMessage]);

        try {
            const response = await fetch(`${apiBaseUrl}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...authHeaders },
                body: JSON.stringify({
                    question: submittedQuestion,
                    top_k: import.meta.env.VITE_RETRIEVE_TOPK || 30,
                    rerank: useRerank,
                    session_id: sessionId
                })
            });
            const result = await response.json();

            if (!response.ok) {
                setAnswer(result.error || 'Query failed.');
                setQuestion(submittedQuestion);
                setHistory((current) => current.filter((item) => item !== optimisticUserMessage));
                return;
            }

            const resolvedSessionId = result.session_id || sessionId;
            const resolvedAnswer = result.answer || 'No answer returned.';

            if (result.session_id) {
                setSessionId(result.session_id);
            }

            setAnswer(resolvedAnswer);
            setContexts(result.contexts || []);

            // The query response already contains the authoritative history.
            // Update both views from it instead of immediately re-reading
            // MongoDB, whose asynchronous write may still be pending.
            const nextHistory = Array.isArray(result.history) && result.history.length > 0
                ? result.history
                : [
                    ...history,
                    optimisticUserMessage,
                    {
                        role: 'assistant',
                        content: resolvedAnswer,
                        created_at: Date.now()
                    }
                ];

            setHistory(nextHistory);
            setSessions((current) => {
                const existing = current.find((item) => item.session_id === resolvedSessionId);
                const updated = {
                    ...(existing || {}),
                    session_id: resolvedSessionId,
                    messages: nextHistory,
                    updated_at: Date.now()
                };
                return [
                    updated,
                    ...current.filter((item) => item.session_id !== resolvedSessionId)
                ];
            });
        } catch (error) {
            setAnswer('Query failed. Please try again.');
            setQuestion(submittedQuestion);
            setHistory((current) => current.filter((item) => item !== optimisticUserMessage));
        } finally {
            setIsQuerying(false);
        }
    };

    const handleQuestionKeyDown = (event) => {
        if (event.key !== 'Enter' || event.shiftKey) return;
        event.preventDefault();
        if (!isQuerying) {
            handleQuery();
        }
    };

    const selectSession = (session) => {
        setSessionId(session.session_id);
        setHistory(session.messages || []);
        const lastAnswer = [...(session.messages || [])].reverse().find((item) => item.role === 'assistant');
        setAnswer(lastAnswer?.content || '');
    };

    const startNewSession = () => {
        setSessionId('');
        setHistory([]);
        setAnswer('');
        setContexts([]);
    };

    return (
        <div className="app">
            <header className="hero">
                <div className='eyebrow_'>
                    <p className="eyebrow">Private Knowledge System</p>
                    <h1>Session-aware RAG Console</h1>
                    <p className="subtitle">
                        {/* JWT keeps user state at the app layer; the pipeline receives text context and returns an answer. */}
                    </p>
                </div>
                <div className="hero-card">
                    <h2>System status</h2>
                    <div className="status-grid">
                        {statusItems.map((item) => (
                            <div className="status-item" key={item.label}>
                                <span>{item.label}</span>
                                <strong>{item.value}</strong>
                            </div>
                        ))}
                    </div>
                </div>
            </header>

            <main className="grid">
                <section className="card card-left">
                    <h3>Account</h3>
                    <p>{loginStatus}</p>
                    <div className="login-form">
                        <input value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Username" />
                        <input value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" type="password" />
                        <div className="input-row">
                            <button type="button" onClick={handleLogin}>Sign in</button>
                            <button type="button" className="ghost-button" onClick={handleLogout}>Sign out</button>
                        </div>
                    </div>
                    <p className="meta-line">{userId || 'No active user'}</p>

                    <div className="section-title">
                        <h3>Sessions</h3>
                        <button type="button" className="ghost-button" onClick={startNewSession}>New</button>
                    </div>
                    <div className="session-list">
                        {sessions.length === 0 ? (
                            <span className="empty-text">No saved sessions</span>
                        ) : sessions.map((session) => (
                            <button
                                type="button"
                                className={session.session_id === sessionId ? 'session-item active' : 'session-item'}
                                key={session.session_id}
                                onClick={() => selectSession(session)}
                            >
                                <strong>{session.session_id}</strong>
                                <span>{session.messages?.length || 0} messages</span>
                            </button>
                        ))}
                    </div>

                    <div className="upload-section">
                        <h3>Upload knowledge</h3>
                        <div className="upload-box">
                            <span className="upload-file-name" title={selectedFile?.name || 'Select PDF files'}>
                                {selectedFile?.name || 'Select PDF files'}
                            </span>
                            <label className={isUploading ? 'upload-button disabled' : 'upload-button'} htmlFor="pdf-upload">
                                {selectedFile ? 'Change' : 'Browse'}
                            </label>
                            <input id="pdf-upload" type="file" accept="application/pdf" onChange={handleFileChange} disabled={isUploading} />
                        </div>
                        <div className="file-list">
                            <div className="file-item">
                                <span>{uploadStatus}</span>
                            </div>
                            <button type="button" onClick={handleUpload} disabled={isUploading}>
                                {isUploading ? 'Uploading...' : 'Send to backend'}
                            </button>
                        </div>
                    </div>
                </section>

                <section className="card card-right">
                    <div className="section-title">
                        <div>
                            <h3>Ask the knowledge base</h3>
                            <p>{sessionId ? `Session: ${sessionId}` : 'New session will be created on first question.'}</p>
                        </div>
                    </div>

                    <div className="chat-container">
                        {history.length === 0 ? (
                            <div className="chat-msg bot">
                                <p>{answer || 'Sign in, upload a PDF, and ask me anything about it.'}</p>
                            </div>
                        ) : history.map((message, index) => (
                            <div className={`chat-msg ${message.role === 'user' ? 'user' : 'bot'}`} key={`${message.created_at}-${index}`}>
                                <p>{message.content}</p>
                            </div>
                        ))}
                        <div className="rag-citation">
                            <p className="citation-label">Top 3 RAG Contexts</p>
                            <div className="token-list">
                                {contexts.length === 0 ? (
                                    <span className="token">No contexts yet</span>
                                ) : contexts.slice(0, 3).map((ctx, index) => (
                                    <span className="token context-token" key={`${ctx.doc_id || 'chunk'}-${index}`} title={ctx.text || formatContextLabel(ctx, index)}>
                                        {formatContextLabel(ctx, index)}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="chat-input-area">
                        <textarea
                            rows="3"
                            placeholder="Type your question..."
                            value={question}
                            onChange={(event) => setQuestion(event.target.value)}
                            onKeyDown={handleQuestionKeyDown}
                        />
                        <div className="input-row">
                            <button type="button" onClick={handleQuery} disabled={isQuerying}>
                                {isQuerying ? 'Asking...' : 'Send'}
                            </button>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}

export default App;
