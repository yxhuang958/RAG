// 这个组件和 App.css 配合使用，每一个组件都是一套独立的 {.jsx, .css}
// App 包含3个主要部分：
// 1. Hero 区：展示系统简介和状态卡片
// 2. 主体区：包含上传、问答、活动三个功能卡片
// 3. 状态卡片：显示 RAG 系统的核心组件状态（向量库、数据库、队列、LLM）
import './App.css';

// statusItems 定义了系统状态卡片中要展示的组件和它们的当前状态，保存在前端作为静态数据。
// 每个对象包含一个 label（组件名称）和一个 value（当前状态）。
// 后续可以通过 API 获取实际状态并动态更新这些值。 -- important
const statusItems = [
    { label: 'Vector Store', value: 'Milvus' },
    { label: 'Metadata DB', value: 'MongoDB' },
    { label: 'Queue', value: 'Redis' },
    { label: 'LLM', value: 'Cloud API' }
];

function App() {
    return (
        <div className="app">
            <header className="hero">
                <div>
                    <p className="eyebrow">RAG System</p>
                    <h1>Retrieve, Reason, Respond.</h1>
                    <p className="subtitle">
                        A lightweight front-end for the Go + Python RAG pipeline.
                        Upload documents, monitor ingestion, and ask questions.
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
                <section className="card">
                    <h3>Upload knowledge</h3>
                    <p>Drag and drop files or select from local storage.</p>
                    <div className="input-row">
                        <input type="text" placeholder="Project name" />
                        <button type="button">Create bucket</button>
                    </div>
                    <div className="upload-box">
                        <span>Drop files here</span>
                        <button type="button">Choose files</button>
                    </div>
                    <div className="tag-row">
                        <span className="tag">PDF</span>
                        <span className="tag">Markdown</span>
                        <span className="tag">Docx</span>
                    </div>
                </section>

                <section className="card">
                    <h3>Ask the knowledge base</h3>
                    <p>Send a question to the RAG pipeline and stream answers.</p>
                    <textarea rows="6" placeholder="Ask anything about your documents..." />
                    <div className="input-row">
                        <select>
                            <option>rerank: on</option>
                            <option>rerank: off</option>
                        </select>
                        <button type="button">Send question</button>
                    </div>
                    <div className="response">
                        <p className="response-label">Latest response</p>
                        <p>
                            Ready to connect to the backend. Configure the API URL in
                            <code> .env</code> when the server is running.
                        </p>
                    </div>
                </section>

                <section className="card wide">
                    <h3>Pipeline activity</h3>
                    <p>Track ingestion, embeddings, and generation tasks.</p>
                    <div className="timeline">
                        <div>
                            <strong>Ingestion</strong>
                            <span>Waiting for files</span>
                        </div>
                        <div>
                            <strong>Embedding</strong>
                            <span>Queue idle</span>
                        </div>
                        <div>
                            <strong>Generation</strong>
                            <span>No active sessions</span>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}

export default App;
