// 这个组件和 App.css 配合使用，每一个组件都是一套独立的 {.jsx, .css}
import { useState } from 'react';
import './App.css';

// statusItems 定义了系统状态卡片中要展示的组件和它们的当前状态，保存在前端作为静态数据。
// 每个对象包含一个 label（组件名称）和一个 value（当前状态）。
// 后续可以通过 API 获取实际状态并动态更新这些值。 -- important
/*
    物理位置：它存在于浏览器的 JS 内存中。
    生命周期：它会在 JS 文件被加载时初始化一次，并且在整个应用运行期间永远只有一份，即便 App 组件被重新渲染，这个数组也不会被重构。
    用途：这种写法最适合放不需要改变的常量（Constants）。
 */
const statusItems = [
    { label: 'Vector Store', value: 'Milvus' },
    { label: 'Metadata DB', value: 'MongoDB' },
    { label: 'Queue', value: 'Redis' },
    { label: 'LLM', value: 'Cloud API' }
];

const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8080';

function App() {
    // return [状态值, 更新状态的函数]，通过这个func来改变状态值
    // params 组件第一次渲染时，useState设定这个状态的默认值
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState('Ready for ingestion...');
    const [isUploading, setIsUploading] = useState(false);

    // <input>下，文件有变化就触发，onchange
    const handleFileChange = (event) => {
        const file = event.target.files?.[0] || null;
        setSelectedFile(file);
        setUploadStatus(file ? `Selected: ${file.name}` : 'Ready for ingestion...');
    };

    // submit点击触发
    const handleUpload = async () => {
        if (!selectedFile) {
            setUploadStatus('Please select a PDF file first.');
            return;
        }

        setIsUploading(true);
        setUploadStatus('Uploading...');

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch(`${apiBaseUrl}/api/upload`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                setUploadStatus(result.error || 'Upload failed.');
                return;
            }

            setUploadStatus(`Uploaded: ${result.filename}`);
        } catch (error) {
            setUploadStatus('Upload failed. Please try again.');
        } finally { // 无论成功还是失败，最后都要执行的清理工作
            setIsUploading(false);
        }
    };

    // page
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
                <section className="card card-left">
                    <h3>Upload knowledge</h3>
                    <p>Upload PDF documents to your vector database.</p>
                    <div className="upload-box">
                        <span>Select PDF files</span>
                        <label className="upload-button" htmlFor="pdf-upload">
                            {selectedFile ? 'Change' : 'Browse'}
                        </label>
                        <input
                            id="pdf-upload"
                            type="file"
                            accept="application/pdf"
                            onChange={handleFileChange}
                        />
                    </div>
                    <div className="file-list">
                        <div className="file-item">
                            <span>{uploadStatus}</span>
                        </div>
                        <button type="button" onClick={handleUpload} disabled={isUploading}>
                            {isUploading ? 'Uploading...' : 'Send to backend'}
                        </button>
                    </div>
                </section>

                <section className="card card-right">
                    <h3>Ask the knowledge base</h3>
                    <p>Context-aware conversation with your documents.</p>

                    <div className="chat-container">
                        <div className="chat-msg bot">
                            <p>Hello! Upload a PDF and ask me anything about it.</p>
                        </div>
                        <div className="rag-citation">
                            <p className="citation-label">Top 3 RAG Contexts (Reranked)</p>
                            <div className="token-list">
                                <span className="token">#chunk_1</span>
                                <span className="token">#chunk_5</span>
                                <span className="token">#chunk_2</span>
                            </div>
                        </div>
                    </div>

                    <div className="chat-input-area">
                        <textarea rows="3" placeholder="Type your question..." />
                        <div className="input-row">
                            <label className="switch-label">
                                <input type="checkbox" defaultChecked /> Rerank
                            </label>
                            <button type="button">Send</button>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}

export default App;
