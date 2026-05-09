import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

// render / createRoot 都是 ReactDOM 带来的
ReactDOM.createRoot(document.getElementById('root')).render(
    // <React.StrictMode> 是一个开发期的“检查模式”，只在开发环境生效
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
