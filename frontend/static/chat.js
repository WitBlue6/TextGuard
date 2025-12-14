let ws;
let isConnected = false;

// 文件上传处理
const fileInput = document.getElementById('file');
const fileInfo = document.getElementById('file-info');

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        fileInfo.textContent = `已选择文件: ${file.name} (${formatFileSize(file.size)})`;
        fileInfo.style.display = 'block';
    } else {
        fileInfo.style.display = 'none';
    }
});

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 发送消息函数
function sendMessage() {
    const message = document.getElementById("message").value;
    const fileElem = document.getElementById("file");
    const file = fileElem.files[0];
    const sendBtn = document.getElementById('send-btn');
    const logs = document.getElementById('logs');

    // 验证输入
    if (!message.trim() && !file) {
        addLog("请输入文本或选择文件！", "error");
        return;
    }

    // 禁用发送按钮并显示加载状态
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<div class="loading"></div> 发送中...';

    // 清空之前的连接
    if (ws) {
        ws.close();
    }

    // 建立新的WebSocket连接
    ws = new WebSocket("ws://localhost:8000/ws/chat");

    ws.onopen = () => {
        isConnected = true;
        addLog("已连接到服务器...", "info");
        
        const reader = new FileReader();
        if (file) {
            reader.onload = () => {
                ws.send(JSON.stringify({message: message, file: {filename: file.name, content: reader.result}}));
            }
            reader.readAsDataURL(file);
        } else {
            ws.send(JSON.stringify({message: message}));
        }
    }

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.log) {
                addLog(data.log, "log");
            } else if (data.results) {
                addLog("\n=== 任务完成 ===", "success");
                // 可以在这里添加结果的特殊格式化显示
                console.log(data.results);
            } else if (data.error) {
                addLog("错误: " + data.error, "error");
            }
        } catch (error) {
            addLog("解析服务器响应出错: " + error.message, "error");
        }
    }

    ws.onclose = () => {
        isConnected = false;
        addLog("连接已关闭", "info");
        // 恢复发送按钮状态
        sendBtn.disabled = false;
        sendBtn.innerHTML = '发送';
    }

    ws.onerror = (error) => {
        addLog("WebSocket错误: " + error.message, "error");
        isConnected = false;
        // 恢复发送按钮状态
        sendBtn.disabled = false;
        sendBtn.innerHTML = '发送';
    }
}

// 添加日志到显示区域
function addLog(message, type = "log") {
    const logs = document.getElementById("logs");
    
    // 根据类型添加不同的前缀或样式
    let formattedMessage = message;
    switch (type) {
        case "error":
            formattedMessage = "[错误] " + message;
            break;
        case "success":
            formattedMessage = "[成功] " + message;
            break;
        case "info":
            formattedMessage = "[信息] " + message;
            break;
    }
    
    logs.value += formattedMessage + "\n";
    logs.scrollTop = logs.scrollHeight;
}

// 页面加载完成后
window.addEventListener('load', function() {
    addLog("欢迎使用文本一致性检测系统！", "info");
    addLog("请输入文本或选择文件，然后点击发送按钮开始检测。", "info");
});