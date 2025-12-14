const chatContainer = document.getElementById("chat-container");
const inputText = document.getElementById("input-text");
const sendBtn = document.getElementById("send-btn");
const fileBtn = document.getElementById("file-btn");

let ws;

// 初始化 WebSocket
function initWebSocket() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const host = location.host;
    ws = new WebSocket(`${protocol}://${host}/ws/chat`);

    ws.onopen = () => console.log("WS 已连接");
    ws.onclose = () => console.log("WS 已关闭");
    ws.onerror = (e) => console.error("WS 出错", e);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if(data.type === "reply") {
            addMessage("assistant", JSON.stringify(data.content, null, 2));
        } else if(data.type === "log") {
            addMessage("log", data.content);
        }
        scrollBottom();
    };
}

// 添加消息到聊天窗口
function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = "message " + role;
    div.textContent = text;
    chatContainer.appendChild(div);
    scrollBottom();
}

function scrollBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 发送文本/文件
sendBtn.addEventListener("click", async () => {
    const message = inputText.value.trim();
    const file = fileBtn.files[0];

    if(!message && !file) return;

    // 显示用户消息
    if(message) addMessage("user", message);

    if(file) {
        // 发送文件 base64
        const reader = new FileReader();
        reader.onload = () => {
            ws.send(JSON.stringify({
                type: "file",
                filename: file.name,
                content: reader.result.split(",")[1]
            }));
        };
        reader.readAsDataURL(file);
    } else {
        ws.send(JSON.stringify({
            type: "message",
            content: message
        }));
    }

    inputText.value = "";
    fileBtn.value = "";
});

// 回车发送
inputText.addEventListener("keypress", (e) => {
    if(e.key === "Enter") sendBtn.click();
});

// 初始化
initWebSocket();
