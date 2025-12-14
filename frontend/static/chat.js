let ws;
function sendMessage() {
    const message = document.getElementById("message").value;
    const fileElem = document.getElementById("file");
    const file = fileElem.files[0];

    ws = new WebSocket("ws://localhost:8000/ws/chat");
    ws.onopen = () => {
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
        const data = JSON.parse(event.data);
        const logs = document.getElementById("logs");
        if (data.log) {
            logs.value += data.log + "\n";
            logs.scrollTop = logs.scrollHeight;
        } else if (data.results) {
            logs.value += "\n=== 任务完成 ===\n";
            console.log(data.results);
        } else if (data.error) {
            logs.value += "错误: " + data.error + "\n";
        }
    }

    ws.onclose = () => {
        console.log("WebSocket closed");
    }
}
