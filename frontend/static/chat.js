let ws;
let isConnected = false;
let currentFullscreenId = null;
let currentResults = { consistency: null, grammar: null };

// 确保DOM加载完成后再执行DOM操作
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM已加载完成');
    
    // 文件上传处理
    const fileInput = document.getElementById('file');
    const fileInfo = document.getElementById('file-info');

    if (fileInput && fileInfo) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                fileInfo.textContent = `已选择文件: ${file.name} (${formatFileSize(file.size)})`;
                fileInfo.style.display = 'block';
            } else {
                fileInfo.style.display = 'none';
            }
        });
        console.log('文件上传事件已绑定');
    } else {
        console.error('文件上传相关元素不存在');
    }
    
    // 为结果区域添加ID
    const consistencySection = document.querySelector('.result-section:nth-child(1)');
    const grammarSection = document.querySelector('.result-section:nth-child(2)');
    if (consistencySection) consistencySection.id = 'consistency-section';
    if (grammarSection) grammarSection.id = 'grammar-section';

    // 绑定反馈表单提交事件
    const feedbackForm = document.getElementById('feedback-form');
    if (feedbackForm) {
        feedbackForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitFeedback();
        });
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

// 获取元素的安全函数
function safeGetElementById(id) {
    const element = document.getElementById(id);
    if (!element) {
        console.error(`元素 ${id} 不存在`);
    }
    return element;
}

// 发送消息函数
function sendMessage(pipelineType) {
    console.log('sendMessage函数被调用，pipelineType:', pipelineType);
    
    try {
        // 使用安全的方式获取DOM元素
        const messageInput = safeGetElementById("message");
        const fileElem = safeGetElementById("file");
        const consistencyBtn = safeGetElementById('consistency-btn');
        const grammarBtn = safeGetElementById('grammar-btn');
        const stopBtn = safeGetElementById('stop-btn');
        const logs = safeGetElementById('logs');

        // 验证必要元素是否存在
        if (!messageInput || !fileElem || !consistencyBtn || !grammarBtn || !stopBtn || !logs) {
            console.error('缺少必要的DOM元素');
            alert('页面加载不完整，请刷新页面重试');
            return;
        }

        const message = messageInput.value;
        const file = fileElem.files ? fileElem.files[0] : null;

        console.log('消息内容:', message);
        console.log('文件:', file);

        // 验证输入
        if (!message.trim() && !file) {
            addLog("请输入文本或选择文件！", "error");
            return;
        }

        // 根据pipeline类型禁用对应的按钮
        if (pipelineType === 'consistency') {
            consistencyBtn.disabled = true;
            consistencyBtn.innerHTML = '<div class="loading"></div> 处理中...';
            console.log('一致性检测按钮已禁用');
        } else {
            grammarBtn.disabled = true;
            grammarBtn.innerHTML = '<div class="loading"></div> 处理中...';
            console.log('语法纠错按钮已禁用');
        }

        // 清空之前的连接
        if (ws) {
            ws.close();
            console.log('之前的WebSocket连接已关闭');
        }

        // 建立新的WebSocket连接
        ws = new WebSocket("ws://localhost:8000/ws/chat");
        console.log('WebSocket连接已创建');

        ws.onopen = () => {
            isConnected = true;
            addLog("已连接到服务器...", "info");
            console.log('WebSocket连接已打开');
            
            const reader = new FileReader();
            if (file) {
                reader.onload = () => {
                    const data = JSON.stringify({message: message, file: {filename: file.name, content: reader.result}, pipeline: pipelineType});
                    ws.send(data);
                    console.log('已发送包含文件的数据:', data);
                }
                reader.readAsDataURL(file);
            } else {
                const data = JSON.stringify({message: message, pipeline: pipelineType});
                ws.send(data);
                console.log('已发送文本数据:', data);
            }
        }

        ws.onmessage = (event) => {
            console.log('收到WebSocket消息:', event.data);
            try {
                const data = JSON.parse(event.data);
                if (data.log) {
                    addLog(data.log, "log");
                } else if (data.results) {
                    addLog("\n=== 任务完成 ===", "success");
                    
                    // 根据pipeline类型显示结果
                    if (data.pipeline === 'consistency') {
                        currentResults.consistency = data.results;
                        displayResult('consistency', data.results);
                        consistencyBtn.disabled = false;
                        consistencyBtn.innerHTML = '一致性检测';
                    } else {
                        currentResults.grammar = data.results;
                        displayResult('grammar', data.results);
                        grammarBtn.disabled = false;
                        grammarBtn.innerHTML = '语法纠错';
                    }
                    
                    console.log('处理结果:', data.results);
                } else if (data.feedback_result) {
                    addLog(`\n=== 反馈已提交 ===\n${data.feedback_result}`, "success");
                    closeFeedbackModal();
                } else if (data.error) {
                    addLog("错误: " + data.error, "error");
                    
                    // 发生错误，恢复按钮状态
                    consistencyBtn.disabled = false;
                    consistencyBtn.innerHTML = '一致性检测';
                    grammarBtn.disabled = false;
                    grammarBtn.innerHTML = '语法纠错';
                }
            } catch (error) {
                addLog("解析服务器响应出错: " + error.message, "error");
                console.error('解析响应出错:', error);
                
                // 发生错误，恢复按钮状态
                consistencyBtn.disabled = false;
                consistencyBtn.innerHTML = '一致性检测';
                grammarBtn.disabled = false;
                grammarBtn.innerHTML = '语法纠错';
            }
        }

        ws.onclose = () => {
            isConnected = false;
            addLog("连接已关闭", "info");
            console.log('WebSocket连接已关闭');
            
            // 连接关闭，恢复按钮状态
            consistencyBtn.disabled = false;
            consistencyBtn.innerHTML = '一致性检测';
            grammarBtn.disabled = false;
            grammarBtn.innerHTML = '语法纠错';
        }

        ws.onerror = (error) => {
            addLog("WebSocket错误: " + error.message, "error");
            console.error('WebSocket错误:', error);
            isConnected = false;
            
            // 发生错误，恢复按钮状态
            consistencyBtn.disabled = false;
            consistencyBtn.innerHTML = '一致性检测';
            grammarBtn.disabled = false;
            grammarBtn.innerHTML = '语法纠错';
        }
    } catch (error) {
        console.error('sendMessage函数出错:', error);
        addLog("发送消息失败: " + error.message, "error");
        
        // 发生错误，恢复按钮状态
        const consistencyBtn = safeGetElementById('consistency-btn');
        const grammarBtn = safeGetElementById('grammar-btn');
        if (consistencyBtn) {
            consistencyBtn.disabled = false;
            consistencyBtn.innerHTML = '一致性检测';
        }
        if (grammarBtn) {
            grammarBtn.disabled = false;
            grammarBtn.innerHTML = '语法纠错';
        }
    }
}

// 显示结果函数
function displayResult(pipelineType, results) {
    // 添加更多调试信息
    console.log('显示结果，pipelineType:', pipelineType);
    console.log('结果类型:', typeof results);
    console.log('结果是否为数组:', Array.isArray(results));
    console.log('结果内容:', results);
    
    const resultElement = safeGetElementById(`${pipelineType}-result`);
    if (!resultElement) {
        console.error(`结果显示元素 ${pipelineType}-result 不存在`);
        return;
    }
    
    // 清空之前的结果
    resultElement.innerHTML = '';
    
    if (pipelineType === 'consistency') {
        // 显示一致性检测结果
        if (Array.isArray(results)) {
            let resultHtml = '<ul>';
            results.forEach((result, index) => {
                resultHtml += `<li>实体 ${index + 1}: ${JSON.stringify(result)}</li>`;
            });
            resultHtml += '</ul>';
            resultElement.innerHTML = resultHtml;
        } else {
            resultElement.innerHTML = JSON.stringify(results, null, 2);
        }
    } else {
        // 显示语法纠错结果
        try {
            if (Array.isArray(results)) {
                let resultHtml = '<div class="grammar-results">';
                
                // 遍历每个结果
                results.forEach((result, index) => {
                    console.log(`第${index + 1}个结果:`, result);
                    
                    // 确保result是对象
                    if (typeof result === 'object' && result !== null) {
                        resultHtml += `
                            <div class="grammar-result-item">
                                <p><strong>索引:</strong> ${index + 1}</p>
                        `;
                        
                        // 显示原始文本
                        if (result.original_text) {
                            resultHtml += `<p><strong>原文:</strong> ${result.original_text}</p>`;
                        } else if (result.original) {
                            resultHtml += `<p><strong>原文:</strong> ${result.original}</p>`;
                        }
                        
                        // 显示纠错后文本
                        if (result.content) {
                            resultHtml += `<p><strong>纠错后:</strong> ${result.content}</p>`;
                        } else if (result.correct) {
                            resultHtml += `<p><strong>纠错后:</strong> ${result.correct}</p>`;
                        }
                        
                        // 显示状态
                        if (result.correct !== undefined) {
                            resultHtml += `<p><strong>状态:</strong> ${result.correct ? '无语法错误' : '存在语法错误'}</p>`;
                        }
                        
                        // 显示错误原因
                        if (result.reason) {
                            resultHtml += `<p><strong>错误原因:</strong> ${result.reason}</p>`;
                        }
                        
                        resultHtml += '</div>';
                    } else {
                        // 如果result不是对象，直接显示
                        resultHtml += `<div class="grammar-result-item">${JSON.stringify(result)}</div>`;
                    }
                });
                
                resultHtml += '</div>';
                resultElement.innerHTML = resultHtml;
            } else if (typeof results === 'object' && results !== null) {
                // 如果results是单个对象，直接显示其内容
                resultElement.innerHTML = JSON.stringify(results, null, 2);
            } else {
                // 其他情况，直接显示
                resultElement.innerHTML = String(results);
            }
        } catch (error) {
            console.error('显示语法纠错结果时出错:', error);
            // 出错时显示原始结果
            resultElement.innerHTML = JSON.stringify(results, null, 2);
        }
    }
}

// 终止pipeline函数
function stopPipeline() {
    console.log('终止pipeline函数被调用');
    if (ws && isConnected) {
        ws.close();
        addLog("已终止pipeline执行", "warning");
    }
}

// 添加日志函数
function addLog(message, type = "info") {
    const logs = safeGetElementById('logs');
    if (!logs) {
        console.error('日志显示元素不存在');
        return;
    }
    
    const timestamp = new Date().toLocaleTimeString();
    
    let logPrefix = "";
    switch(type) {
        case "error":
            logPrefix = "[ERROR]";
            break;
        case "warning":
            logPrefix = "[WARNING]";
            break;
        case "success":
            logPrefix = "[SUCCESS]";
            break;
        case "info":
            logPrefix = "[INFO]";
            break;
        default:
            logPrefix = "[LOG]";
    }
    
    logs.value += `${timestamp} ${logPrefix} ${message}\n`;
    logs.scrollTop = logs.scrollHeight;
}

// 全屏切换功能
function toggleFullscreen(elementId) {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`元素 ${elementId} 不存在`);
        return;
    }
    
    // 检查是否点击的是同一个元素
    const isSameElement = currentFullscreenId === elementId;
    
    // 如果已经有全屏元素，先退出
    if (currentFullscreenId) {
        const currentFullscreenElement = document.getElementById(currentFullscreenId);
        if (currentFullscreenElement) {
            currentFullscreenElement.classList.remove('fullscreen');
            // 恢复全屏按钮状态
            const fullscreenBtn = currentFullscreenElement.querySelector('.fullscreen-btn');
            if (fullscreenBtn) {
                fullscreenBtn.classList.remove('active');
                fullscreenBtn.textContent = '↗';
            }
        }
        currentFullscreenId = null;
        
        // 如果点击的是同一个元素，直接返回
        if (isSameElement) {
            return;
        }
    }
    
    // 进入全屏
    element.classList.add('fullscreen');
    currentFullscreenId = elementId;
    
    // 更新全屏按钮状态
    const fullscreenBtn = element.querySelector('.fullscreen-btn');
    if (fullscreenBtn) {
        fullscreenBtn.classList.add('active');
        fullscreenBtn.textContent = '↙';
    }
}

// 监听ESC键退出全屏
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && currentFullscreenId) {
        const element = document.getElementById(currentFullscreenId);
        if (element) {
            element.classList.remove('fullscreen');
            // 恢复全屏按钮状态
            const fullscreenBtn = element.querySelector('.fullscreen-btn');
            if (fullscreenBtn) {
                fullscreenBtn.classList.remove('active');
                fullscreenBtn.textContent = '↗';
            }
            currentFullscreenId = null;
        }
    }
});

// 打开反馈模态框
function openFeedbackModal(pipelineType) {
    // 检查是否有结果
    if (!currentResults[pipelineType]) {
        alert('请先运行对应的检测任务，获得结果后再提交反馈！');
        return;
    }
    
    const modal = document.getElementById('feedback-modal');
    const modalTitle = document.getElementById('feedback-modal-title');
    const pipelineTypeInput = document.getElementById('feedback-pipeline-type');
    
    if (modal && modalTitle && pipelineTypeInput) {
        modalTitle.textContent = `${pipelineType === 'consistency' ? '一致性检测' : '语法纠错'} - 提交反馈`;
        pipelineTypeInput.value = pipelineType;
        modal.style.display = 'block';
    }
}

// 关闭反馈模态框
function closeFeedbackModal() {
    const modal = document.getElementById('feedback-modal');
    const feedbackForm = document.getElementById('feedback-form');
    
    if (modal) {
        modal.style.display = 'none';
    }
    
    if (feedbackForm) {
        feedbackForm.reset();
    }
}

// 提交反馈
function submitFeedback() {
    const pipelineTypeInput = document.getElementById('feedback-pipeline-type');
    const ratingInputs = document.querySelectorAll('input[name="rating"]');
    const commentInput = document.getElementById('feedback-comment');
    const submitBtn = document.getElementById('feedback-submit-btn');
    
    if (!pipelineTypeInput || !ratingInputs || !commentInput || !submitBtn) {
        console.error('反馈表单元素不存在');
        return;
    }
    
    // 获取选中的评分
    let rating = null;
    for (const input of ratingInputs) {
        if (input.checked) {
            rating = parseInt(input.value);
            break;
        }
    }
    
    if (rating === null) {
        alert('请选择评分！');
        return;
    }
    
    const pipelineType = pipelineTypeInput.value;
    const comment = commentInput.value.trim();
    const results = currentResults[pipelineType];
    
    if (!results) {
        alert('无法获取检测结果，请重新运行检测任务！');
        return;
    }
    
    // 禁用提交按钮
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<div class="loading"></div> 提交中...';
    
    try {
        // 构建反馈数据
        const feedbackData = {
            action: 'feedback',
            pipeline: pipelineType,
            results: results,
            rating: rating,
            comment: comment
        };
        
        // 发送反馈数据
        if (ws && isConnected) {
            ws.send(JSON.stringify(feedbackData));
            console.log('已发送反馈数据:', feedbackData);
        } else {
            // 如果WebSocket断开，重新连接并发送
            ws = new WebSocket("ws://localhost:8000/ws/chat");
            
            ws.onopen = () => {
                isConnected = true;
                ws.send(JSON.stringify(feedbackData));
                console.log('已重新连接并发送反馈数据:', feedbackData);
            };
            
            ws.onerror = () => {
                alert('无法连接到服务器，请稍后重试！');
                submitBtn.disabled = false;
                submitBtn.innerHTML = '提交反馈';
            };
        }
    } catch (error) {
        console.error('提交反馈时出错:', error);
        alert('提交反馈失败，请重试！');
        submitBtn.disabled = false;
        submitBtn.innerHTML = '提交反馈';
    }
}

// 点击模态框外部关闭模态框
window.onclick = function(event) {
    const modal = document.getElementById('feedback-modal');
    if (event.target === modal) {
        closeFeedbackModal();
    }
}