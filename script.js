// ============================================
// STATE MANAGEMENT SYSTEM
// ============================================

// AppState Enum
const AppState = {
    IDLE: 'IDLE',
    FOCUS: 'FOCUS',
    RELAX: 'RELAX'
};

// State Management Object
const StateManager = {
    currentState: AppState.IDLE,
    sessionStartTime: null,
    sessionTimerInterval: null,

    // Initialize the state manager
    init() {
        this.applyTheme(this.currentState);
        this.startSessionTimer();
        this.attachKeyboardListeners();
        this.updateStateDisplay();
    },

    // Set app state
    setState(newState) {
        if (newState === this.currentState) return;
        
        this.currentState = newState;
        this.applyTheme(newState);
        this.resetSessionTimer();
        this.updateStateDisplay();
        
        console.log(`[StateManager] State changed to: ${newState}`);
    },

    // Apply theme based on state
    applyTheme(state) {
        const root = document.documentElement;
        
        // Remove all state classes
        root.classList.remove('state-idle', 'state-focus', 'state-relax');
        
        // Add the appropriate state class
        switch (state) {
            case AppState.FOCUS:
                root.classList.add('state-focus');
                break;
            case AppState.RELAX:
                root.classList.add('state-relax');
                break;
            case AppState.IDLE:
            default:
                root.classList.add('state-idle');
        }
    },

    // Reset session timer
    resetSessionTimer() {
        this.sessionStartTime = Date.now();
    },

    // Start session timer
    startSessionTimer() {
        this.sessionStartTime = Date.now();
        
        if (this.sessionTimerInterval) {
            clearInterval(this.sessionTimerInterval);
        }
        
        this.sessionTimerInterval = setInterval(() => {
            this.updateSessionTimer();
        }, 1000);
    },

    // Update session timer display
    updateSessionTimer() {
        const timerElement = document.getElementById('session-timer');
        if (!timerElement) return;

        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const hours = Math.floor(elapsed / 3600);
        const minutes = Math.floor((elapsed % 3600) / 60);
        const seconds = elapsed % 60;

        timerElement.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    },

    // Update state display
    updateStateDisplay() {
        const stateElement = document.getElementById('app-state');
        if (stateElement) {
            stateElement.textContent = `STATE · ${this.currentState}`;
        }
    },

    // Attach keyboard listeners
    attachKeyboardListeners() {
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case '1':
                    this.setState(AppState.IDLE);
                    event.preventDefault();
                    break;
                case '2':
                    this.setState(AppState.FOCUS);
                    event.preventDefault();
                    break;
                case '3':
                    this.setState(AppState.RELAX);
                    event.preventDefault();
                    break;
            }
        });
    }
};

// Initialize the state manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => StateManager.init());
} else {
    StateManager.init();
}

// --- 全局配置与数据 ---

const bandConfig = {
    "Relaxed & Focused Level": { color: "#34e7e4", data: [] }, 
    "Attention":               { color: "#ffdd59", data: [] }, 
    "Engagement":              { color: "#0be881", data: [] }  
};

let currentActiveBand = "Relaxed & Focused Level";
const MAX_DATA_POINTS = 600; 
const MAX_AMP_VALUE = 2;   

// --- 1. 右上角时间更新 ---
function updateTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        timeElement.textContent = `${hours}:${minutes}:${seconds}`;
    }
}
setInterval(updateTime, 1000);

// --- 2. 实时绘制脑电波波形 ---
function drawRealtimeWave() {
    const svg = document.getElementById('waveform');
    const wavePath = document.getElementById('wavePath');
    const waveStroke = document.getElementById('waveStroke');
    const waveDot = document.getElementById('waveDot'); 
    
    if (!svg || !wavePath || !waveStroke) return;

    // 获取实际物理像素尺寸
    const width = svg.clientWidth > 0 ? svg.clientWidth : 1000;
    const height = svg.clientHeight > 0 ? svg.clientHeight : 100;

    // 映射坐标系
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const currentData = bandConfig[currentActiveBand].data;
    
    // 【修改核心】：向左退回 8px (3px半径 + 4px阴影 + 1px缓冲)，防止小白点被右侧切边
    const paddingRight = 8; 
    const drawWidth = width - paddingRight;

    if (currentData.length === 0) {
        wavePath.setAttribute('d', `M 0 ${height} L ${drawWidth} ${height} Z`);
        waveStroke.setAttribute('points', `0,${height} ${drawWidth},${height}`);
        if(waveDot) waveDot.setAttribute('opacity', '0');
        return;
    }

    let points = [];
    let pathData = '';

    // 数轴上下限
    const Y_MAX = 2; 
    const Y_MIN = 0;

    // 【修改核心】：步长计算现在基于扣除右侧边距后的 drawWidth
    const stepX = drawWidth / (MAX_DATA_POINTS - 1);

    for (let i = 0; i < currentData.length; i++) {
        // 【修改核心】：X坐标的基础偏移量从 width 改为 drawWidth
        const x = drawWidth - ((currentData.length - 1 - i) * stepX);
        
        let val = currentData[i];
        if (val > Y_MAX) val = Y_MAX;
        if (val < Y_MIN) val = Y_MIN;
        
        const y = height - ((val - Y_MIN) / (Y_MAX - Y_MIN)) * height;

        points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        
        if (i === 0) {
            pathData = `M ${x.toFixed(1)} ${y.toFixed(1)}`;
        } else {
            pathData += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
        }
    }

    // 闭合路径，底部边界也对齐到 drawWidth
    pathData += ` L ${drawWidth} ${height} L ${(drawWidth - ((currentData.length - 1) * stepX)).toFixed(1)} ${height} Z`;

    wavePath.setAttribute('d', pathData);
    waveStroke.setAttribute('points', points.join(' '));

    if (waveDot && currentData.length > 0) {
        waveDot.setAttribute('opacity', '1');
        let latestVal = currentData[currentData.length - 1];
        
        if (latestVal > Y_MAX) latestVal = Y_MAX;
        if (latestVal < Y_MIN) latestVal = Y_MIN;
        
        // 【修改核心】：圆心的X坐标固定在 drawWidth，完美避开右侧被裁的命运
        const latestX = drawWidth; 
        const latestY = height - ((latestVal - Y_MIN) / (Y_MAX - Y_MIN)) * height;

        waveDot.setAttribute('cx', latestX.toFixed(1));
        waveDot.setAttribute('cy', latestY.toFixed(1));
    }
}

drawRealtimeWave();


// 增加窗口缩放监听，保证用户在调整浏览器窗口大小时，波形立刻自适应填满
window.addEventListener('resize', drawRealtimeWave);

// --- 3. WebSocket 实时数据接收 ---
const socket = new WebSocket('ws://localhost:8080'); 

socket.onopen = function() { console.log("[WebSocket] 已成功连接到 Python 后端"); };

socket.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);
        
        if (data.relaxed !== undefined) bandConfig["Relaxed & Focused Level"].data.push(data.relaxed);
        if (data.attention !== undefined) bandConfig["Attention"].data.push(data.attention);
        if (data.engagement !== undefined) bandConfig["Engagement"].data.push(data.engagement);

        for (let key in bandConfig) {
            if (bandConfig[key].data.length > MAX_DATA_POINTS) {
                bandConfig[key].data.shift(); 
            }
        }

        const activeDataList = bandConfig[currentActiveBand].data;
        if (activeDataList.length > 0) {
            const latestValue = activeDataList[activeDataList.length - 1];
            const statAmp = document.getElementById('stat-amp');
            if (statAmp) {
                statAmp.innerHTML = `${latestValue.toFixed(1)}<span class="unit">µV</span>`;
            }
        }
        
        drawRealtimeWave();
    } catch (e) {
        console.error("解析 WebSocket 数据出错: ", e);
    }
};

socket.onerror = function(error) { console.log("[WebSocket] 连接出错，请检查 Python 后端是否已启动"); };

// --- 4. 侧边栏与底部导航栏交互 (同步更新) ---
function setActiveTab(tabName) {
    // 移除所有的 active
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    
    // 给对应的左侧和底部都加上 active
    const desktopTab = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
    const mobileTab = document.querySelector(`.nav-tab[data-tab="${tabName}"]`);
    
    if (desktopTab) desktopTab.classList.add('active');
    if (mobileTab) mobileTab.classList.add('active');
    
    console.log('Navigating to: ' + tabName);
}

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        setActiveTab(this.dataset.tab);
    });
});

document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', function() {
        setActiveTab(this.dataset.tab);
    });
});


// --- 5. 普通按钮交互 ---
document.querySelectorAll('.btn:not(.btn-ai-analysis):not(.band-btn)').forEach(btn => {
    btn.addEventListener('click', function() {
        console.log(this.textContent.trim() + ' clicked');
    });
});

// --- 6. 频段切换逻辑 ---
document.querySelectorAll('.band-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.band-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');

        currentActiveBand = this.dataset.band;
        const freqValue = this.dataset.freq;

        const titleEl = document.getElementById('current-stream-name');
        if(titleEl) titleEl.textContent = `${currentActiveBand}`;
        
        const freqEl = document.getElementById('stat-freq');
        if(freqEl) freqEl.innerHTML = `${freqValue}<span class="unit"> Hz</span>`;

        const activeDataList = bandConfig[currentActiveBand].data;
        const ampEl = document.getElementById('stat-amp');
        if (activeDataList.length > 0) {
            const latestValue = activeDataList[activeDataList.length - 1];
            if(ampEl) ampEl.innerHTML = `${latestValue.toFixed(1)}<span class="unit">µV</span>`;
        } else {
            if(ampEl) ampEl.innerHTML = `${this.dataset.amp}<span class="unit">µV</span>`;
        }

        const newColor = bandConfig[currentActiveBand].color;
        const waveStroke = document.getElementById('waveStroke');
        if(waveStroke) waveStroke.setAttribute('stroke', newColor);
        
        const stops = document.querySelectorAll('#waveGradient stop');
        if(stops.length >= 2) {
            stops[0].style.stopColor = newColor;
            stops[1].style.stopColor = newColor;
        }

        drawRealtimeWave();
    });
});

// --- 7. AI Analysis 面板展开逻辑 ---
const btnAiAnalysis = document.getElementById('btn-ai-analysis');
const aiAnalysisPanel = document.getElementById('ai-analysis-panel');

if(btnAiAnalysis && aiAnalysisPanel) {
    btnAiAnalysis.addEventListener('click', function() {
        if (aiAnalysisPanel.style.display === 'none' || aiAnalysisPanel.style.display === '') {
            aiAnalysisPanel.style.display = 'block';
            aiAnalysisPanel.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else {
            aiAnalysisPanel.style.display = 'none';
        }
    });
}

// --- 8. 动态更新进度环 (Circle Progress Ring) ---
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 155; // r=155

function updateProgressRing() {
    const focusValueElement = document.querySelector('.focus-value');
    const progressRing = document.getElementById('progressRing');
    
    if (!focusValueElement || !progressRing) return;
    
    const focusValue = parseInt(focusValueElement.textContent);
    const percentage = Math.max(0, Math.min(100, focusValue)) / 100;
    const offset = CIRCLE_CIRCUMFERENCE * (1 - percentage);
    
    progressRing.style.strokeDashoffset = offset;
}

// 初始更新
updateProgressRing();

// 监听焦点值变化（如果有动态更新）
const focusValueElement = document.querySelector('.focus-value');
if (focusValueElement) {
    const observer = new MutationObserver(updateProgressRing);
    observer.observe(focusValueElement, { childList: true, characterData: true, subtree: true });
}


// --- 9. AI Analysis API 交互逻辑 ---
document.addEventListener('DOMContentLoaded', () => {
    const aiAnalysisBtn = document.getElementById('btn-ai-analysis');
    const aiPanel = document.getElementById('ai-analysis-panel');
    const aiContent = aiPanel.querySelector('.ai-analysis-content');

    // 替换为你的真实 Gemini API Key
    const GEMINI_API_KEY = 'AIzaSyBbveMsXU2aMVsyn23qoyvAXLznnydN8R4'; 

    aiAnalysisBtn.addEventListener('click', async () => {
        
        // 1. 获取当前实时的脑波数据（核心修复部分）
        let brainwaveData = "Here is the current real-time brainwave data from the user:\n";
        
        // 遍历目前配置里的每一项波形，获取最新的数据点
        for (const bandName in bandConfig) {
            const dataArray = bandConfig[bandName].data;
            let currentAmp = 0;
            
            // 如果数组里有数据，就取最后一个（最新接收到的）
            if (dataArray.length > 0) {
                currentAmp = dataArray[dataArray.length - 1].toFixed(3);
            }
            
            brainwaveData += `- ${bandName}: Amplitude ${currentAmp}µV\n`;
        }

        // 2. 显示面板并设置加载动画
        aiPanel.style.display = 'block';
        aiContent.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px; color: #a855f7;">
                <div class="live-dot"></div>
                <p>Synthesizing real-time neural data...</p>
            </div>`;

        // 3. 构建发送给 Gemini 的 Prompt
        const prompt = `
        You are a professional AI Coach for a Neuro-Focus BCI dashboard. 
        Analyze the following real-time brainwave data and provide concise insights.
        
        ${brainwaveData}
        
        Please format your response EXACTLY in this HTML structure, keeping it brief and professional:
        <p><strong>Flow State Analysis:</strong> [Your analysis here based on the data]</p>
        <p><strong>Recommendation:</strong> [Your actionable recommendation here]</p>
        `;

        // 4. 调用 Gemini API
        try {
            const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contents: [{
                        parts: [{ text: prompt }]
                    }],
                    generationConfig: {
                        temperature: 0.7, 
                        maxOutputTokens: 200
                    }
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // 提取大模型返回的文本内容
            const aiResponseText = data.candidates[0].content.parts[0].text;

            // 5. 将结果渲染到面板中
            aiContent.innerHTML = aiResponseText;

        } catch (error) {
            console.error("Error fetching AI analysis:", error);
            aiContent.innerHTML = `<p style="color: #ef4444;"><strong>Connection Error:</strong> Unable to reach the AI Coach. Please check your network or API key.</p>`;
        }
    });
});





// ============================================
// 10. 本地 AI 分类器集成 (Focus/Relax 状态判断)
// ============================================

// 获取各个波形的最新数值的辅助函数
function getLatestBandValue(bandName) {
    const dataArray = bandConfig[bandName].data;
    if (dataArray && dataArray.length > 0) {
        return dataArray[dataArray.length - 1]; // 返回最新收到的那个点
    }
    return 0.0;
}

// 请求本地后端并更新 UI 状态的函数
async function classifyCurrentState() {
    // 1. 组装给后端的数据
    // 注意：你前端叫 Attention/Relaxed，但后端接口要求传入 theta, alpha, beta
    // 这里需要做一个映射 (你可以根据实际科学模型调整对应关系)
    const currentData = {
        "theta": getLatestBandValue("Engagement"),              // 暂时代替 theta
        "alpha": getLatestBandValue("Relaxed & Focused Level"), // 暂时代替 alpha
        "beta":  getLatestBandValue("Attention")                // 暂时代替 beta
    };

    // 如果还没有收到任何数据，先不发请求
    if (currentData.theta === 0 && currentData.alpha === 0 && currentData.beta === 0) {
        return; 
    }

    try {
        // 2. 发送 POST 请求到你的 Python 本地接口
        const response = await fetch('http://127.0.0.1:8000/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentData)
        });

        if (!response.ok) throw new Error("网络响应不正常");

        const result = await response.json();
        
        // 3. 解析后端返回的结果并改变 UI 颜色
        // 假设后端返回格式为: { "status": "focus" } 或 { "state": "relax" }
        // 注意：请将 `result.state` 替换为你后端实际返回的字段名！
        const predictedState = result.state || result.status || result.label; 
        
        // 根据 AI 判断的结果，切换你写好的 StateManager
        if (predictedState === 'focus' || predictedState === 'focused') {
            StateManager.setState(AppState.FOCUS); // 圆环变紫
        } 
        else if (predictedState === 'relax' || predictedState === 'relaxed') {
            StateManager.setState(AppState.RELAX); // 圆环变绿
        } 
        else {
            StateManager.setState(AppState.IDLE);  // 圆环变灰
        }

    } catch (error) {
        // console.error("本地分类接口请求失败 (可能后端没开或跨域):", error);
        // 静默失败，不打断前端渲染
    }
}

// 4. 设置定时器，每 2 秒钟请求一次 AI 分类接口 (不要太快，以免卡顿)
setInterval(classifyCurrentState, 2000);