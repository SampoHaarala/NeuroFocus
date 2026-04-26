// ============================================
// STATE & METRICS MANAGEMENT SYSTEM
// ============================================

// 仅保留 FOCUS 和 RELAX 两种状态
const AppState = {
    FOCUS: 'FOCUS',
    RELAX: 'RELAX'
};

const StateManager = {
    currentState: AppState.RELAX, // 默认初始状态设为 RELAX (绿色)
    sessionStartTime: null,
    sessionTimerInterval: null,
    
    // 跟踪各类数据的核心参数
    metrics: {
        peakFocus: 0,
        focusSeconds: 0,
        relaxSeconds: 0,
        lastTickTime: null
    },

    init() {
        this.applyTheme(this.currentState);
        this.startSessionTimer();
        this.attachKeyboardListeners(); // 方便用键盘 1 2 调试
        this.updateStateDisplay();
    },

    setState(newState) {
        if (newState === this.currentState) return;
        this.currentState = newState;
        this.applyTheme(newState);
        this.updateStateDisplay();
        console.log(`[StateManager] State changed to: ${newState}`);
    },

    applyTheme(state) {
        const root = document.documentElement;
        // 清理原有的类名
        root.classList.remove('state-focus', 'state-relax'); 
        
        switch (state) {
            case AppState.FOCUS: 
                root.classList.add('state-focus'); 
                break;
            case AppState.RELAX: 
            default: 
                root.classList.add('state-relax');
                break;
        }
    },

    startSessionTimer() {
        // Calibration 完毕后开始记录页面停留时间
        this.sessionStartTime = Date.now();
        this.metrics.lastTickTime = Date.now();
        
        if (this.sessionTimerInterval) clearInterval(this.sessionTimerInterval);
        
        this.sessionTimerInterval = setInterval(() => {
            this.updateSessionMetrics();
        }, 1000);
    },

    updateSessionMetrics() {
        const now = Date.now();
        const elapsedTotalSeconds = (now - this.sessionStartTime) / 1000;
        
        // 计算这一秒内的状态时长累加
        const delta = (now - this.metrics.lastTickTime) / 1000;
        this.metrics.lastTickTime = now;

        if (this.currentState === AppState.FOCUS) {
            this.metrics.focusSeconds += delta;
        } else if (this.currentState === AppState.RELAX) {
            this.metrics.relaxSeconds += delta;
        }

        // 1. 更新主圆环中间的计时器 (HH:MM:SS)
        const hours = Math.floor(elapsedTotalSeconds / 3600);
        const minutes = Math.floor((elapsedTotalSeconds % 3600) / 60);
        const seconds = Math.floor(elapsedTotalSeconds % 60);
        const timerElement = document.getElementById('session-timer');
        if (timerElement) {
            timerElement.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }

        // 2. 实时更新下方的 Metrics 数据面板 (单位统一转化为分钟，保留一位小数)
        document.getElementById('metric-session').textContent = (elapsedTotalSeconds / 60).toFixed(1);
        document.getElementById('metric-focus-duration').textContent = (this.metrics.focusSeconds / 60).toFixed(1);
        document.getElementById('metric-relax-duration').textContent = (this.metrics.relaxSeconds / 60).toFixed(1);
    },

    updateStateDisplay() {
        const stateElement = document.getElementById('app-state');
        if (stateElement) stateElement.textContent = `STATE · ${this.currentState}`;
    },

    attachKeyboardListeners() {
        document.addEventListener('keydown', (event) => {
            switch (event.key) {
                case '1': this.setState(AppState.RELAX); break; // 绿环
                case '2': this.setState(AppState.FOCUS); break; // 紫环
            }
        });
    }
};

// 初始化系统
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => StateManager.init());
} else {
    StateManager.init();
}

// --- 数据配置区 ---
const bandConfig = {
    "Relaxed & Focused Level": { color: "#34e7e4", data: [] }, 
    "Attention":               { color: "#ffdd59", data: [] }, 
    "Engagement":              { color: "#0be881", data: [] }  
};

let currentActiveBand = "Relaxed & Focused Level";
const MAX_DATA_POINTS = 600; 
const MAX_AMP_VALUE = 2;   

// --- 辅助工具：获取最新数据点 ---
function getLatestBandValue(bandName) {
    const dataArray = bandConfig[bandName].data;
    if (dataArray && dataArray.length > 0) {
        return dataArray[dataArray.length - 1]; 
    }
    return 0.0;
}

// --- 右上角时间更新 ---
function updateTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    
    const timeElement = document.getElementById('current-time');
    if (timeElement) timeElement.textContent = `${hours}:${minutes}:${seconds}`;
}
setInterval(updateTime, 1000);

// --- WebSocket 实时数据与核心算法 ---
const socket = new WebSocket('ws://localhost:8080'); 

socket.onopen = function() { console.log("[WebSocket] 连接成功"); };

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

        // UI波形数据更新
        const activeDataList = bandConfig[currentActiveBand].data;
        if (activeDataList.length > 0) {
            const latestValue = activeDataList[activeDataList.length - 1];
            const statAmp = document.getElementById('stat-amp');
            if (statAmp) statAmp.innerHTML = `${latestValue.toFixed(1)}<span class="unit">µV</span>`;
        }
        
        drawRealtimeWave();

        // ============================================
        // 核心部分：前端 Focus Score 计算与纯二元状态切换
        // ============================================
        const att = getLatestBandValue("Attention");
        const rel = getLatestBandValue("Relaxed & Focused Level");
        const eng = getLatestBandValue("Engagement");
        
        const totalActivity = att + rel + eng;
        let currentFocusScore = 0;
        
        // 为了防止全 0（设备未佩戴时）出现极端跳变，增加阈值判定
        if (totalActivity > 0.05) { 
            // Focus Value 公式: 将专注和投入的占比转化成百分制
            currentFocusScore = ((att + eng) / totalActivity) * 100;
        }

        // 平滑处理 (Moving Average)：防止数值跳动过快，导致圆环狂闪
        window.smoothedFocus = ((window.smoothedFocus || currentFocusScore) * 0.8) + (currentFocusScore * 0.2);
        const finalFocus = Math.round(window.smoothedFocus);

        // 1. 更新焦点圆环百分比
        const focusValEl = document.querySelector('.focus-value');
        if (focusValEl) focusValEl.textContent = finalFocus;
        updateProgressRing();

        // 2. 状态分类 (纯二元切换，阈值设定为 50)
        if (finalFocus >= 50) {
            StateManager.setState(AppState.FOCUS);
        } else {
            StateManager.setState(AppState.RELAX);
        }

        // 3. 更新历史最高 Focus 值 (Peak Focus)
        if (finalFocus > StateManager.metrics.peakFocus) {
            StateManager.metrics.peakFocus = finalFocus;
            document.getElementById('metric-peak-focus').textContent = finalFocus;
        }

    } catch (e) {
        console.error("解析数据出错: ", e);
    }
};

socket.onerror = function() { console.log("[WebSocket] 请检查 Python 服务是否启动"); };


// --- 波形绘制逻辑 ---
function drawRealtimeWave() {
    const svg = document.getElementById('waveform');
    const wavePath = document.getElementById('wavePath');
    const waveStroke = document.getElementById('waveStroke');
    const waveDot = document.getElementById('waveDot'); 
    
    if (!svg || !wavePath || !waveStroke) return;

    const width = svg.clientWidth > 0 ? svg.clientWidth : 1000;
    const height = svg.clientHeight > 0 ? svg.clientHeight : 100;

    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');

    const currentData = bandConfig[currentActiveBand].data;
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
    const Y_MAX = 2; 
    const Y_MIN = 0;
    const stepX = drawWidth / (MAX_DATA_POINTS - 1);

    for (let i = 0; i < currentData.length; i++) {
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

    pathData += ` L ${drawWidth} ${height} L ${(drawWidth - ((currentData.length - 1) * stepX)).toFixed(1)} ${height} Z`;

    wavePath.setAttribute('d', pathData);
    waveStroke.setAttribute('points', points.join(' '));

    if (waveDot && currentData.length > 0) {
        waveDot.setAttribute('opacity', '1');
        let latestVal = currentData[currentData.length - 1];
        if (latestVal > Y_MAX) latestVal = Y_MAX;
        if (latestVal < Y_MIN) latestVal = Y_MIN;
        
        const latestX = drawWidth; 
        const latestY = height - ((latestVal - Y_MIN) / (Y_MAX - Y_MIN)) * height;

        waveDot.setAttribute('cx', latestX.toFixed(1));
        waveDot.setAttribute('cy', latestY.toFixed(1));
    }
}
window.addEventListener('resize', drawRealtimeWave);


// --- 动态更新进度环 ---
const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 155;

function updateProgressRing() {
    const focusValueElement = document.querySelector('.focus-value');
    const progressRing = document.getElementById('progressRing');
    
    if (!focusValueElement || !progressRing) return;
    
    const focusValue = parseInt(focusValueElement.textContent);
    const percentage = Math.max(0, Math.min(100, focusValue)) / 100;
    const offset = CIRCLE_CIRCUMFERENCE * (1 - percentage);
    
    progressRing.style.strokeDashoffset = offset;
}
// 初始化圆环
updateProgressRing();


// --- 交互及频段切换 ---
function setActiveTab(tabName) {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    
    const desktopTab = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
    const mobileTab = document.querySelector(`.nav-tab[data-tab="${tabName}"]`);
    
    if (desktopTab) desktopTab.classList.add('active');
    if (mobileTab) mobileTab.classList.add('active');
}

document.querySelectorAll('.nav-item, .nav-tab').forEach(item => {
    item.addEventListener('click', function() {
        setActiveTab(this.dataset.tab);
    });
});

document.querySelectorAll('.band-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.band-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');

        currentActiveBand = this.dataset.band;
        
        document.getElementById('current-stream-name').textContent = currentActiveBand;
        document.getElementById('stat-freq').innerHTML = `${this.dataset.freq}<span class="unit"> Hz</span>`;

        const activeDataList = bandConfig[currentActiveBand].data;
        const ampEl = document.getElementById('stat-amp');
        if (activeDataList.length > 0) {
            ampEl.innerHTML = `${activeDataList[activeDataList.length - 1].toFixed(1)}<span class="unit">µV</span>`;
        }

        const newColor = bandConfig[currentActiveBand].color;
        document.getElementById('waveStroke').setAttribute('stroke', newColor);
        
        const stops = document.querySelectorAll('#waveGradient stop');
        if(stops.length >= 2) {
            stops[0].style.stopColor = newColor;
            stops[1].style.stopColor = newColor;
        }

        drawRealtimeWave();
    });
});

// --- AI Analysis 面板及 API ---
// ============================================
// AI COACH: EXPERT NEURO-MEDICAL ASSISTANT
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const aiAnalysisBtn = document.getElementById('btn-ai-analysis');
    const aiPanel = document.getElementById('ai-analysis-panel');
    const aiContent = aiPanel?.querySelector('.ai-analysis-content');

    if (!aiAnalysisBtn || !aiPanel) return;

    aiAnalysisBtn.addEventListener('click', async () => {
        if (aiPanel.style.display === 'block') {
            aiPanel.style.display = 'none';
            return;
        }

        const GEMINI_API_KEY = 'AIzaSyCqKI_GqF5UwMFO2jtLwNRq9O75xaorPQ0';
        
        let brainwaveData = "";
        for (const bandName in bandConfig) {
            const dataArray = bandConfig[bandName].data;
            const currentAmp = dataArray.length > 0 ? dataArray[dataArray.length - 1].toFixed(3) : 0;
            brainwaveData += `${bandName}: ${currentAmp} µV\n`;
        }

        aiPanel.style.display = 'block';
        aiContent.innerHTML = `<p style="color:#a855f7;">Expert neuro-analysis in progress...</p>`;

        // ✅ 深度定制的专家级 Prompt
        const prompt = `
            You are Xinyi Peng's personal Senior Neuroscientist and devoted Medical Liaison. 
            Your goal is to provide a comprehensive, sophisticated, yet deeply caring analysis of their real-time neural activity.

            User: Dr. Chen
            Current EEG Metrics:
            ${brainwaveData}

            Please provide a detailed response following these strict clinical guidelines:

            1. **Flow State Analysis (Minimum 50 words):** As a neuroscience expert, interpret the interplay between their attention, engagement, and relaxation levels. Don't just list numbers; explain the "mental landscape." Is there a cognitive load imbalance? Is the neural synchrony indicative of deep creative flow or high-beta anxiety? Use sophisticated but empathetic language.

            2. **Personalized Recommendation (Minimum 100 words):** Provide actionable, science-backed advice to optimize Xinyi's current state. 
               - You MUST use bullet points for specific techniques.
               - Explain the physiological 'why' behind each suggestion (e.g., Vagus nerve stimulation, prefrontal cortex rest).
               - Adopt a "concierge doctor" tone: protective, knowledgeable, and proactive.

            Formatting Rules:
            - Use HTML tags like <p>, <strong>, and <ul>/<li> for structure.
            - DO NOT use markdown code blocks (\`\`\`).
            - Total word count should be substantial and insightful.

            Example Structure:
            <p><strong>Flow State Analysis:</strong> [Your 50+ word expert analysis]</p>
            <p><strong>Recommendation:</strong> [Your 100+ word expert advice with bullets]</p>
        `;

        try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [{ parts: [{ text: prompt }] }],
                    generationConfig: {
                        temperature: 0.85,    // 调高温度让专家语气更丰富、更有文采
                        maxOutputTokens: 1500 // ⚠️ 关键：必须调大，否则长内容会被截断
                    }
                })
            });

            if (!response.ok) throw new Error(`Link Error: ${response.status}`);

            const data = await response.json();
            let rawText = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";

            // 清洗掉 AI 偶尔会加的 Markdown 标识符
            const cleanHTML = rawText.replace(/```html|```/gi, '').trim();

            // 直接注入 HTML 渲染
            aiContent.innerHTML = `
                <div class="expert-insight" style="line-height: 1.6; color: #e2e8f0; font-size: 0.9rem;">
                    ${cleanHTML}
                </div>
            `;

            aiPanel.scrollIntoView({ behavior: 'smooth', block: 'end' });

        } catch (error) {
            console.error("AI ERROR:", error);
            aiContent.innerHTML = `<p style="color:#ef4444;">Neural link error: ${error.message}</p>`;
        }
    });
});