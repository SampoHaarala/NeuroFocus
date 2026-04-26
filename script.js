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

    // 【修复核心1】动态获取 SVG 当前实际渲染的像素宽和高
    const rect = svg.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 1000;
    const height = rect.height > 0 ? rect.height : 100;

    // 【修复核心2】动态更新 viewBox，将内部坐标系与物理尺寸 1:1 映射，消除左右留白
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    const currentData = bandConfig[currentActiveBand].data;
    
    if (currentData.length === 0) {
        wavePath.setAttribute('d', `M 0 ${height} L ${width} ${height} Z`);
        waveStroke.setAttribute('points', `0,${height} ${width},${height}`);
        if(waveDot) waveDot.setAttribute('opacity', '0');
        return;
    }

    let points = [];
    let pathData = '';

    // 【修复核心3】利用真实的动态屏幕宽度来计算每一个数据点的 X 轴步长
    const stepX = width / (MAX_DATA_POINTS - 1);

    for (let i = 0; i < currentData.length; i++) {
        const x = width - ((currentData.length - 1 - i) * stepX);
        
        let val = currentData[i];
        if (val > MAX_AMP_VALUE) val = MAX_AMP_VALUE;
        if (val < 0) val = 0;
        
        // 【修复核心4】利用真实的动态高度来映射 Y 坐标
        const y = height - (val / MAX_AMP_VALUE) * height;

        points.push(`${x},${y}`);
        
        if (i === 0) {
            pathData = `M ${x} ${y}`;
        } else {
            pathData += ` L ${x} ${y}`;
        }
    }

    pathData += ` L ${width} ${height} L ${width - ((currentData.length - 1) * stepX)} ${height} Z`;

    wavePath.setAttribute('d', pathData);
    waveStroke.setAttribute('points', points.join(' '));

    if (waveDot && currentData.length > 0) {
        waveDot.setAttribute('opacity', '1');
        let latestVal = currentData[currentData.length - 1];
        if (latestVal > MAX_AMP_VALUE) latestVal = MAX_AMP_VALUE;
        if (latestVal < 0) latestVal = 0;
        
        const latestX = width; 
        const latestY = height - (latestVal / MAX_AMP_VALUE) * height;

        waveDot.setAttribute('cx', latestX);
        waveDot.setAttribute('cy', latestY);
    }
}

drawRealtimeWave();

// 【修复核心5】增加窗口缩放监听，保证用户在调整浏览器窗口大小时，波形立刻自适应填满
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

// --- 5. 动态更新进度环 (Circle Progress Ring) ---
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