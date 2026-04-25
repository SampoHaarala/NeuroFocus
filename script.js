// --- 全局配置与数据 ---
const bandConfig = {
    "Relaxed & Focused Level": { color: "#34e7e4", data: [] }, 
    "Attention":               { color: "#ffdd59", data: [] }, 
    "Engagement":              { color: "#0be881", data: [] }  
};

let currentActiveBand = "Relaxed & Focused Level";
const MAX_DATA_POINTS = 600; 
const MAX_AMP_VALUE = 3;   

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

    const currentData = bandConfig[currentActiveBand].data;
    
    if (currentData.length === 0) {
        wavePath.setAttribute('d', 'M 0 100 L 1000 100 Z');
        waveStroke.setAttribute('points', '0,100 1000,100');
        if(waveDot) waveDot.setAttribute('opacity', '0');
        return;
    }

    const width = 1000;
    const height = 100;
    let points = [];
    let pathData = '';

    const stepX = width / (MAX_DATA_POINTS - 1);

    for (let i = 0; i < currentData.length; i++) {
        const x = width - ((currentData.length - 1 - i) * stepX);
        
        let val = currentData[i];
        if (val > MAX_AMP_VALUE) val = MAX_AMP_VALUE;
        if (val < 0) val = 0;
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