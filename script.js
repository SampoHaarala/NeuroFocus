// 1. 时间更新
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

// 2. 生成波形
function generateWaveform() {
    const svg = document.getElementById('waveform');
    const wavePath = document.getElementById('wavePath');
    const waveStroke = document.getElementById('waveStroke');
    
    if (!svg || !wavePath || !waveStroke) return;

    let points = [];
    let pathData = 'M 0 50';
    
    for (let i = 0; i <= 1000; i += 10) {
        const y = 50 + 30 * Math.sin(i / 100) * Math.cos(i / 150);
        points.push(`${i},${y}`);
        pathData += ` L ${i} ${y}`;
    }
    
    pathData += ' L 1000 100 L 0 100 Z';
    wavePath.setAttribute('d', pathData);
    waveStroke.setAttribute('points', points.join(' '));
}
generateWaveform();

// 3. 动画波形移动
let offset = 0;
setInterval(() => {
    offset = (offset + 5) % 1000;
    const svg = document.getElementById('waveform');
    if (svg) {
        svg.style.transform = `translateX(${offset}px)`;
    }
}, 50);

// 4. 左侧导航栏交互
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        this.classList.add('active');
    });
});

// 5. 移动端底部导航栏交互
document.querySelectorAll('.nav-nav').forEach(tab => {
    tab.addEventListener('click', function() {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        this.classList.add('active');
        const tabName = this.dataset.tab;
        console.log('Navigating to: ' + tabName);
    });
});

// 6. 底部固定按钮交互
document.querySelectorAll('.btn:not(.btn-ai-analysis):not(.band-btn)').forEach(btn => {
    btn.addEventListener('click', function() {
        console.log(this.textContent + ' clicked');
    });
});

// 7. 频段切换逻辑 (α, β, θ, δ)
document.querySelectorAll('.band-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        // 移除所有按钮的 active 状态
        document.querySelectorAll('.band-btn').forEach(b => b.classList.remove('active'));
        // 激活当前点击的按钮
        this.classList.add('active');

        // 提取数据并更新界面
        const bandName = this.dataset.band;
        const freqValue = this.dataset.freq;
        const ampValue = this.dataset.amp;

        document.getElementById('current-stream-name').textContent = `${bandName}`;
        document.getElementById('stat-freq').innerHTML = `${freqValue}<span class="unit"> Hz</span>`;
        document.getElementById('stat-amp').innerHTML = `${ampValue}<span class="unit">µV</span>`;
    });
});

// 8. AI Analysis 面板切换逻辑
const btnAiAnalysis = document.getElementById('btn-ai-analysis');
const aiAnalysisPanel = document.getElementById('ai-analysis-panel');

if(btnAiAnalysis && aiAnalysisPanel) {
    btnAiAnalysis.addEventListener('click', function() {
        if (aiAnalysisPanel.style.display === 'none' || aiAnalysisPanel.style.display === '') {
            // 展开
            aiAnalysisPanel.style.display = 'block';
            // 平滑滚动到底部
            aiAnalysisPanel.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else {
            // 收起
            aiAnalysisPanel.style.display = 'none';
        }
    });
}
