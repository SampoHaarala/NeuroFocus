// Update time
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

// Generate waveform
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

// Animate waveform
let offset = 0;
setInterval(() => {
    offset = (offset + 5) % 1000;
    const svg = document.getElementById('waveform');
    if (svg) {
        svg.style.transform = `translateX(${offset}px)`;
    }
}, 50);

// Navigation interaction (Sidebar)
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        this.classList.add('active');
    });
});

// Bottom navigation tabs (Mobile)
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', function() {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        this.classList.add('active');
        const tabName = this.dataset.tab;
        console.log('Navigating to: ' + tabName);
    });
});

// Button interactions
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function() {
        console.log(this.textContent + ' clicked');
    });
});
