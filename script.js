// Update time
function updateTime() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    document.getElementById('current-time').textContent = `${hours}:${minutes}:${seconds}`;
}
setInterval(updateTime, 1000);

// Generate waveform
function generateWaveform() {
    const svg = document.getElementById('waveform');
    const wavePath = document.getElementById('wavePath');
    const waveStroke = document.getElementById('waveStroke');
    
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
    svg.style.transform = `translateX(${offset}px)`;
}, 50);

// Navigation interaction
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', function() {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        this.classList.add('active');
    });
});

// Button interactions
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('click', function() {
        console.log(this.textContent + ' clicked');
    });
});
