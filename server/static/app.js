// hdgAtHome Admin Dashboard JS

const API_BASE = window.location.origin;

// State
let lossHistory = [];
let stepCount = 0;
let lossChart = null;

// Initialize Chart
function initChart() {
    const ctx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Fetch Server Stats
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/`);
        const data = await res.json();
        
        document.getElementById('status-text').textContent = data.status === 'online' ? 'Online' : 'Offline';
        document.getElementById('metric-peers').textContent = data.active_peers || 0;
        
        // Update Peer Table
        const tbody = document.getElementById('peer-table-body');
        const assignments = data.assignments || {};
        
        if (Object.keys(assignments).length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No peers connected</td></tr>`;
        } else {
            tbody.innerHTML = Object.entries(assignments).map(([pid, assign]) => `
                <tr>
                    <td class="peer-id">${pid.substring(0, 8)}...</td>
                    <td><span class="layer-badge">L${assign.start_layer}-${assign.end_layer}</span></td>
                    <td><span class="status-dot" style="display:inline-block;margin-right:0.5rem;"></span>Ready</td>
                    <td>—</td>
                </tr>
            `).join('');
        }
        
        // Update Network Map (Simple)
        updateNetworkMap(assignments);
        
    } catch (e) {
        document.getElementById('status-text').textContent = 'Disconnected';
    }
}

// Fetch Loss from Relay
async function fetchLoss() {
    try {
        const res = await fetch(`${API_BASE}/relay/poll/loss_sink`);
        if (res.status === 200) {
            const text = await res.text();
            if (text) {
                const loss = parseFloat(text);
                if (!isNaN(loss)) {
                    stepCount++;
                    lossHistory.push(loss);
                    
                    // Keep last 100
                    if (lossHistory.length > 100) lossHistory.shift();
                    
                    // Update UI
                    document.getElementById('metric-loss').textContent = loss.toFixed(4);
                    document.getElementById('metric-step').textContent = stepCount;
                    
                    // Update Chart
                    lossChart.data.labels = lossHistory.map((_, i) => i);
                    lossChart.data.datasets[0].data = lossHistory;
                    lossChart.update('none');
                }
            }
        }
    } catch (e) {
        // Ignore polling errors
    }
}

// Simple Network Map using CSS
function updateNetworkMap(assignments) {
    const container = document.getElementById('network-map');
    const peers = Object.entries(assignments);
    
    if (peers.length === 0) {
        container.innerHTML = '<span>No peers connected</span>';
        return;
    }
    
    // Sort by start_layer
    peers.sort((a, b) => a[1].start_layer - b[1].start_layer);
    
    container.innerHTML = `
        <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem;">
            <div style="text-align: center;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #6366f1, #7c3aed); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 700;">IN</div>
                <span style="font-size: 0.7rem; color: #94a3b8;">Input</span>
            </div>
            ${peers.map(([pid, assign]) => `
                <div style="color: #6366f1;">→</div>
                <div style="text-align: center;">
                    <div style="width: 80px; height: 60px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <span style="font-size: 0.7rem; color: #94a3b8;">L${assign.start_layer}-${assign.end_layer}</span>
                        <span style="font-size: 0.65rem; color: #6366f1;">${pid.substring(0,6)}</span>
                    </div>
                    <span style="font-size: 0.7rem; color: #94a3b8;">Peer</span>
                </div>
            `).join('')}
            <div style="color: #6366f1;">→</div>
            <div style="text-align: center;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #22c55e, #16a34a); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 700;">OUT</div>
                <span style="font-size: 0.7rem; color: #94a3b8;">Output</span>
            </div>
        </div>
    `;
}

// Start/Stop Handlers (Placeholder)
document.getElementById('btn-start').addEventListener('click', () => {
    alert('Training control API not yet implemented. Use trainer.py manually.');
});
document.getElementById('btn-stop').addEventListener('click', () => {
    alert('Training control API not yet implemented.');
});

// Init
initChart();
fetchStats();

// Polling
setInterval(fetchStats, 2000);
setInterval(fetchLoss, 500);
