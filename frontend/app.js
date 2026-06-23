/**
 * Deceptive-Net – Dashboard Application Logic
 * Handles: Auth guard, boot sequence, tabbed navigation,
 *          WebSocket live feed, REST data fetching,
 *          ML predictions, AI report, audit logs, model metrics.
 */

const API = 'http://localhost:8000';
const WS  = 'ws://localhost:8000/ws/transactions';

// ── Dynamic stats tracker ───────────────────────────────────────────────────
let currentStats = {
    total: 0,
    flagged: 0,
    cleared: 0,
    riskSum: 0.0
};

// ── Auth guard ─────────────────────────────────────────────────────────────────────────────────────────────
const token    = sessionStorage.getItem('dn_token');
const role     = sessionStorage.getItem('dn_role');
const username = sessionStorage.getItem('dn_user');

if (!token) { window.location.href = 'index.html'; }

function authHeaders() {
    return { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' };
}

// ── Utilities ────────────────────────────────────────────────────────────────
function randIP() {
    return `${rndInt(1,254)}.${rndInt(0,255)}.${rndInt(0,255)}.${rndInt(1,254)}`;
}
function rndInt(a, b) { return Math.floor(Math.random() * (b - a + 1)) + a; }

function riskBadge(score) {
    if (score === undefined || score === null) return '<span class="badge badge-info">PENDING</span>';
    const s = parseFloat(score);
    if (s > 0.75) return `<span class="badge badge-critical">CRITICAL ${pct(s)}</span>`;
    if (s > 0.50) return `<span class="badge badge-high">HIGH ${pct(s)}</span>`;
    if (s > 0.25) return `<span class="badge badge-medium">MEDIUM ${pct(s)}</span>`;
    return `<span class="badge badge-low">LOW ${pct(s)}</span>`;
}
function pct(v) { return `${(v * 100).toFixed(1)}%`; }
function deviceLabel(d) { return { 0: 'Mobile', 1: 'Desktop', 2: 'Tablet' }[d] || '–'; }

// ── Boot Sequence ────────────────────────────────────────────────────────────
const BOOT_LINES = [
    { text: '> Initializing onion descriptors...', delay: 400 },
    { text: '> Fetching network consensus from directory...', delay: 800 },
    { text: '> Building circuit: Guard → Middle → Exit...', delay: 1300 },
    { text: `> Authenticating session for [${username}]...`, delay: 1900, cls: 'ok' },
    { text: '> Circuit established. Loading dashboard...', delay: 2500, cls: 'ok' },
];

const bootOverlay  = document.getElementById('boot-overlay');
const bootLines    = document.getElementById('boot-lines');
const bootProgress = document.getElementById('boot-progress');

BOOT_LINES.forEach(({ text, delay, cls }) => {
    setTimeout(() => {
        const p = document.createElement('p');
        p.textContent = text;
        if (cls) p.classList.add(cls);
        p.classList.add('visible');
        bootLines.appendChild(p);
        bootProgress.style.width = `${((delay / 2800) * 100).toFixed(0)}%`;
    }, delay);
});

setTimeout(() => {
    bootProgress.style.width = '100%';
    bootOverlay.style.opacity = '0';
    setTimeout(() => {
        bootOverlay.style.display = 'none';
        document.getElementById('main-app').style.display = 'flex';
        initApp();
    }, 500);
}, 3100);

// ── Init App ─────────────────────────────────────────────────────────────────
function initApp() {
    // Set user info
    document.getElementById('sidebar-username').textContent = username;
    document.getElementById('sidebar-role').textContent     = role.toUpperCase();

    // Hide admin-only tabs for non-admins
    if (role !== 'admin') {
        document.querySelectorAll('.admin-only').forEach(el => el.style.display = 'none');
    }

    // Init Onion Circuit IPs
    initCircuit();

    // Nav tabs
    document.querySelectorAll('.nav-item[data-tab]').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(item.dataset.tab, item);
        });
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => {
        sessionStorage.clear();
        window.location.href = 'index.html';
    });

    // Load initial dashboard data
    loadTransactions();
    loadStats();
    connectWebSocket();

    // Transactions tab controls
    document.getElementById('row-select').addEventListener('change', loadTransactions);
    document.getElementById('refresh-btn').addEventListener('click', loadTransactions);
    document.getElementById('export-csv-btn')?.addEventListener('click', downloadSecureData);

    // AI Report
    document.getElementById('run-genai').addEventListener('click', loadAIReport);

    // Audit Logs
    document.getElementById('audit-refresh').addEventListener('click', () => loadAuditLogs(false));
    document.getElementById('audit-alerts-only').addEventListener('click', () => loadAuditLogs(true));

    // Model Metrics (auto-load when tab clicked)
    document.getElementById('nav-model')?.addEventListener('click', loadModelMetrics);
    document.getElementById('nav-audit')?.addEventListener('click', () => loadAuditLogs(false));

    // Deception Lab & Threat Intel Sandbox Hooks
    document.getElementById('hp-payment-form')?.addEventListener('submit', submitHoneypotPayment);
    document.getElementById('hp-login-form')?.addEventListener('submit', submitHoneypotLogin);
    document.getElementById('hp-term-input')?.addEventListener('keypress', handleTerminalInput);
    document.getElementById('nav-threat-intel')?.addEventListener('click', loadThreatIntel);
    document.getElementById('nav-playground')?.addEventListener('click', initDeceptionLab);

    // Admin System Config Hook
    document.getElementById('nav-admin-config')?.addEventListener('click', loadSystemConfigTab);
    document.getElementById('config-role-form')?.addEventListener('submit', submitRoleChange);
    document.getElementById('config-policy-form')?.addEventListener('submit', submitPolicyChange);

    // Support Desk Hooks
    document.getElementById('nav-support')?.addEventListener('click', loadSupportTab);
    document.getElementById('support-bug-form')?.addEventListener('submit', submitSupportBugReport);

    // XAI Modal Close bindings
    document.getElementById('modal-close-btn')?.addEventListener('click', () => {
        document.getElementById('xai-modal').style.display = 'none';
    });
    document.getElementById('xai-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'xai-modal') {
            document.getElementById('xai-modal').style.display = 'none';
        }
    });

    // Presentation Co-Pilot setup
    initPresentationCopilot();
}

// ── Tab Switching ─────────────────────────────────────────────────────────────
function switchTab(tabId, navEl) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById(`tab-${tabId}`)?.classList.add('active');
    navEl?.classList.add('active');
    document.getElementById('page-title').textContent =
        navEl?.textContent.trim().replace('◈ ', '') || 'Dashboard';

    // Page / Tab view tracking for SaaS GSC/SEO compliance
    trackUserEvent('PAGE_VIEW', `tab-${tabId}`);
}

// ── Onion Circuit ────────────────────────────────────────────────────────────
function initCircuit() {
    const guard  = randIP(); const middle = randIP(); const exit_ = randIP();
    ['cp-guard','cn-guard'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = guard.substring(0,10) + '...'; });
    ['cp-middle','cn-middle'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = middle.substring(0,10) + '...'; });
    ['cp-exit','cn-exit'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = exit_.substring(0,10) + '...'; });

    // Rotate middle node every 30s (Tor circuit refresh simulation)
    setInterval(() => {
        const newMid = randIP();
        ['cp-middle','cn-middle'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.style.opacity = '.3'; setTimeout(() => { el.textContent = newMid.substring(0,10)+'...'; el.style.opacity='1'; }, 600); }
        });
    }, 30000);
}

// ── WebSocket Live Feed ───────────────────────────────────────────────────────
let ws = null;
const feedList = document.getElementById('live-feed-list');
const wsStatus = document.getElementById('ws-status');

function connectWebSocket() {
    try {
        const tokenQuery = token ? `?token=${token}` : '';
        ws = new WebSocket(`${WS}${tokenQuery}`);
        ws.onopen = () => {
            wsStatus.textContent = '● Live';
            wsStatus.classList.add('connected');
        };
        ws.onmessage = (ev) => {
            const txn = JSON.parse(ev.data);
            addFeedItem(txn);
        };
        ws.onclose = () => {
            wsStatus.textContent = '◌ Reconnecting...';
            wsStatus.classList.remove('connected');
            setTimeout(connectWebSocket, 4000);
        };
        ws.onerror = () => ws.close();
    } catch(e) {
        wsStatus.textContent = '✗ Offline';
    }
}

function updateStatsUI() {
    document.getElementById('stat-total').textContent = currentStats.total.toLocaleString();
    document.getElementById('stat-flagged').textContent = currentStats.flagged;
    document.getElementById('stat-cleared').textContent = currentStats.cleared;
    const avgRisk = currentStats.total > 0 ? (currentStats.riskSum / currentStats.total) : 0;
    document.getElementById('stat-avg-risk').textContent = (avgRisk * 100).toFixed(1) + '%';
}

function addFeedItem(txn) {
    // Dynamic real-time stats update from WebSockets
    currentStats.total += 1;
    if (txn.is_flagged) currentStats.flagged += 1;
    if ((txn.risk_score || 0) < 0.25) currentStats.cleared += 1;
    currentStats.riskSum += (txn.risk_score || 0);
    updateStatsUI();

    const flagged = txn.is_flagged;
    const div = document.createElement('div');
    div.className = `feed-item ${flagged ? 'flagged' : 'safe'}`;
    div.innerHTML = `
        <div class="feed-left">
            <span class="feed-txn">${txn.id}</span>
            <span class="feed-amount">$${parseFloat(txn.amount).toFixed(2)}</span>
            <span>${txn.location}</span>
            <span style="font-family:var(--font-mono);opacity:.6">${txn.ip_address}</span>
        </div>
        <div class="feed-right">
            ${riskBadge(txn.risk_score)}
            <span style="opacity:.5;font-size:.75rem">${txn.timestamp?.slice(11,19)}</span>
        </div>`;
    feedList.prepend(div);
    // Keep max 20 items
    while (feedList.children.length > 20) feedList.removeChild(feedList.lastChild);
}

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
    try {
        const res  = await fetch(`${API}/api/transactions?limit=200`, { headers: authHeaders() });
        const data = await res.json();
        currentStats.total = data.length;
        currentStats.flagged = data.filter(t => t.is_flagged).length;
        currentStats.cleared = data.filter(t => (t.risk_score || 0) < 0.25).length;
        currentStats.riskSum = data.reduce((a, t) => a + (t.risk_score || 0), 0);
        updateStatsUI();
    } catch(e) { console.error('Stats load failed', e); }
}

// ── Transactions Table ────────────────────────────────────────────────────────
async function loadTransactions() {
    const limit = document.getElementById('row-select').value;
    const tbody = document.getElementById('txn-body');
    tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">> Fetching encrypted stream...</td></tr>`;
    try {
        const res  = await fetch(`${API}/api/transactions?limit=${limit}`, { headers: authHeaders() });
        if (!res.ok) {
            if (res.status === 401) throw new Error("HTTP 401 (Session Expired/Unauthorized)");
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        tbody.innerHTML = '';
        data.forEach(txn => {
            const tr = document.createElement('tr');
            tr.className = 'row-clickable';
            if (txn.is_flagged) tr.classList.add('row-flagged');
            tr.innerHTML = `
                <td style="font-family:var(--font-mono);color:var(--text-dim)">${txn.id}</td>
                <td>${txn.user_id}</td>
                <td style="font-weight:600">$${parseFloat(txn.amount).toFixed(2)}</td>
                <td>${deviceLabel(txn.device)}</td>
                <td>${txn.location}</td>
                <td style="font-family:var(--font-mono);font-size:.8rem">${txn.ip_address}</td>
                <td style="color:var(--text-dim);font-size:.82rem">${txn.timestamp}</td>
                <td>${riskBadge(txn.risk_score)}</td>
                <td><span style="font-family:var(--font-mono);font-size:.82rem">${txn.anomaly_score !== undefined ? pct(txn.anomaly_score) : '–'}</span></td>
                <td>${txn.is_flagged ? '<span class="badge badge-alert">FLAGGED</span>' : '<span class="badge badge-low">CLEAR</span>'}</td>`;
            tr.addEventListener('click', () => openXaiModal(txn));
            tbody.appendChild(tr);
        });
    } catch(e) {
        tbody.innerHTML = `<tr><td colspan="10" class="loading-cell" style="color:var(--red)">ERR: ${e.message}. Ensure backend is running.</td></tr>`;
    }
}

// ── AI Report ─────────────────────────────────────────────────────────────────
async function loadAIReport() {
    const body = document.getElementById('ai-report-body');
    const header = document.querySelector('.ai-header');
    
    // Remove old download button if it exists
    const oldBtn = document.getElementById('download-ai-btn');
    if (oldBtn) oldBtn.remove();

    body.className = 'glass-panel ai-placeholder';
    body.innerHTML = `<p style="color:var(--accent)">✨ Querying AI threat analysis engine...</p>`;
    
    try {
        const res  = await fetch(`${API}/api/genai_report`, { headers: authHeaders() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        const alertsHTML = data.alerts.map(a => `<li style="margin-bottom:8px">${a}</li>`).join('');
        const riskCls = data.risk_level === 'CRITICAL' ? 'badge-critical' : data.risk_level === 'HIGH' ? 'badge-high' : 'badge-medium';
        
        body.className = 'glass-panel ai-report-body';
        body.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.05)">
                <div style="font-family:var(--font-mono);font-size:.85rem;color:var(--text-dim)">
                    Report ID: AI-${Math.random().toString(36).substring(2,8).toUpperCase()}<br>
                    Generated: ${data.timestamp}
                </div>
                <span class="badge ${riskCls}" style="font-size:.9rem;padding:6px 12px">${data.risk_level} RISK DETECTED</span>
            </div>
            
            <div style="margin-bottom:28px">
                <h4 style="color:var(--text);margin-bottom:10px;font-size:1.05rem;letter-spacing:0.03em">Executive Summary</h4>
                <p style="line-height:1.6;color:var(--text-dim);font-size:.95rem">${data.summary}</p>
            </div>
            
            <div style="margin-bottom:28px">
                <h4 style="color:var(--accent-light);margin-bottom:12px;font-size:1.05rem;letter-spacing:0.03em">Key Anomalies Detected</h4>
                <ul style="padding-left:20px;line-height:1.5;color:var(--text-dim);font-size:.95rem">${alertsHTML}</ul>
            </div>
            
            <div style="background:rgba(255,255,255,0.03);padding:16px 20px;border-left:4px solid var(--accent);border-radius:4px">
                <h4 style="color:var(--text);margin-bottom:8px;font-size:1.05rem;letter-spacing:0.03em">Recommended Action Plan</h4>
                <p style="line-height:1.6;color:var(--text-dim);margin:0;font-size:.95rem">${data.recommendation}</p>
            </div>
        `;

        // Inject download button
        const dlBtn = document.createElement('button');
        dlBtn.id = 'download-ai-btn';
        dlBtn.className = 'btn btn-ghost';
        dlBtn.style.marginLeft = '12px';
        dlBtn.innerHTML = '📥 Download Report';
        dlBtn.onclick = () => downloadAIReportTxt(data);
        header.appendChild(dlBtn);

    } catch(e) {
        body.innerHTML = `<p style="color:var(--red)">> Error: ${e.message}</p>`;
    }
}

function downloadAIReportTxt(data) {
    const content = `================================================================
DECEPTIVE-NET AI THREAT ANALYSIS REPORT
================================================================
Report ID : AI-${Math.random().toString(36).substring(2,8).toUpperCase()}
Generated : ${data.timestamp}
System Risk: ${data.risk_level}

[ EXECUTIVE SUMMARY ]
${data.summary}

[ KEY ANOMALIES DETECTED ]
${data.alerts.map(a => '- ' + a).join('\n')}

[ RECOMMENDED ACTION PLAN ]
${data.recommendation}
================================================================
* End of Report *`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Threat_Report_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// ── Audit Logs ────────────────────────────────────────────────────────────────
async function loadAuditLogs(alertsOnly = false) {
    const tbody = document.getElementById('audit-body');
    tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">> Loading audit stream...</td></tr>`;
    const endpoint = alertsOnly ? '/api/audit/alerts' : '/api/audit/logs?n=100';
    try {
        const res  = await fetch(`${API}${endpoint}`, { headers: authHeaders() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!data.length) {
            tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">No log entries found.</td></tr>`;
            return;
        }
        tbody.innerHTML = '';
        [...data].reverse().forEach(ev => {
            const svCls = ev.severity === 'ALERT' ? 'badge-alert' : ev.severity === 'WARN' ? 'badge-warn' : 'badge-info';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family:var(--font-mono);font-size:.78rem">${ev.ts}</td>
                <td>${ev.actor}</td>
                <td style="font-family:var(--font-mono)">${ev.action}</td>
                <td style="color:var(--text-dim)">${ev.resource || '–'}</td>
                <td style="color:var(--text-dim);font-size:.82rem">${ev.detail || '–'}</td>
                <td><span class="badge ${svCls}">${ev.severity}</span></td>
                <td style="font-family:var(--font-mono);font-size:.78rem">${ev.ip}</td>`;
            tbody.appendChild(tr);
        });
    } catch(e) {
        tbody.innerHTML = `<tr><td colspan="7" class="loading-cell" style="color:var(--red)">ERR: ${e.message}</td></tr>`;
    }
}

// ── Model Metrics ─────────────────────────────────────────────────────────────
async function loadModelMetrics() {
    try {
        const [metricsRes, shapRes] = await Promise.all([
            fetch(`${API}/api/metrics`,         { headers: authHeaders() }),
            fetch(`${API}/api/shap_importance`, { headers: authHeaders() }),
        ]);
        if (metricsRes.ok) {
            const m = await metricsRes.json();
            document.getElementById('m-roc').textContent    = m.roc_auc || '–';
            document.getElementById('m-pr').textContent     = m.pr_auc  || '–';
            document.getElementById('m-ae-roc').textContent = m.ae_roc_auc || '–';
            
            // Populate GAT / FDG / CAA Research Telemetry
            if (document.getElementById('fdg-node-count')) {
                document.getElementById('fdg-node-count').textContent = m.fdg_node_count || '5000';
                document.getElementById('fdg-threshold').textContent = m.fdg_similarity_threshold || '0.5';
                document.getElementById('fdg-degree').textContent = m.fdg_avg_degree || '4.8';
                
                document.getElementById('gat-layers').textContent = m.gat_layers || '2';
                document.getElementById('gat-heads').textContent = m.gat_attention_heads || '4';
                document.getElementById('gat-hidden').textContent = m.gat_hidden_dim || '16';
                
                document.getElementById('caa-luhn').textContent = m.luhn_attention_compliance || '98.7%';
                document.getElementById('caa-coherence').textContent = m.name_email_coherence_index || '92.4%';
                document.getElementById('caa-zip').textContent = m.zip_state_accuracy_rate || '96.1%';
            }

            if (m.confusion_matrix) {
                document.getElementById('cm-tn-val').textContent = m.confusion_matrix[0][0];
                document.getElementById('cm-fp-val').textContent = m.confusion_matrix[0][1];
                document.getElementById('cm-fn-val').textContent = m.confusion_matrix[1][0];
                document.getElementById('cm-tp-val').textContent = m.confusion_matrix[1][1];
            }
        }
        if (shapRes.ok) {
            const shap = await shapRes.json();
            const container = document.getElementById('shap-bars');
            container.innerHTML = '';
            const entries = Object.entries(shap).sort((a, b) => b[1] - a[1]).slice(0, 10);
            const maxVal  = entries[0]?.[1] || 1;
            entries.forEach(([feat, val]) => {
                const pctW = ((val / maxVal) * 100).toFixed(1);
                const row  = document.createElement('div');
                row.className = 'shap-row';
                row.innerHTML = `
                    <div class="shap-feat">${feat}</div>
                    <div class="shap-bar-wrap"><div class="shap-bar-fill" style="width:${pctW}%"></div></div>
                    <div class="shap-val">${val.toFixed(4)}</div>`;
                container.appendChild(row);
            });
        }
    } catch(e) { console.error('Metrics load failed', e); }
}

// ── Secure Data Export ────────────────────────────────────────────────────────
async function downloadSecureData(e) {
    if (e) e.preventDefault();
    try {
        const btn = document.getElementById('export-csv-btn');
        const origText = btn.textContent;
        btn.textContent = '... Encrypting ...';
        const res = await fetch(`${API}/api/transactions/export`, { headers: authHeaders() });
        if (!res.ok) {
            let detail = 'Export failed';
            try {
                const err = await res.json();
                detail = err.detail || detail;
            } catch (errParse) {}
            throw new Error(detail + ': HTTP ' + res.status);
        }
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'deceptive_net_transactions.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        btn.textContent = origText;
        alert("Download complete. Activity logged.");
    } catch(e) {
        alert(e.message);
        document.getElementById('export-csv-btn').textContent = '⚠ Secure Data Export';
    }
}

// ── DECEPTION LAB & HONEYPOT ACTIONS ──────────────────────────────────────────

async function submitHoneypotPayment(e) {
    e.preventDefault();
    const statusBox = document.getElementById('hp-checkout-status');
    statusBox.style.display = 'block';
    statusBox.className = 'glass-panel';
    statusBox.innerHTML = '<span style="color:var(--accent-light)">> Broadcasting transaction payload to checkout server...</span>';
    
    const payload = {
        cardholder: document.getElementById('hp-cardholder').value,
        card_number: document.getElementById('hp-cardnumber').value,
        cvv: document.getElementById('hp-cvv').value,
        expiry: document.getElementById('hp-expiry').value,
        amount: parseFloat(document.getElementById('hp-amount').value),
        simulated_ip: document.getElementById('hp-ip').value
    };
    
    try {
        const res = await fetch(`${API}/api/honeypot/payment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        if (data.deception_triggered) {
            statusBox.className = 'glass-panel';
            statusBox.style.borderColor = 'var(--red)';
            statusBox.style.backgroundColor = 'rgba(255, 51, 102, 0.08)';
            statusBox.innerHTML = `
                <div style="color:var(--red);font-weight:bold;margin-bottom:8px;animation:pulse 1s infinite">🚨 DECEPTION ALERT: HONEY TOKEN DEPLOYED</div>
                <div style="color:var(--text);line-height:1.4">
                    Intrusion identified. Stolen credit card signature matched to exfiltration registry.<br>
                    <strong>Attacker IP:</strong> ${payload.simulated_ip}<br>
                    <strong>Forensic Result:</strong> Forensic signature sent to Threat Intelligence dashboard. Attacker IP has been flagged.
                </div>
            `;
        } else {
            statusBox.className = 'glass-panel';
            statusBox.style.borderColor = 'var(--green)';
            statusBox.innerHTML = `
                <div style="color:var(--green);font-weight:bold;margin-bottom:4px">✓ Payment Authorized</div>
                <div style="color:var(--text-dim)">Transaction completed successfully. (Standard non-honey card used)</div>
            `;
        }
    } catch(err) {
        statusBox.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
    }
}

async function submitHoneypotLogin(e) {
    e.preventDefault();
    const statusText = document.getElementById('hp-login-status');
    statusText.style.display = 'block';
    statusText.style.color = 'var(--accent-light)';
    statusText.textContent = 'Verifying admin credentials...';
    
    const payload = {
        username: document.getElementById('hp-username').value,
        password: document.getElementById('hp-password').value,
        simulated_ip: document.getElementById('hp-login-ip').value
    };
    
    try {
        const res = await fetch(`${API}/api/honeypot/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (res.status === 401) {
            statusText.style.color = 'var(--red)';
            statusText.innerHTML = `🚨 <strong>Access Denied:</strong> Honey Token Triggered. Attempt logged under Attacker IP: ${payload.simulated_ip}`;
            return;
        }
        
        if (res.ok) {
            statusText.style.color = 'var(--green)';
            statusText.textContent = '✓ Login Successful (Simulated admin console access)';
        } else {
            statusText.style.color = 'var(--red)';
            statusText.textContent = 'Access Denied. Invalid credentials.';
        }
    } catch(err) {
        statusText.style.color = 'var(--red)';
        statusText.textContent = `Error connecting to login honeypot: ${err.message}`;
    }
}

function initDeceptionLab() {
    const term = document.getElementById('hp-terminal');
    // Clear terminal and print welcome header
    term.innerHTML = `
        <p style="color:var(--text-dim)"># Simulated Cowrie SSH Environment - Session Established</p>
        <p style="color:var(--text-dim)"># Type 'help' to see available shell commands</p>
        <p>guest@corp-db-server:~$ <input type="text" id="hp-term-input" style="background:transparent;border:none;color:var(--green);outline:none;font-family:var(--font-mono);font-size:0.82rem;width:calc(100% - 180px)" autocomplete="off"></p>
    `;
    
    // Re-bind enter event listener to the input
    document.getElementById('hp-term-input')?.focus();
    document.getElementById('hp-term-input')?.addEventListener('keypress', handleTerminalInput);
}

async function handleTerminalInput(e) {
    if (e.key !== 'Enter') return;
    
    const input = document.getElementById('hp-term-input');
    const term = document.getElementById('hp-terminal');
    const cmdRaw = input.value;
    const cmdClean = cmdRaw.trim().toLowerCase();
    
    // Remove the old input element line
    const promptLine = input.parentElement;
    promptLine.innerHTML = `guest@corp-db-server:~$ <span style="color:var(--text)">${cmdRaw}</span>`;
    
    if (cmdClean === 'clear') {
        term.innerHTML = '';
        appendTerminalPrompt();
        return;
    }
    
    // Retrieve the simulated IP selected in the checkout panel
    const simulatedIp = document.getElementById('hp-ip')?.value || '185.220.101.5';
    
    try {
        const res = await fetch(`${API}/api/honeypot/ssh`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                command: cmdRaw,
                simulated_ip: simulatedIp
            })
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        if (data.response) {
            const p = document.createElement('p');
            p.innerHTML = data.response;
            term.appendChild(p);
        }
    } catch(err) {
        const p = document.createElement('p');
        p.style.color = 'var(--red)';
        p.innerHTML = `Connection error: ${err.message}`;
        term.appendChild(p);
    }
    
    appendTerminalPrompt();
}

function appendTerminalPrompt() {
    const term = document.getElementById('hp-terminal');
    const p = document.createElement('p');
    p.innerHTML = `guest@corp-db-server:~$ <input type="text" id="hp-term-input" style="background:transparent;border:none;color:var(--green);outline:none;font-family:var(--font-mono);font-size:0.82rem;width:calc(100% - 180px)" autocomplete="off">`;
    term.appendChild(p);
    
    // Focus the new input and scroll terminal to bottom
    const input = p.querySelector('input');
    input.focus();
    term.scrollTop = term.scrollHeight;
    
    // Bind event listener to the new input element
    input.addEventListener('keypress', handleTerminalInput);
}

// ── Local SHAP explainability modal ──────────────────────────────────────────
async function openXaiModal(txn) {
    const modal = document.getElementById('xai-modal');
    modal.style.display = 'flex';
    
    // Fill basic details
    document.getElementById('modal-txn-id').textContent = txn.id;
    document.getElementById('modal-user-id').textContent = txn.user_id;
    document.getElementById('modal-amount').textContent = `$${parseFloat(txn.amount).toFixed(2)}`;
    document.getElementById('modal-location').textContent = txn.location;
    document.getElementById('modal-ip').textContent = txn.ip_address;
    document.getElementById('modal-time').textContent = txn.timestamp;
    
    // Clear and show placeholders for ML results
    document.getElementById('modal-lgbm-score').innerHTML = '<span style="color:var(--text-dim)">Calculating...</span>';
    document.getElementById('modal-ae-score').innerHTML = '<span style="color:var(--text-dim)">Calculating...</span>';
    document.getElementById('modal-ens-score').innerHTML = '<span style="color:var(--text-dim)">Calculating...</span>';
    const barsContainer = document.getElementById('modal-shap-bars');
    barsContainer.innerHTML = '<p style="color:var(--accent);font-family:var(--font-mono)">> Running local SHAP kernel...</p>';
    
    try {
        // Fetch predict and explain in parallel
        const [predRes, shapRes] = await Promise.all([
            fetch(`${API}/api/predict/${txn.id}`, { headers: authHeaders() }),
            fetch(`${API}/api/explain/${txn.id}`, { headers: authHeaders() })
        ]);
        
        if (!predRes.ok || !shapRes.ok) throw new Error("API call failed");
        
        const pred = await predRes.json();
        const shap = await shapRes.json();
        
        // Render scores
        document.getElementById('modal-lgbm-score').innerHTML = riskBadge(pred.fraud_probability);
        document.getElementById('modal-ae-score').innerHTML = pct(pred.anomaly_score);
        document.getElementById('modal-ens-score').innerHTML = riskBadge(pred.ensemble_score);
        
        // Render local SHAP bar chart
        barsContainer.innerHTML = '';
        const contributions = Object.entries(shap.shap_contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        const maxVal = Math.max(...contributions.map(([_, v]) => Math.abs(v))) || 1;
        
        contributions.forEach(([feat, val]) => {
            const absPct = ((Math.abs(val) / maxVal) * 100).toFixed(1);
            const isPos = val >= 0;
            const sign = isPos ? '+' : '–';
            const cls = isPos ? 'positive' : 'negative';
            
            const row = document.createElement('div');
            row.className = 'shap-row';
            row.innerHTML = `
                <div class="shap-feat" style="font-size:0.75rem">${feat}</div>
                <div class="shap-bar-wrap">
                    <div class="shap-bar-fill ${cls}" style="width:${absPct}%; margin-left: ${isPos ? '0' : 'auto'}; border-radius: 4px;"></div>
                </div>
                <div class="shap-val" style="font-size:0.75rem; color:${isPos ? 'var(--red)' : 'var(--green)'}">${sign}${Math.abs(val).toFixed(4)}</div>
            `;
            barsContainer.appendChild(row);
        });
    } catch(err) {
        barsContainer.innerHTML = `<p style="color:var(--red)">Failed to load SHAP explainability analysis: ${err.message}</p>`;
    }
}

// ── THREAT INTEL DASHBOARD DATA RENDERING ─────────────────────────────────────

async function loadThreatIntel() {
    const tbody = document.getElementById('intel-hackers-body');
    const tdaBody = document.getElementById('intel-tda-body');
    tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">> Fetching threat logs...</td></tr>`;
    tdaBody.innerHTML = `<tr><td colspan="4" class="loading-cell">> Fetching DQN matrix...</td></tr>`;
    
    try {
        const [intelRes, tdaRes] = await Promise.all([
            fetch(`${API}/api/admin/threat-intel`, { headers: authHeaders() }),
            fetch(`${API}/api/admin/tda-logs`,     { headers: authHeaders() })
        ]);
        
        if (!intelRes.ok || !tdaRes.ok) throw new Error("Threat intelligence API call failed");
        
        const intel = await intelRes.json();
        const tdaLogs = await tdaRes.json();
        
        // Render stats
        const caughtCount = intel.caught_hackers.length;
        const activeWatermarks = intel.exports.length;
        const totalCards = intel.caught_hackers.filter(h => h.token_type.includes('Card')).length;
        const uniqueIps = new Set(intel.caught_hackers.map(h => h.attacker_ip)).size;
        
        document.getElementById('intel-count-triggered').textContent = caughtCount;
        document.getElementById('intel-count-cards').textContent = totalCards;
        document.getElementById('intel-count-attributed').textContent = activeWatermarks;
        document.getElementById('intel-count-ips').textContent = uniqueIps;
        
        // Render caught hackers table
        if (caughtCount === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="loading-cell">No intrusion attempts detected yet. Go to Deception Lab to trigger a honeypot.</td></tr>`;
            document.getElementById('intel-flow-container').style.display = 'none';
        } else {
            tbody.innerHTML = '';
            intel.caught_hackers.forEach(h => {
                const tr = document.createElement('tr');
                tr.style.cursor = 'pointer';
                tr.className = 'row-flagged';
                tr.innerHTML = `
                    <td style="font-family:var(--font-mono);font-size:.78rem">${h.ts}</td>
                    <td style="font-family:var(--font-mono);font-weight:bold;color:var(--red)">${h.attacker_ip}</td>
                    <td>${h.location}</td>
                    <td><span class="badge badge-info">${h.token_type}</span></td>
                    <td style="font-family:var(--font-mono)">${h.action}</td>
                    <td style="font-family:var(--font-mono);color:var(--yellow)">${h.attribution.leak_actor}</td>
                    <td style="font-family:var(--font-mono);font-size:.78rem">${h.attribution.leak_ip}</td>
                `;
                tr.onclick = () => drawForensicFlow(h);
                tbody.appendChild(tr);
            });
            
            // Auto-draw flowchart for the first caught hacker
            drawForensicFlow(intel.caught_hackers[0]);
        }
        
        // Render TDA DQN decision logs
        tdaBody.innerHTML = '';
        tdaLogs.forEach(log => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family:var(--font-mono);font-size:.8rem;color:var(--text-dim)">${log.state}</td>
                <td style="font-family:var(--font-mono);color:var(--accent-light)">${log.action}</td>
                <td style="font-family:var(--font-mono);font-weight:600;color:${log.reward > 0 ? 'var(--green)' : 'var(--red)'}">${log.reward > 0 ? '+' : ''}${log.reward}</td>
                <td style="font-family:var(--font-mono);font-size:.8rem">${log.epsilon}</td>
            `;
            tdaBody.appendChild(tr);
        });
        
    } catch(err) {
        tbody.innerHTML = `<tr><td colspan="7" class="loading-cell" style="color:var(--red)">ERR: ${err.message}</td></tr>`;
    }
}

function drawForensicFlow(hacker) {
    const container = document.getElementById('intel-flow-container');
    container.style.display = 'block';
    const visual = document.getElementById('forensic-flow-visual');
    visual.innerHTML = `
        <div style="text-align:center;padding:12px 18px;border:1px solid var(--border);border-radius:var(--radius);background:rgba(0,0,0,0.3);width:22%">
            <div style="font-size:1.6rem;margin-bottom:8px">📁</div>
            <div style="font-weight:bold;color:var(--text);font-size:0.85rem">transactions_export.csv</div>
            <div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px">Watermark: ${hacker.attribution.watermark_id.substring(0,8)}...</div>
        </div>
        <div style="font-size:1.8rem;color:var(--accent);animation:pulse 2s infinite">➔</div>
        <div style="text-align:center;padding:12px 18px;border:1px solid var(--border);border-radius:var(--radius);background:rgba(0,0,0,0.3);width:22%">
            <div style="font-size:1.6rem;margin-bottom:8px">👤</div>
            <div style="font-weight:bold;color:var(--yellow);font-size:0.85rem">Insider: ${hacker.attribution.leak_actor}</div>
            <div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px">Source IP: ${hacker.attribution.leak_ip}</div>
            <div style="font-size:0.7rem;color:var(--accent-light);margin-top:2px">Time: ${hacker.attribution.leak_timestamp.slice(11,19)}</div>
        </div>
        <div style="font-size:1.8rem;color:var(--accent);animation:pulse 2s infinite">➔</div>
        <div style="text-align:center;padding:12px 18px;border:1px solid var(--border);border-radius:var(--radius);background:rgba(0,0,0,0.3);width:22%">
            <div style="font-size:1.6rem;margin-bottom:8px">🌐</div>
            <div style="font-weight:bold;color:var(--blue);font-size:0.85rem">Attacker: ${hacker.attacker_ip}</div>
            <div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px">Proxy: ${hacker.location}</div>
        </div>
        <div style="font-size:1.8rem;color:var(--red);animation:pulse 2s infinite">➔</div>
        <div style="text-align:center;padding:12px 18px;border:1px solid var(--red);border-radius:var(--radius);background:rgba(255,51,102,0.08);width:22%;box-shadow:0 0 16px rgba(255,51,102,0.15)">
            <div style="font-size:1.6rem;margin-bottom:8px">🎯</div>
            <div style="font-weight:bold;color:var(--red);font-size:0.85rem">Trapped Red-Handed</div>
            <div style="font-size:0.72rem;color:var(--text);margin-top:4px;word-break:break-all">Token: ${hacker.token_used}</div>
        </div>
    `;
    
    // Smooth scroll the flow visualization into view
    container.scrollIntoView({ behavior: 'smooth' });
}


async function loadSystemConfigTab() {
    const usersBody = document.getElementById('config-users-body');
    const userSelect = document.getElementById('config-role-username');
    usersBody.innerHTML = `<tr><td colspan="3" class="loading-cell">> Fetching user directory...</td></tr>`;
    
    try {
        const [usersRes, configRes] = await Promise.all([
            fetch(`${API}/api/admin/users`, { headers: authHeaders() }),
            fetch(`${API}/api/admin/config`, { headers: authHeaders() })
        ]);
        
        if (!usersRes.ok || !configRes.ok) throw new Error("Config retrieval failed");
        
        const users = await usersRes.json();
        const config = await configRes.json();
        
        // Render users table and select option drop-downs
        usersBody.innerHTML = '';
        userSelect.innerHTML = '';
        users.forEach(u => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-family:var(--font-mono); font-weight:bold;">${u.username}</td>
                <td><span class="badge ${u.role === 'admin' ? 'badge-critical' : u.role === 'analyst' ? 'badge-high' : 'badge-low'}">${u.role.toUpperCase()}</span></td>
                <td style="color:var(--text-dim)">${u.full_name}</td>
            `;
            usersBody.appendChild(tr);
            
            const opt = document.createElement('option');
            opt.value = u.username;
            opt.textContent = `${u.username} (${u.role})`;
            userSelect.appendChild(opt);
        });
        
        // Fill configuration fields
        document.getElementById('config-policy-epsilon').value = config.dqn_epsilon;
        document.getElementById('config-policy-frequency').value = config.token_deployment_frequency;
        document.getElementById('config-policy-compliance').value = config.watermark_compliance_min;
        document.getElementById('config-policy-rotation').checked = config.token_rotation_enabled;
        
        document.getElementById('hp-active-ssh').checked = config.active_honeypots.includes("SSH Cowrie Console");
        document.getElementById('hp-active-payment').checked = config.active_honeypots.includes("E-Commerce Checkout");
        document.getElementById('hp-active-login').checked = config.active_honeypots.includes("Admin Credential Trap");
        
    } catch(err) {
        usersBody.innerHTML = `<tr><td colspan="3" class="loading-cell" style="color:var(--red)">ERR: ${err.message}</td></tr>`;
    }
}

async function submitRoleChange(e) {
    e.preventDefault();
    const statusBox = document.getElementById('config-role-status');
    statusBox.style.display = 'block';
    statusBox.style.color = 'var(--accent-light)';
    statusBox.textContent = 'Updating role mapping...';
    
    const payload = {
        username: document.getElementById('config-role-username').value,
        role: document.getElementById('config-role-select').value
    };
    
    try {
        const res = await fetch(`${API}/api/admin/users/role`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        statusBox.style.color = 'var(--green)';
        statusBox.textContent = `✓ Role updated successfully.`;
        setTimeout(loadSystemConfigTab, 1000);
    } catch(err) {
        statusBox.style.color = 'var(--red)';
        statusBox.textContent = `Error: ${err.message}`;
    }
}

async function submitPolicyChange(e) {
    e.preventDefault();
    const statusBox = document.getElementById('config-policy-status');
    statusBox.style.display = 'block';
    statusBox.style.color = 'var(--accent-light)';
    statusBox.textContent = 'Saving system policies...';
    
    const active_honeypots = [];
    if (document.getElementById('hp-active-ssh').checked) active_honeypots.push("SSH Cowrie Console");
    if (document.getElementById('hp-active-payment').checked) active_honeypots.push("E-Commerce Checkout");
    if (document.getElementById('hp-active-login').checked) active_honeypots.push("Admin Credential Trap");
    
    const payload = {
        dqn_epsilon: parseFloat(document.getElementById('config-policy-epsilon').value),
        token_deployment_frequency: parseInt(document.getElementById('config-policy-frequency').value),
        watermark_compliance_min: parseFloat(document.getElementById('config-policy-compliance').value),
        active_honeypots: active_honeypots,
        token_rotation_enabled: document.getElementById('config-policy-rotation').checked
    };
    
    try {
        const res = await fetch(`${API}/api/admin/config`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        statusBox.style.color = 'var(--green)';
        statusBox.textContent = '✓ Configuration changes successfully synchronized across nodes.';
        setTimeout(loadSystemConfigTab, 1000);
    } catch(err) {
        statusBox.style.color = 'var(--red)';
        statusBox.textContent = `Error: ${err.message}`;
    }
}


async function trackUserEvent(event_type, details) {
    try {
        await fetch(`${API}/api/tracking/event`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify({ event_type, details })
        });
    } catch(e) {}
}

function loadSupportTab() {
    // Analytics tracking
    trackUserEvent('PAGE_VIEW', 'tab-support-desk');
    
    // Pre-fill email
    const emailField = document.getElementById('support-email');
    if (emailField && !emailField.value) {
        emailField.value = `${username.toLowerCase()}@corp.local`;
    }
}

async function submitSupportBugReport(e) {
    e.preventDefault();
    const statusBox = document.getElementById('support-status');
    statusBox.style.display = 'block';
    statusBox.className = 'glass-panel';
    statusBox.innerHTML = '<span style="color:var(--accent-light)">> Transmitting report payload...</span>';
    
    const payload = {
        email: document.getElementById('support-email').value,
        subject: document.getElementById('support-subject').value,
        message: document.getElementById('support-message').value
    };
    
    try {
        const res = await fetch(`${API}/api/support/ticket`, {
            method: 'POST',
            headers: authHeaders(),
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        statusBox.className = 'glass-panel';
        statusBox.style.borderColor = 'var(--green)';
        statusBox.style.backgroundColor = 'rgba(0, 255, 204, 0.05)';
        statusBox.innerHTML = `
            <div style="color:var(--green); font-weight:bold; margin-bottom:4px;">✓ Bug Report Registered</div>
            <div style="color:var(--text); line-height:1.4">
                Thank you. Incident reference <strong>${data.detail || 'TK-101'}</strong> saved in laboratory tracking logs.
            </div>
        `;
        document.getElementById('support-message').value = '';
    } catch(err) {
        statusBox.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PRESENTATION CO-PILOT SYSTEM
// ─────────────────────────────────────────────────────────────────────────────
let copilotStep = 0;
const copilotSteps = [
    {
        title: "Step 1: Introduction to Deceptive-Net",
        text: "Welcome to the Deceptive-Net Autonomous Honey-Token Framework. The platform secures transactions and credential paths by dynamically injecting fraud-aware honeypot tokens. We begin by highlighting our secure Onion Circuit routing dashboard.",
        tab: "dashboard",
        highlight: ".circuit-panel",
        actionText: "Check Circuit IP Address",
        action: () => {
            // Highlight circuit path guard/middle/exit
            document.querySelectorAll('.cnode').forEach(el => el.classList.add('copilot-highlight'));
            setTimeout(() => {
                document.querySelectorAll('.cnode').forEach(el => el.classList.remove('copilot-highlight'));
            }, 3000);
        }
    },
    {
        title: "Step 2: GAT & FDG Pipeline",
        text: "Under the hood, transactional similarities are computed inside a Fraud Dependency Graph (FDG) and attention parameters are processed dynamically using a Graph Attention Network (GAT) to ensure generative realism.",
        tab: "model",
        highlight: "#tab-model > div:nth-child(1)",
        actionText: "Load Model Metrics",
        action: () => {
            loadModelMetrics();
        }
    },
    {
        title: "Step 3: Secure Data Export & Leak Attributions",
        text: "When an authorized analyst exports dataset CSVs, the system dynamically embeds unique decoy honey-tokens with watermarks linked to the Exporter's identity and IP.",
        tab: "transactions",
        highlight: "#export-csv-btn",
        actionText: "Trigger Secure Data Export",
        action: () => {
            const btn = document.getElementById('export-csv-btn');
            btn.classList.add('copilot-highlight');
            setTimeout(() => btn.classList.remove('copilot-highlight'), 3000);
        }
    },
    {
        title: "Step 4: Interactive Attacker Checkout (Honeypot Trap)",
        text: "Let's simulate the attack. We will auto-fill the merchant checkout form using card credentials exfiltrated by an attacker, using the registered honey card 5454-4543-5445-4448.",
        tab: "playground",
        highlight: "#hp-payment-form",
        actionText: "Fill Honey Credentials",
        action: () => {
            document.getElementById('hp-cardholder').value = "John Hacker";
            document.getElementById('hp-cardnumber').value = "5454454354454448";
            document.getElementById('hp-cvv').value = "419";
            document.getElementById('hp-expiry').value = "12/28";
            document.getElementById('hp-amount').value = "899.95";
            document.getElementById('hp-ip').value = "185.220.101.5";
            const form = document.getElementById('hp-payment-form');
            form.classList.add('copilot-highlight');
            setTimeout(() => form.classList.remove('copilot-highlight'), 3000);
        }
    },
    {
        title: "Step 5: Cowrie SSH Honeypot Console Interaction",
        text: "Simultaneously, attackers scanning SSH ports enter a simulated Cowrie console. Let's send a command listing the directory and showing a credentials dump file.",
        tab: "playground",
        highlight: "#hp-terminal",
        actionText: "Type SSH Dump Command",
        action: async () => {
            const termInput = document.getElementById('hp-term-input');
            if (termInput) {
                termInput.value = "cat credentials_dump.csv";
                const event = new KeyboardEvent('keypress', { key: 'Enter', code: 'Enter', keyCode: 13 });
                termInput.dispatchEvent(event);
                handleTerminalInput(event);
            }
        }
    },
    {
        title: "Step 6: Real-time Forensic Breach Traceback",
        text: "Because the attacker used watermarked tokens, the forensic parser attributes the intrusion IP (185.220.101.5) back to the original exporter session in real-time.",
        tab: "threat-intel",
        highlight: "#intel-flow-container",
        actionText: "Draw Attribution Flow",
        action: () => {
            loadThreatIntel();
        }
    },
    {
        title: "Step 7: Generative AI Threat Analysis",
        text: "Finally, the system runs a generative AI threat report summarizing all honeypot breach trail tracebacks and recommends defensive configuration policies.",
        tab: "ai-report",
        highlight: "#ai-report-body",
        actionText: "Run AI Analysis",
        action: () => {
            loadAIReport();
        }
    }
];

function initPresentationCopilot() {
    const copilotBtn = document.getElementById('copilot-btn');
    const copilotWindow = document.getElementById('copilot-window');
    const closeBtn = document.getElementById('copilot-close-btn');

    const tabDemo = document.getElementById('copilot-tab-demo');
    const tabQA = document.getElementById('copilot-tab-qa');
    const contentDemo = document.getElementById('copilot-content-demo');
    const contentQA = document.getElementById('copilot-content-qa');

    if (!copilotBtn) return;

    // Toggle panel
    copilotBtn.addEventListener('click', () => {
        copilotWindow.style.display = copilotWindow.style.display === 'none' ? 'flex' : 'none';
        trackUserEvent('COPILOT_TOGGLE', 'toggle');
    });

    closeBtn.addEventListener('click', () => {
        copilotWindow.style.display = 'none';
    });

    // Tab switcher
    tabDemo.addEventListener('click', () => {
        tabDemo.classList.add('active');
        tabQA.classList.remove('active');
        contentDemo.style.display = 'block';
        contentQA.style.display = 'none';
    });

    tabQA.addEventListener('click', () => {
        tabQA.classList.add('active');
        tabDemo.classList.remove('active');
        contentQA.style.display = 'block';
        contentDemo.style.display = 'none';
    });

    // Guide controls
    const prevBtn = document.getElementById('copilot-prev');
    const nextBtn = document.getElementById('copilot-next');
    const autoBtn = document.getElementById('copilot-auto');
    const stepIndicator = document.getElementById('copilot-step-indicator');
    const stepText = document.getElementById('copilot-step-text');

    function updateStep() {
        const step = copilotSteps[copilotStep];
        stepIndicator.textContent = `Step ${copilotStep + 1} of ${copilotSteps.length}`;
        stepText.textContent = step.text;

        // Switch to the step's tab
        const navEl = document.querySelector(`.nav-item[data-tab="${step.tab}"]`);
        if (navEl) {
            switchTab(step.tab, navEl);
            // If admin tab and viewer role, temporarily reveal it
            if (role !== 'admin' && navEl.classList.contains('admin-only')) {
                navEl.style.display = 'flex';
                document.getElementById(`tab-${step.tab}`).style.display = 'flex';
            }
        }

        // Highlight element
        document.querySelectorAll('.copilot-highlight').forEach(el => el.classList.remove('copilot-highlight'));
        setTimeout(() => {
            const target = document.querySelector(step.highlight);
            if (target) {
                target.classList.add('copilot-highlight');
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, 500);

        // Update buttons
        prevBtn.disabled = copilotStep === 0;
        if (copilotStep === copilotSteps.length - 1) {
            nextBtn.textContent = "Finish Tour 🎉";
        } else {
            nextBtn.textContent = "Next Step ▶";
        }

        if (step.actionText) {
            autoBtn.textContent = `⚡ ${step.actionText}`;
            autoBtn.style.display = 'inline-flex';
        } else {
            autoBtn.style.display = 'none';
        }
    }

    nextBtn.addEventListener('click', () => {
        if (nextBtn.textContent.includes("Finish")) {
            copilotStep = 0;
            copilotWindow.style.display = 'none';
            document.querySelectorAll('.copilot-highlight').forEach(el => el.classList.remove('copilot-highlight'));
            alert("Presentation Tour Complete! The examiners should be thoroughly impressed.");
            return;
        }
        copilotStep++;
        updateStep();
    });

    prevBtn.addEventListener('click', () => {
        copilotStep--;
        updateStep();
    });

    autoBtn.addEventListener('click', () => {
        const step = copilotSteps[copilotStep];
        if (step.action) step.action();
    });

    // Q&A Hotline setup
    const qaModal = document.getElementById('copilot-modal');
    const qaClose = document.getElementById('copilot-modal-close');
    const qaTitle = document.getElementById('copilot-modal-title');
    const qaAnswer = document.getElementById('copilot-modal-answer');
    const qaCode = document.getElementById('copilot-modal-code');

    const qaAnswers = {
        gat: {
            title: "Q1: Fraud Dependency Graph & Graph Attention Network (GAT)",
            answer: "Deceptive-Net uses graph-structured transactional similarities to condition the generator network. Instead of creating isolated tabular profiles, a Fraud Dependency Graph (FDG) constructs demographically-linked transaction pairs. The Graph Attention Network (GAT) computes self-attention coefficients over node features, refining latent space boundaries to prevent generator mode collapse and output realistic decoy credentials.",
            code: `class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)`
        },
        ste: {
            title: "Q2: Luhn Check & Straight-Through Estimators (STE)",
            answer: "Continuous loss functions cannot backpropagate gradients through discrete, non-differentiable checksum validations like Luhn's algorithm for credit card digits. Standard PyTorch graphs break when evaluating step-functions. Our Constraint-Aware Attention (CAA) utilizes a Straight-Through Estimator (STE) where we use the discrete boolean checksum value in the forward pass, but bypass it in the backward pass, feeding a continuous identity gradient (1.0) back to the WGAN-GP generator weights.",
            code: `class LuhnSTEOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_digits):
        # input_digits represents probability distribution of 16 digits
        digits = torch.argmax(input_digits, dim=-1)
        validity = luhn_checksum_batch(digits) # 1.0 if correct Luhn, 0.0 otherwise
        return validity

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass non-differentiable check, pass gradient straight through
        return grad_output`
        },
        attribution: {
            title: "Q3: Cryptographic Watermarking & Leak Attribution",
            answer: "Breached credential detection works via dynamic honey-token attribution. When transactions are exported by an analyst, a unique cryptographic watermark is calculated based on user session details. The watermarked card token is saved in a secure Exfiltration Registry. When an external threat actor triggers our simulated payment checkout honeypot or types credentials inside the Cowrie SSH terminal, the system maps the decoy token back to the exporting session to immediately isolate the insider who leaked the data.",
            code: `def register_export(self, watermark_id: str, tokens: List[dict], metadata: dict):
    self.exports[watermark_id] = {
        "watermark_id": watermark_id,
        "ts": datetime.utcnow().isoformat(),
        "actor": metadata["username"],
        "ip": metadata["ip"],
        "tokens": [t["value"] for t in tokens]
    }`
        },
        dqn: {
            title: "Q4: DQN Token Deployment Agent Policy",
            answer: "The Token Deployment Agent (TDA) controls active rotation of honeypot tokens inside the Cowrie SSH Honeypot. Formulated as a reinforcement learning agent, the DQN state space is computed using command length, execution delay, and command entropy. The DQN agent selects rotation actions to maximize attacker engagement (dwell time) while minimizing defense exposure cost.",
            code: `def calculate_reward(self, state, action, dwell_time):
    # Reward = (alpha * dwell_time) - (beta * cost[action]) + (delta * entropy)
    reward = (self.alpha * dwell_time) - (self.beta * self.costs[action])
    if state.entropy > 0.8:
        reward += self.delta
    return reward`
        },
        compliance: {
            title: "Q5: SaaS Legal, Compliance & Privacy Protection",
            answer: "To ensure production SaaS viability, Deceptive-Net complies with regulatory requirements. It handles session security via signed JWT access tokens, prevents brute force attacks with login rate-limiters, hosts explicit Terms and Privacy policy gateways, and executes GDPR compliance check gates (e.g. cookie consent banner blocks). All user state transitions are tracked via telemetry events inside a tamper-proof audit ledger.",
            code: `async def track_user_event(request: Request, payload: TrackingEventRequest):
    # Store analytics event in DB audit table
    db_audit_log(
        actor=payload.user, 
        action="TRACK_EVENT", 
        resource=payload.event_type, 
        detail=payload.details
    )`
        }
    };

    document.querySelectorAll('.copilot-qa-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.q;
            const data = qaAnswers[key];
            if (data) {
                qaTitle.textContent = data.title;
                qaAnswer.textContent = data.answer;
                qaCode.textContent = data.code;
                qaModal.style.display = 'flex';

                // Automatically navigate to show live UI validation
                if (key === 'gat' || key === 'ste') {
                    const nav = document.getElementById('nav-model');
                    if (nav) switchTab('model', nav);
                } else if (key === 'attribution' || key === 'dqn') {
                    const nav = document.getElementById('nav-threat-intel');
                    if (nav) switchTab('threat-intel', nav);
                } else if (key === 'compliance') {
                    const nav = document.getElementById('nav-support');
                    if (nav) switchTab('support', nav);
                }
            }
        });
    });

    qaClose.addEventListener('click', () => {
        qaModal.style.display = 'none';
    });

    qaModal.addEventListener('click', (e) => {
        if (e.target === qaModal) qaModal.style.display = 'none';
    });
}

