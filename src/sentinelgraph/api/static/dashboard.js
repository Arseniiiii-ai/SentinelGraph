const keyInput = document.querySelector('#api-key');
const message = document.querySelector('#message');
const list = document.querySelector('#case-list');
const detail = document.querySelector('#case-detail');
keyInput.value = sessionStorage.getItem('sentinelgraph-api-key') || '';

async function api(path, options = {}) {
  const key = keyInput.value;
  sessionStorage.setItem('sentinelgraph-api-key', key);
  const response = await fetch(path, {
    ...options,
    headers: {'X-API-Key': key, 'Content-Type': 'application/json', ...(options.headers || {})},
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}

async function loadCases() {
  message.textContent = '';
  const status = document.querySelector('#status-filter').value;
  try {
    const data = await api(`/v1/cases?limit=100${status ? `&status=${status}` : ''}`);
    document.querySelector('#case-count').textContent = `${data.count} cases`;
    list.innerHTML = data.cases.map(item => `
      <div class="case-card" data-id="${escapeHtml(item.case_id)}">
        <span class="risk">${item.risk_points}</span>
        <div><strong>${escapeHtml(item.external_id)}</strong><div class="case-meta">${item.decision} · ${item.status}</div></div>
        <span class="badge ${item.priority}">${item.priority}</span>
      </div>`).join('') || '<p class="empty-state">Queue is empty.</p>';
    document.querySelectorAll('.case-card').forEach(card => card.addEventListener('click', () => loadCase(card.dataset.id)));
  } catch (error) { message.textContent = error.message; }
}

async function loadCase(id) {
  message.textContent = '';
  try {
    const item = await api(`/v1/cases/${id}`);
    detail.innerHTML = `
      <p class="eyebrow">Case ${escapeHtml(item.case_id)}</p>
      <h2>${escapeHtml(item.external_id)}</h2>
      <div class="detail-grid">
        <div class="metric"><span>Risk points</span><strong>${item.risk_points}</strong></div>
        <div class="metric"><span>Probability</span><strong>${(item.risk_probability * 100).toFixed(2)}%</strong></div>
        <div class="metric"><span>Amount</span><strong>${Number(item.transaction.amount).toLocaleString()}</strong></div>
      </div>
      <h3>Reason codes</h3>
      ${item.reason_codes.map(reason => `<div class="reason"><strong>${escapeHtml(reason.code)}</strong><br>${escapeHtml(reason.description)} <small>Δ ${Number(reason.contribution).toFixed(4)}</small></div>`).join('') || '<p>No positive local contributions.</p>'}
      <form class="decision-form" id="decision-form">
        <select name="status"><option>in_review</option><option>closed</option></select>
        <input name="investigator" placeholder="Investigator" required>
        <select name="disposition"><option value="">No disposition</option><option>confirmed_fraud</option><option>legitimate</option><option>inconclusive</option></select>
        <input name="assigned_to" placeholder="Assign to">
        <textarea name="notes" placeholder="Evidence-based notes"></textarea>
        <button>Record decision</button>
      </form>`;
    document.querySelector('#decision-form').addEventListener('submit', async event => {
      event.preventDefault();
      const form = new FormData(event.target);
      const status = form.get('status');
      const disposition = form.get('disposition') || null;
      try {
        await api(`/v1/cases/${id}/decision`, {method: 'POST', body: JSON.stringify({
          status, investigator: form.get('investigator'), expected_version: item.version,
          assigned_to: form.get('assigned_to') || null, disposition,
          fraud_confirmed: disposition === 'confirmed_fraud' ? true : disposition === 'legitimate' ? false : null,
          notes: form.get('notes') || null,
        })});
        await loadCases(); await loadCase(id);
      } catch (error) { message.textContent = error.message; }
    });
  } catch (error) { message.textContent = error.message; }
}

document.querySelector('#load-cases').addEventListener('click', loadCases);
