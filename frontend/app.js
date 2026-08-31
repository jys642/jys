// ============================================================
// 药板药片外观缺陷智能检测系统 —— 前端交互逻辑
// 对接后端 FastAPI 接口，实现上传检测、历史查询、统计分析
// ============================================================

const API = {
  detect: '/api/detect',
  records: '/api/records',
  record: (id) => `/api/records/${id}`,
  statistics: '/api/statistics',
};

let defectChart = null;
let selectedFile = null;

// ---------------- 视图切换 ----------------
document.querySelectorAll('.app-nav .nav-link').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.app-nav .nav-link').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
    document.getElementById('view-' + btn.dataset.view).classList.add('active');

    if (btn.dataset.view === 'history') loadHistory();
    if (btn.dataset.view === 'stats') loadStatistics();
  });
});

// ---------------- 上传与检测 ----------------
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const detectBtn = document.getElementById('detectBtn');

uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

['dragover', 'dragenter'].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach((evt) =>
  uploadZone.addEventListener(evt, (e) => { e.preventDefault(); uploadZone.classList.remove('dragover'); })
);
uploadZone.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) { alert('请选择图片文件'); return; }
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  preview.hidden = false;
  detectBtn.disabled = false;
}

detectBtn.addEventListener('click', detect);

async function detect() {
  if (!selectedFile) return;
  const status = document.getElementById('detectStatus');
  detectBtn.disabled = true;
  detectBtn.textContent = '检测中…';
  status.hidden = false;
  status.textContent = '正在执行预处理、YOLO 检测与随机森林校验…';
  status.className = 'status';

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch(API.detect, { method: 'POST', body: form });
    if (!res.ok) { throw new Error((await res.json()).detail || '检测失败'); }
    const data = await res.json();
    renderResult(data);
    status.hidden = true;
  } catch (err) {
    status.textContent = '检测失败：' + err.message;
    status.className = 'status error';
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = '开始检测';
  }
}

function renderResult(data) {
  document.getElementById('emptyPanel').hidden = true;
  const panel = document.getElementById('resultPanel');
  panel.hidden = false;

  const badge = document.getElementById('conclusionBadge');
  const ng = data.defect_count > 0;
  badge.textContent = ng ? '不合格' : '合格';
  badge.className = 'badge conclusion ' + (ng ? 'ng' : 'ok');

  document.getElementById('resultImage').src = data.annotated_path;
  document.getElementById('defectList').innerHTML = data.defects.map(defectCard).join('');
}

function defectCard(d) {
  return `
    <div class="defect-item">
      <div class="defect-badge">${d.class_cn}</div>
      <div class="defect-info">
        <div class="defect-name">${d.class_cn}（${d.class_name}）</div>
        <div class="defect-sub">YOLO ${d.yolo_class}/${d.yolo_conf} · RF 校验 ${d.rf_class}/${d.rf_conf} · 框 [${d.bbox.map((v) => Math.round(v)).join(', ')}]</div>
      </div>
      <div class="defect-conf">
        <div class="conf-value">${(d.confidence * 100).toFixed(1)}%</div>
        <div class="conf-label">置信度</div>
      </div>
    </div>`;
}

// ---------------- 历史记录 ----------------
async function loadHistory() {
  const body = document.getElementById('historyBody');
  const empty = document.getElementById('historyEmpty');
  try {
    const res = await fetch(API.records + '?limit=100');
    const records = await res.json();
    empty.hidden = records.length > 0;
    body.innerHTML = records.map((r) => `
      <tr>
        <td>${r.id}</td>
        <td><img class="thumb" src="/api/image/${r.image_name}" alt="${r.image_name}"></td>
        <td>${r.created_at}</td>
        <td>${r.defect_count}</td>
        <td><span class="badge ${r.defect_count > 0 ? 'text-bg-danger' : 'text-bg-success'}">${r.conclusion}</span></td>
        <td><button class="btn btn-sm btn-outline-primary" onclick="viewDetail(${r.id})">详情</button></td>
      </tr>`).join('');
  } catch (err) {
    body.innerHTML = `<tr><td colspan="6" class="text-center text-muted">加载失败</td></tr>`;
  }
}

async function viewDetail(id) {
  try {
    const res = await fetch(API.record(id));
    const d = await res.json();
    document.getElementById('detailMeta').textContent =
      `#${d.id} · ${d.created_at} · 缺陷 ${d.defect_count} 处 · ${d.conclusion}`;
    document.getElementById('detailImage').src = `/api/image/${d.image_name.replace(/\.[^.]+$/, '')}_annotated.jpg`;
    document.getElementById('detailDefects').innerHTML =
      (d.defects.length ? d.defects.map(defectCard).join('') : '<div class="empty-hint">无缺陷</div>');
    new bootstrap.Modal(document.getElementById('detailModal')).show();
  } catch (err) {
    alert('加载详情失败');
  }
}

// ---------------- 统计分析 ----------------
async function loadStatistics() {
  try {
    const res = await fetch(API.statistics);
    const s = await res.json();
    const ng = s.total_records ? s.by_class.reduce((a, c) => a + c.count, 0) : 0;
    const ngRate = s.total_records ? ((ng / s.total_records) * 100).toFixed(1) : '0.0';

    document.getElementById('statCards').innerHTML = `
      <div class="col-md-4"><div class="card panel stat-card"><div class="panel-body">
        <div class="stat-label">累计检测次数</div>
        <div class="stat-value stat-accent">${s.total_records}<span class="unit"> 次</span></div>
      </div></div></div>
      <div class="col-md-4"><div class="card panel stat-card"><div class="panel-body">
        <div class="stat-label">累计缺陷数</div>
        <div class="stat-value">${s.total_defects}<span class="unit"> 处</span></div>
      </div></div></div>
      <div class="col-md-4"><div class="card panel stat-card"><div class="panel-body">
        <div class="stat-label">平均每板缺陷</div>
        <div class="stat-value">${s.total_records ? (s.total_defects / s.total_records).toFixed(2) : '0.00'}<span class="unit"> 处</span></div>
      </div></div></div>`;

    drawChart(s.by_class);
  } catch (err) {
    console.error(err);
  }
}

function drawChart(byClass) {
  const ctx = document.getElementById('defectChart');
  if (defectChart) defectChart.destroy();
  defectChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: byClass.map((c) => c.class_cn),
      datasets: [{
        label: '检出数量',
        data: byClass.map((c) => c.count),
        backgroundColor: 'rgba(13, 148, 136, .75)',
        borderColor: '#0f766e',
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 1 } },
      },
    },
  });
}
