"""
对比两组 Duplex 评估结果的可视化页面。
读取两个 eval_results.json（如 baseline vs compile），
生成一个静态 HTML 对比报告。

用法:
    python compare_viewer.py \
        --baseline /root/test/speed_test/eval_report/eval_results.json \
        --experiment /root/test/speed_test/eval_report_compiled/eval_results.json \
        --output /root/test/speed_test/compare_report/index.html
"""

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Duplex 评估对比</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d2e;
    --card-hover: #222640;
    --border: #2a2e45;
    --text: #e4e6f0;
    --text2: #9ba0b8;
    --accent: #6c5ce7;
    --accent2: #a29bfe;
    --green: #00cec9;
    --orange: #fdcb6e;
    --red: #ff7675;
    --blue: #74b9ff;
    --pink: #fd79a8;
    --teal: #55efc4;
    --colorA: #74b9ff;
    --colorB: #55efc4;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .header {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d3561 100%);
    padding: 24px 32px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
  }
  .header h1 { font-size: 22px; font-weight: 600; }
  .header .meta { color: var(--text2); font-size: 13px; margin-top: 4px; }
  .legend-bar {
    display: flex; gap: 20px; margin-top: 8px; font-size: 13px;
  }
  .legend-bar .leg { display: flex; align-items: center; gap: 6px; }
  .leg .dot { width: 12px; height: 12px; border-radius: 3px; }
  .dot-a { background: var(--colorA); }
  .dot-b { background: var(--colorB); }

  .container { max-width: 1600px; margin: 0 auto; padding: 24px; }

  /* ─── Summary Table ─── */
  .summary-section { margin-bottom: 28px; }
  .summary-section h2 { font-size: 16px; margin-bottom: 12px; }
  .summary-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .summary-table th, .summary-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }
  .summary-table th {
    background: var(--card);
    color: var(--text2);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    position: sticky; top: 0; z-index: 1;
  }
  .summary-table th:first-child, .summary-table td:first-child { text-align: left; }
  .summary-table tr:hover td { background: rgba(108,92,231,0.05); }
  .val-a { color: var(--colorA); font-weight: 600; }
  .val-b { color: var(--colorB); font-weight: 600; }
  .delta-pos { color: var(--green); font-weight: 700; }
  .delta-neg { color: var(--red); font-weight: 700; }
  .delta-neutral { color: var(--text2); }

  /* ─── Charts ─── */
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }
  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .chart-card h3 { font-size: 14px; color: var(--text2); margin-bottom: 12px; }
  .chart-card canvas { max-height: 320px; }

  /* ─── Per-video detail ─── */
  .video-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 14px;
    overflow: hidden;
  }
  .video-item:hover { border-color: var(--accent); }
  .video-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; cursor: pointer; user-select: none; flex-wrap: wrap; gap: 8px;
  }
  .video-header:hover { background: var(--card-hover); }
  .video-title { display: flex; align-items: center; gap: 10px; }
  .video-title .id-badge {
    background: var(--accent); color: #fff; font-size: 11px; font-weight: 700;
    padding: 2px 8px; border-radius: 12px;
  }
  .video-title .name { font-weight: 600; font-size: 13px; }
  .video-stats-mini { display: flex; gap: 12px; font-size: 11px; color: var(--text2); }
  .video-stats-mini .val { font-weight: 600; }
  .arrow { transition: transform 0.3s; font-size: 16px; color: var(--text2); }
  .arrow.open { transform: rotate(90deg); }
  .video-body { display: none; padding: 0 20px 16px; }
  .video-body.open { display: block; }
  .video-body canvas { max-height: 220px; }

  .mini-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; }
  .mini-table th, .mini-table td { padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--border); }
  .mini-table th { background: var(--bg); color: var(--text2); font-size: 10px; text-transform: uppercase; }
  .mini-table td:first-child, .mini-table th:first-child { text-align: left; }

  @media (max-width: 1000px) { .charts-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <h1>Duplex 评估对比报告</h1>
  <div class="meta" id="headerMeta"></div>
  <div class="legend-bar">
    <div class="leg"><div class="dot dot-a"></div><span id="labelA"></span></div>
    <div class="leg"><div class="dot dot-b"></div><span id="labelB"></span></div>
  </div>
</div>

<div class="container">
  <div class="summary-section" id="summarySection"></div>
  <div class="charts-grid" id="chartsGrid">
    <div class="chart-card"><h3>各视频 Wall Total 对比</h3><canvas id="chartWallTotal"></canvas></div>
    <div class="chart-card"><h3>各视频 Wall Prefill 对比</h3><canvas id="chartWallPrefill"></canvas></div>
    <div class="chart-card"><h3>各视频 Wall Generate 对比</h3><canvas id="chartWallGenerate"></canvas></div>
    <div class="chart-card"><h3>各阶段平均耗时对比</h3><canvas id="chartBreakdown"></canvas></div>
  </div>
  <h2 style="font-size:16px; margin-bottom:14px;">逐视频详情对比</h2>
  <div id="videoList"></div>
</div>

<script>
const A = __JSON_A__;
const B = __JSON_B__;
const LABEL_A = __LABEL_A__;
const LABEL_B = __LABEL_B__;

Chart.defaults.color = '#9ba0b8';
Chart.defaults.borderColor = 'rgba(42,46,69,0.6)';

const ms = (v) => ((v||0) * 1000).toFixed(1);
const g = (o,k,d=0) => o[k] !== undefined ? o[k] : d;
const pct = (a,b) => {
  if (!a || a === 0) return { text: '-', cls: 'delta-neutral' };
  const d = ((b - a) / a * 100);
  if (Math.abs(d) < 0.5) return { text: '~0%', cls: 'delta-neutral' };
  return d < 0
    ? { text: d.toFixed(1) + '%', cls: 'delta-pos' }
    : { text: '+' + d.toFixed(1) + '%', cls: 'delta-neg' };
};

document.getElementById('labelA').textContent = LABEL_A;
document.getElementById('labelB').textContent = LABEL_B;
document.getElementById('headerMeta').textContent =
  `${LABEL_A}: ${A.metadata.eval_time} | ${LABEL_B}: ${B.metadata.eval_time}`;

// ─── Match videos by name ───
const videoMap = {};
A.video_results.forEach(v => { videoMap[v.video_name] = { a: v }; });
B.video_results.forEach(v => {
  if (!videoMap[v.video_name]) videoMap[v.video_name] = {};
  videoMap[v.video_name].b = v;
});
const videoNames = Object.keys(videoMap).sort();

// ─── Summary Table ───
function renderSummary() {
  // Use no-warmup averages if available
  const oa = A.overall_averages_no_warmup || A.overall_averages;
  const ob = B.overall_averages_no_warmup || B.overall_averages;

  const metrics = [
    ['wall_prefill', 'Wall Prefill'],
    ['wall_generate', 'Wall Generate'],
    ['wall_chunk_total', 'Wall Total'],
    ['cost_prefill_all', 'Internal Prefill'],
    ['cost_vision_process', 'Vision Process'],
    ['cost_vision_embed', 'Vision Embed (ViT)'],
    ['cost_vision_feed', 'Vision Feed'],
    ['cost_audio_process', 'Audio Process'],
    ['cost_audio_embed', 'Audio Embed'],
    ['cost_audio_feed', 'Audio Feed'],
    ['cost_generate_all', 'Internal Generate'],
    ['cost_llm', 'LLM'],
    ['cost_tts_prep', 'TTS Prep'],
    ['cost_tts', 'TTS'],
    ['cost_token2wav', 'Token2Wav'],
    ['cost_chunk_total', 'Internal Total'],
    ['n_tokens', 'Tokens/chunk'],
    ['n_tts_tokens', 'TTS Tokens/chunk'],
  ];

  const isTime = (k) => !k.startsWith('n_');

  let html = `<h2>总体平均对比</h2>
    <table class="summary-table">
      <thead><tr><th>指标</th><th>${LABEL_A}</th><th>${LABEL_B}</th><th>变化</th></tr></thead><tbody>`;

  metrics.forEach(([k, label]) => {
    const va = g(oa, k);
    const vb = g(ob, k);
    const d = pct(va, vb);
    const fmtV = isTime(k) ? (v => ms(v) + 'ms') : (v => v.toFixed(2));
    html += `<tr>
      <td>${label}</td>
      <td class="val-a">${fmtV(va)}</td>
      <td class="val-b">${fmtV(vb)}</td>
      <td class="${d.cls}">${d.text}</td>
    </tr>`;
  });

  html += `<tr><td>总视频数</td><td>${A.metadata.total_videos}</td><td>${B.metadata.total_videos}</td><td></td></tr>`;
  html += `<tr><td>总 Chunks</td><td>${A.metadata.total_chunks}</td><td>${B.metadata.total_chunks}</td><td></td></tr>`;
  html += '</tbody></table>';

  document.getElementById('summarySection').innerHTML = html;
}

// ─── Global Charts ───
function renderGlobalCharts() {
  const labels = videoNames.map(n => n.replace('.mp4','').replace('omni_demo_duplex_','#'));
  const getAvg = (data, key) => {
    return videoNames.map(n => {
      const v = data === 'a' ? (videoMap[n]?.a) : (videoMap[n]?.b);
      return v ? g(v.averages, key) : 0;
    });
  };

  // Wall Total
  new Chart(document.getElementById('chartWallTotal'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: LABEL_A, data: getAvg('a','wall_chunk_total'), backgroundColor: 'rgba(116,185,255,0.6)', borderRadius: 3 },
      { label: LABEL_B, data: getAvg('b','wall_chunk_total'), backgroundColor: 'rgba(85,239,196,0.6)', borderRadius: 3 },
    ]},
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } } } }
  });

  // Wall Prefill
  new Chart(document.getElementById('chartWallPrefill'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: LABEL_A, data: getAvg('a','wall_prefill'), backgroundColor: 'rgba(116,185,255,0.6)', borderRadius: 3 },
      { label: LABEL_B, data: getAvg('b','wall_prefill'), backgroundColor: 'rgba(85,239,196,0.6)', borderRadius: 3 },
    ]},
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } } } }
  });

  // Wall Generate
  new Chart(document.getElementById('chartWallGenerate'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: LABEL_A, data: getAvg('a','wall_generate'), backgroundColor: 'rgba(116,185,255,0.6)', borderRadius: 3 },
      { label: LABEL_B, data: getAvg('b','wall_generate'), backgroundColor: 'rgba(85,239,196,0.6)', borderRadius: 3 },
    ]},
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } } } }
  });

  // Breakdown bar
  const oa = A.overall_averages_no_warmup || A.overall_averages;
  const ob = B.overall_averages_no_warmup || B.overall_averages;
  const bk = ['cost_vision_process','cost_vision_embed','cost_vision_feed',
              'cost_audio_process','cost_audio_embed','cost_audio_feed',
              'cost_llm','cost_tts_prep','cost_tts','cost_token2wav'];
  const bkLabels = ['vis_proc','vis_embed','vis_feed','aud_proc','aud_embed','aud_feed','llm','tts_prep','tts','tok2wav'];

  new Chart(document.getElementById('chartBreakdown'), {
    type: 'bar',
    data: {
      labels: bkLabels,
      datasets: [
        { label: LABEL_A, data: bk.map(k => g(oa,k)*1000), backgroundColor: 'rgba(116,185,255,0.7)', borderRadius: 3 },
        { label: LABEL_B, data: bk.map(k => g(ob,k)*1000), backgroundColor: 'rgba(85,239,196,0.7)', borderRadius: 3 },
      ]
    },
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: 'ms' } } } }
  });
}

// ─── Per-video list ───
const chartRendered = {};

function renderVideoList() {
  const container = document.getElementById('videoList');
  videoNames.forEach((name, idx) => {
    const va = videoMap[name]?.a;
    const vb = videoMap[name]?.b;
    const avgA = va ? va.averages : {};
    const avgB = vb ? vb.averages : {};
    const dTotal = pct(g(avgA,'wall_chunk_total'), g(avgB,'wall_chunk_total'));

    const el = document.createElement('div');
    el.className = 'video-item';
    el.innerHTML = `
      <div class="video-header" onclick="toggleVid(${idx})">
        <div class="video-title">
          <span class="id-badge">${idx+1}</span>
          <span class="name">${name}</span>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
          <div class="video-stats-mini">
            <span class="val-a">${ms(g(avgA,'wall_chunk_total'))}ms</span>
            <span>vs</span>
            <span class="val-b">${ms(g(avgB,'wall_chunk_total'))}ms</span>
            <span class="${dTotal.cls}">(${dTotal.text})</span>
          </div>
          <span class="arrow" id="varrow-${idx}">&#9654;</span>
        </div>
      </div>
      <div class="video-body" id="vbody-${idx}">
        <canvas id="vchart-${idx}" height="180"></canvas>
        <table class="mini-table">
          <thead><tr>
            <th>指标</th><th>${LABEL_A}</th><th>${LABEL_B}</th><th>变化</th>
          </tr></thead>
          <tbody>
            ${renderMiniRows(avgA, avgB)}
          </tbody>
        </table>
      </div>
    `;
    container.appendChild(el);
  });
}

function renderMiniRows(avgA, avgB) {
  const rows = [
    ['wall_prefill','Wall Prefill'],['wall_generate','Wall Generate'],['wall_chunk_total','Wall Total'],
    ['cost_prefill_all','Prefill Internal'],['cost_vision_embed','Vision Embed'],['cost_vision_feed','Vision Feed'],
    ['cost_generate_all','Generate Internal'],['cost_llm','LLM'],['cost_tts','TTS'],['cost_token2wav','Token2Wav'],
  ];
  return rows.map(([k,label]) => {
    const va = g(avgA,k); const vb = g(avgB,k); const d = pct(va,vb);
    return `<tr><td>${label}</td><td class="val-a">${ms(va)}ms</td><td class="val-b">${ms(vb)}ms</td><td class="${d.cls}">${d.text}</td></tr>`;
  }).join('');
}

function toggleVid(idx) {
  const body = document.getElementById(`vbody-${idx}`);
  const arrow = document.getElementById(`varrow-${idx}`);
  const isOpen = body.classList.toggle('open');
  arrow.classList.toggle('open', isOpen);
  if (isOpen && !chartRendered[idx]) {
    renderVidChart(idx);
    chartRendered[idx] = true;
  }
}

function renderVidChart(idx) {
  const name = videoNames[idx];
  const ca = videoMap[name]?.a?.chunk_results || [];
  const cb = videoMap[name]?.b?.chunk_results || [];
  const maxLen = Math.max(ca.length, cb.length);
  const labels = Array.from({length: maxLen}, (_,i) => i);

  new Chart(document.getElementById(`vchart-${idx}`), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: `${LABEL_A} wall_total`, data: ca.map(c => g(c,'wall_chunk_total')), borderColor: 'rgba(116,185,255,0.9)', backgroundColor: 'rgba(116,185,255,0.1)', fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: `${LABEL_B} wall_total`, data: cb.map(c => g(c,'wall_chunk_total')), borderColor: 'rgba(85,239,196,0.9)', backgroundColor: 'rgba(85,239,196,0.1)', fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: `${LABEL_A} wall_prefill`, data: ca.map(c => g(c,'wall_prefill')), borderColor: 'rgba(116,185,255,0.5)', borderDash: [4,3], tension: 0.3, pointRadius: 1, borderWidth: 1.5 },
        { label: `${LABEL_B} wall_prefill`, data: cb.map(c => g(c,'wall_prefill')), borderColor: 'rgba(85,239,196,0.5)', borderDash: [4,3], tension: 0.3, pointRadius: 1, borderWidth: 1.5 },
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { title: { display: true, text: 'Chunk' } },
        y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });
}

renderSummary();
renderGlobalCharts();
renderVideoList();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="生成两组 Duplex 评估结果的对比报告")
    parser.add_argument("--baseline", type=str, required=True,
                        help="基线结果 JSON（如无 compile 版本）")
    parser.add_argument("--experiment", type=str, required=True,
                        help="实验结果 JSON（如 compile 版本）")
    parser.add_argument("--baseline_label", type=str, default="Baseline (no compile)",
                        help="基线标签名称")
    parser.add_argument("--experiment_label", type=str, default="torch.compile",
                        help="实验标签名称")
    parser.add_argument("--output", type=str,
                        default="/root/test/speed_test/compare_report/index.html",
                        help="输出 HTML 路径")
    args = parser.parse_args()

    print(f"[INFO] 加载基线: {args.baseline}")
    with open(args.baseline, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    print(f"[INFO] 加载实验: {args.experiment}")
    with open(args.experiment, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    json_a = json.dumps(data_a, ensure_ascii=False)
    json_b = json.dumps(data_b, ensure_ascii=False)
    label_a = json.dumps(args.baseline_label, ensure_ascii=False)
    label_b = json.dumps(args.experiment_label, ensure_ascii=False)

    html = HTML_TEMPLATE
    html = html.replace("__JSON_A__", json_a)
    html = html.replace("__JSON_B__", json_b)
    html = html.replace("__LABEL_A__", label_a)
    html = html.replace("__LABEL_B__", label_b)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] 对比报告已生成: {output_path}")
    print(f"[INFO] 启动服务: cd {output_path.parent} && python -m http.server 8766")


if __name__ == "__main__":
    main()
