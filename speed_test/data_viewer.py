"""
Duplex 评估结果数据查看器 - 生成纯静态 HTML 文件。
读取 eval_results.json，生成 index.html，
原始视频和渲染结果视频均通过相对路径引用（懒加载）。

包含完整的 prefill（vision/audio）+ generate（llm/tts）两阶段耗时展示。

用法:
    python data_viewer.py --results /root/test/eval_report/eval_results.json
"""

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Duplex 评估结果查看器</title>
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

  .container { max-width: 1600px; margin: 0 auto; padding: 24px; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 10px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    text-align: center;
  }
  .stat-card .label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.4px; }
  .stat-card .value { font-size: 20px; font-weight: 700; margin-top: 3px; }
  .stat-card .value.accent { color: var(--accent2); }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.orange { color: var(--orange); }
  .stat-card .value.blue { color: var(--blue); }
  .stat-card .value.pink { color: var(--pink); }
  .stat-card .value.teal { color: var(--teal); }

  .charts-section {
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
  .chart-card canvas { max-height: 300px; }

  .video-list { margin-bottom: 24px; }
  .video-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 16px;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .video-item:hover { border-color: var(--accent); }

  .video-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 20px;
    cursor: pointer;
    user-select: none;
    flex-wrap: wrap;
    gap: 8px;
  }
  .video-header:hover { background: var(--card-hover); }
  .video-title {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .video-title .id-badge {
    background: var(--accent);
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    min-width: 32px;
    text-align: center;
  }
  .video-title .name { font-weight: 600; font-size: 14px; }

  .video-stats-mini {
    display: flex; gap: 14px; font-size: 11px; color: var(--text2);
  }
  .video-stats-mini span { white-space: nowrap; }
  .video-stats-mini .val { color: var(--green); font-weight: 600; }

  .arrow { transition: transform 0.3s; font-size: 18px; color: var(--text2); }
  .arrow.open { transform: rotate(90deg); }

  .video-body { display: none; padding: 0 20px 20px; }
  .video-body.open { display: block; }

  .dual-video-area {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .video-panel { min-width: 0; }
  .video-panel h4 {
    font-size: 13px;
    color: var(--text2);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .video-panel h4 .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
  }
  .dot-src { background: var(--blue); }
  .dot-out { background: var(--green); }
  .video-placeholder {
    width: 100%;
    height: 200px;
    background: #111;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border: 2px dashed var(--border);
    transition: border-color 0.2s;
  }
  .video-placeholder:hover { border-color: var(--accent); }
  .video-placeholder span { color: var(--text2); font-size: 13px; }
  video {
    width: 100%;
    max-height: 200px;
    object-fit: contain;
    border-radius: 8px;
    background: #000;
  }

  .model-out-section { margin-bottom: 16px; }
  .model-out-section h4 { font-size: 13px; color: var(--text2); margin-bottom: 6px; }
  .model-out-box {
    background: var(--bg);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
    line-height: 1.7;
    border-left: 3px solid var(--green);
  }

  /* ─── 平均指标分组 ─── */
  .avg-section { margin-bottom: 16px; }
  .avg-section h4 { font-size: 12px; color: var(--text2); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .avg-bar {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;
  }
  .avg-item {
    background: var(--bg);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
  }
  .avg-item .lbl { color: var(--text2); }
  .avg-item .v { font-weight: 700; margin-left: 3px; }

  .video-charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  .video-charts-grid canvas { max-height: 220px; }

  .chunk-table-wrapper { overflow-x: auto; max-height: 450px; overflow-y: auto; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }
  th, td {
    padding: 6px 8px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }
  th {
    background: var(--card);
    color: var(--text2);
    font-weight: 600;
    position: sticky;
    top: 0;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    z-index: 1;
  }
  th.group-prefill { border-top: 2px solid var(--pink); }
  th.group-generate { border-top: 2px solid var(--blue); }
  td:first-child, th:first-child { text-align: center; }
  td:nth-child(2), th:nth-child(2) { text-align: center; }
  td:nth-child(3), th:nth-child(3) { text-align: left; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
  tr:hover td { background: rgba(108,92,231,0.05); }
  .listen-row td { color: var(--text2); }
  .speak-row td:nth-child(3) { color: var(--green); font-weight: 500; }

  @media (max-width: 1000px) {
    .charts-section { grid-template-columns: 1fr; }
    .dual-video-area { grid-template-columns: 1fr; }
    .video-charts-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>Duplex 评估结果查看器</h1>
  <div class="meta" id="headerMeta"></div>
</div>

<div class="container">
  <div class="stats-grid" id="statsGrid"></div>
  <div class="charts-section" id="globalCharts">
    <div class="chart-card">
      <h3>各视频 Chunk 平均总耗时 (prefill + generate)</h3>
      <canvas id="chartTotal"></canvas>
    </div>
    <div class="chart-card">
      <h3>各视频 Prefill 耗时组成</h3>
      <canvas id="chartPrefill"></canvas>
    </div>
    <div class="chart-card">
      <h3>各视频 Generate 耗时组成 (speak)</h3>
      <canvas id="chartGenerate"></canvas>
    </div>
    <div class="chart-card">
      <h3>Prefill vs Generate 占比</h3>
      <canvas id="chartPieAvg"></canvas>
    </div>
  </div>
  <h2 style="font-size:18px; margin-bottom:16px;">视频详情 (共 <span id="videoCount"></span> 个)</h2>
  <div class="video-list" id="videoList"></div>
</div>

<script>
const DATA = __JSON_DATA__;
const ms = (v) => (v * 1000).toFixed(1);

Chart.defaults.color = '#9ba0b8';
Chart.defaults.borderColor = 'rgba(42,46,69,0.6)';

function fmt(v, d=4) { return typeof v === 'number' ? v.toFixed(d) : v; }
function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// safe access for backward compat (old JSON without prefill fields)
function g(obj, key, def=0) { return obj[key] !== undefined ? obj[key] : def; }

function renderStats() {
  const oa = DATA.overall_averages;
  const m = DATA.metadata;
  document.getElementById('headerMeta').textContent =
    `模型: ${m.model_path} | 视频目录: ${m.video_dir} | 评估时间: ${m.eval_time}`;

  const items = [
    { label: '总视频数', value: m.total_videos, cls: 'accent' },
    { label: '总 Chunks', value: m.total_chunks, cls: 'blue' },
    { label: 'Speak / Listen', value: `${g(oa,'speak_chunks')} / ${g(oa,'listen_chunks')}`, cls: 'green' },
    { label: 'Avg Chunk Total', value: ms(g(oa,'cost_chunk_total')) + 'ms', cls: 'accent' },
    { label: 'Avg Prefill', value: ms(g(oa,'cost_prefill_all')) + 'ms', cls: 'pink' },
    { label: 'Avg ViT Embed', value: ms(g(oa,'cost_vision_embed')) + 'ms', cls: 'teal' },
    { label: 'Avg Generate', value: ms(g(oa,'cost_generate_all')) + 'ms', cls: 'blue' },
    { label: 'Avg LLM', value: ms(g(oa,'cost_llm')) + 'ms', cls: 'green' },
    { label: 'Avg TTS', value: ms(g(oa,'cost_tts')) + 'ms', cls: 'orange' },
    { label: 'Avg Generate (speak)', value: ms(g(oa,'cost_generate_all_speak_only')) + 'ms', cls: 'orange' },
    { label: 'Wall Prefill', value: ms(g(oa,'wall_prefill')) + 'ms', cls: 'pink' },
    { label: 'Wall Generate', value: ms(g(oa,'wall_generate')) + 'ms', cls: 'blue' },
    { label: 'Wall Total', value: ms(g(oa,'wall_chunk_total')) + 'ms', cls: 'accent' },
  ];
  document.getElementById('statsGrid').innerHTML = items.map(i => `
    <div class="stat-card">
      <div class="label">${i.label}</div>
      <div class="value ${i.cls}">${i.value}</div>
    </div>
  `).join('');
  document.getElementById('videoCount').textContent = m.total_videos;
}

function renderOverallCharts() {
  const R = DATA.video_results;
  const labels = R.map((r, i) => r.video_name.replace('.mp4','').replace('omni_demo_duplex_','#'));

  // Chart 1: chunk total = prefill + generate
  new Chart(document.getElementById('chartTotal'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Prefill', data: R.map(r => g(r.averages,'cost_prefill_all')), backgroundColor: 'rgba(253,121,168,0.6)', borderRadius: 2 },
        { label: 'Generate', data: R.map(r => g(r.averages,'cost_generate_all')), backgroundColor: 'rgba(116,185,255,0.6)', borderRadius: 2 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });

  // Chart 2: Prefill breakdown
  new Chart(document.getElementById('chartPrefill'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'vision_process', data: R.map(r => g(r.averages,'cost_vision_process')), backgroundColor: 'rgba(253,121,168,0.7)', borderRadius: 2 },
        { label: 'vision_embed (ViT)', data: R.map(r => g(r.averages,'cost_vision_embed')), backgroundColor: 'rgba(85,239,196,0.7)', borderRadius: 2 },
        { label: 'vision_feed', data: R.map(r => g(r.averages,'cost_vision_feed')), backgroundColor: 'rgba(162,155,254,0.7)', borderRadius: 2 },
        { label: 'audio_process', data: R.map(r => g(r.averages,'cost_audio_process')), backgroundColor: 'rgba(253,203,110,0.5)', borderRadius: 2 },
        { label: 'audio_embed', data: R.map(r => g(r.averages,'cost_audio_embed')), backgroundColor: 'rgba(0,206,201,0.5)', borderRadius: 2 },
        { label: 'audio_feed', data: R.map(r => g(r.averages,'cost_audio_feed')), backgroundColor: 'rgba(116,185,255,0.5)', borderRadius: 2 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });

  // Chart 3: Generate breakdown (speak only)
  new Chart(document.getElementById('chartGenerate'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'cost_llm', data: R.map(r => g(r.averages,'cost_llm_speak_only')), backgroundColor: 'rgba(0,206,201,0.7)', borderRadius: 2 },
        { label: 'cost_tts_prep', data: R.map(r => g(r.averages,'cost_tts_prep_speak_only')), backgroundColor: 'rgba(253,203,110,0.7)', borderRadius: 2 },
        { label: 'cost_tts', data: R.map(r => g(r.averages,'cost_tts_speak_only')), backgroundColor: 'rgba(116,185,255,0.7)', borderRadius: 2 },
        { label: 'cost_token2wav', data: R.map(r => g(r.averages,'cost_token2wav_speak_only')), backgroundColor: 'rgba(162,155,254,0.7)', borderRadius: 2 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });

  // Chart 4: Pie - overall prefill vs generate
  const oa = DATA.overall_averages;
  new Chart(document.getElementById('chartPieAvg'), {
    type: 'doughnut',
    data: {
      labels: ['Vision Process', 'Vision Embed (ViT)', 'Vision Feed', 'Audio Process', 'Audio Embed', 'Audio Feed', 'LLM', 'TTS Prep', 'TTS', 'Token2Wav'],
      datasets: [{
        data: [
          g(oa,'cost_vision_process'), g(oa,'cost_vision_embed'), g(oa,'cost_vision_feed'),
          g(oa,'cost_audio_process'), g(oa,'cost_audio_embed'), g(oa,'cost_audio_feed'),
          g(oa,'cost_llm'), g(oa,'cost_tts_prep'), g(oa,'cost_tts'), g(oa,'cost_token2wav'),
        ],
        backgroundColor: [
          'rgba(253,121,168,0.8)', 'rgba(85,239,196,0.8)', 'rgba(162,155,254,0.8)',
          'rgba(253,203,110,0.6)', 'rgba(0,206,201,0.6)', 'rgba(116,185,255,0.6)',
          'rgba(0,206,201,0.9)', 'rgba(253,203,110,0.9)', 'rgba(116,185,255,0.9)', 'rgba(162,155,254,0.9)',
        ],
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'right', labels: { boxWidth: 10, font: { size: 10 } } } }
    }
  });
}

function renderVideoList() {
  const container = document.getElementById('videoList');
  DATA.video_results.forEach((vr, idx) => {
    const avg = vr.averages;
    const el = document.createElement('div');
    el.className = 'video-item';
    el.innerHTML = `
      <div class="video-header" onclick="toggleVideo(${idx})">
        <div class="video-title">
          <span class="id-badge">${idx + 1}</span>
          <span class="name">${escHtml(vr.video_name)}</span>
        </div>
        <div style="display:flex;align-items:center;gap:16px;">
          <div class="video-stats-mini">
            <span>chunks: <span class="val">${g(avg,'total_chunks')}</span></span>
            <span>speak: <span class="val">${g(avg,'speak_chunks')}</span></span>
            <span>wall_pf: <span class="val">${ms(g(avg,'wall_prefill'))}ms</span></span>
            <span>wall_gen: <span class="val">${ms(g(avg,'wall_generate'))}ms</span></span>
            <span>wall_total: <span class="val">${ms(g(avg,'wall_chunk_total'))}ms</span></span>
          </div>
          <span class="arrow" id="arrow-${idx}">&#9654;</span>
        </div>
      </div>
      <div class="video-body" id="body-${idx}">
        <div class="dual-video-area">
          <div class="video-panel">
            <h4><span class="dot dot-src"></span> 原始视频</h4>
            <div class="video-placeholder" id="vp-src-${idx}" onclick="loadSrcVideo(${idx})">
              <span>点击加载原始视频</span>
            </div>
          </div>
          <div class="video-panel">
            <h4><span class="dot dot-out"></span> 推理结果视频</h4>
            <div class="video-placeholder" id="vp-out-${idx}" onclick="loadOutVideo(${idx})">
              <span>点击加载结果视频</span>
            </div>
          </div>
        </div>

        <div class="model-out-section">
          <h4>模型输出文本</h4>
          <div class="model-out-box">${escHtml(vr.model_output_text) || '<i style="color:#666">无文本输出</i>'}</div>
        </div>

        <div class="avg-section">
          <h4>Prefill 平均耗时</h4>
          <div class="avg-bar">
            ${['cost_vision_process','cost_vision_embed','cost_vision_feed','cost_audio_process','cost_audio_embed','cost_audio_feed','cost_prefill_all'].map(k =>
              `<div class="avg-item"><span class="lbl">${k.replace('cost_','')}:</span><span class="v">${ms(g(avg,k))}ms</span></div>`
            ).join('')}
          </div>
          <h4>Generate 平均耗时</h4>
          <div class="avg-bar">
            ${['cost_llm','cost_tts_prep','cost_tts','cost_token2wav','cost_generate_all'].map(k =>
              `<div class="avg-item"><span class="lbl">${k.replace('cost_','')}:</span><span class="v">${ms(g(avg,k))}ms</span></div>`
            ).join('')}
          </div>
          <h4>Wall 外部计时</h4>
          <div class="avg-bar">
            <div class="avg-item"><span class="lbl">wall_prefill:</span><span class="v">${ms(g(avg,'wall_prefill'))}ms</span></div>
            <div class="avg-item"><span class="lbl">wall_generate:</span><span class="v">${ms(g(avg,'wall_generate'))}ms</span></div>
            <div class="avg-item"><span class="lbl">wall_total:</span><span class="v">${ms(g(avg,'wall_chunk_total'))}ms</span></div>
            <div class="avg-item"><span class="lbl">internal_total:</span><span class="v">${ms(g(avg,'cost_chunk_total'))}ms</span></div>
            <div class="avg-item"><span class="lbl">n_tokens:</span><span class="v">${fmt(g(avg,'n_tokens'),2)}</span></div>
            <div class="avg-item"><span class="lbl">n_tts_tokens:</span><span class="v">${fmt(g(avg,'n_tts_tokens'),2)}</span></div>
          </div>
        </div>

        <div class="video-charts-grid">
          <canvas id="vchart-prefill-${idx}" height="180"></canvas>
          <canvas id="vchart-generate-${idx}" height="180"></canvas>
        </div>

        <div class="chunk-table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Chunk</th><th>状态</th><th>文本</th>
                <th class="group-prefill">vis_proc</th><th class="group-prefill">vis_embed</th><th class="group-prefill">vis_feed</th>
                <th class="group-prefill">aud_proc</th><th class="group-prefill">aud_embed</th><th class="group-prefill">aud_feed</th>
                <th class="group-prefill">prefill</th>
                <th class="group-generate">llm</th><th class="group-generate">tts_prep</th>
                <th class="group-generate">tts</th><th class="group-generate">tok2wav</th>
                <th class="group-generate">generate</th>
                <th>internal</th>
                <th>wall_pf</th><th>wall_gen</th><th>wall_total</th>
                <th>tok</th><th>tts_tok</th><th>EoT</th>
              </tr>
            </thead>
            <tbody>
              ${vr.chunk_results.map(c => `
                <tr class="${c.is_listen ? 'listen-row' : 'speak-row'}">
                  <td>${c.chunk_idx}</td>
                  <td>${c.is_listen ? 'Listen' : 'Speak'}</td>
                  <td>${escHtml(c.text)}</td>
                  <td>${ms(g(c,'cost_vision_process'))}</td>
                  <td>${ms(g(c,'cost_vision_embed'))}</td>
                  <td>${ms(g(c,'cost_vision_feed'))}</td>
                  <td>${ms(g(c,'cost_audio_process'))}</td>
                  <td>${ms(g(c,'cost_audio_embed'))}</td>
                  <td>${ms(g(c,'cost_audio_feed'))}</td>
                  <td>${ms(g(c,'cost_prefill_all'))}</td>
                  <td>${ms(g(c,'cost_llm'))}</td>
                  <td>${ms(g(c,'cost_tts_prep'))}</td>
                  <td>${ms(g(c,'cost_tts'))}</td>
                  <td>${ms(g(c,'cost_token2wav'))}</td>
                  <td>${ms(g(c,'cost_generate_all'))}</td>
                  <td>${ms(g(c,'cost_chunk_total'))}</td>
                  <td>${ms(g(c,'wall_prefill'))}</td>
                  <td>${ms(g(c,'wall_generate'))}</td>
                  <td style="font-weight:600">${ms(g(c,'wall_chunk_total'))}</td>
                  <td>${c.n_tokens}</td>
                  <td>${c.n_tts_tokens}</td>
                  <td>${c.end_of_turn ? 'Y' : ''}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
    container.appendChild(el);
  });
}

const videoChartRendered = {};

function toggleVideo(idx) {
  const body = document.getElementById(`body-${idx}`);
  const arrow = document.getElementById(`arrow-${idx}`);
  const isOpen = body.classList.toggle('open');
  arrow.classList.toggle('open', isOpen);
  if (isOpen && !videoChartRendered[idx]) {
    renderVideoCharts(idx);
    videoChartRendered[idx] = true;
  }
}

function loadSrcVideo(idx) {
  const vr = DATA.video_results[idx];
  document.getElementById(`vp-src-${idx}`).outerHTML =
    `<video controls preload="metadata" src="${vr.source_video_rel}"></video>`;
}

function loadOutVideo(idx) {
  const vr = DATA.video_results[idx];
  document.getElementById(`vp-out-${idx}`).outerHTML =
    `<video controls preload="metadata" src="${vr.rendered_video_rel}"></video>`;
}

function renderVideoCharts(idx) {
  const chunks = DATA.video_results[idx].chunk_results;
  const labels = chunks.map(c => c.chunk_idx);

  // Prefill chart
  new Chart(document.getElementById(`vchart-prefill-${idx}`), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'prefill_all', data: chunks.map(c => g(c,'cost_prefill_all')), borderColor: '#fd79a8', backgroundColor: 'rgba(253,121,168,0.1)', fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: 'generate_all', data: chunks.map(c => g(c,'cost_generate_all')), borderColor: '#74b9ff', borderDash: [6,3], tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: 'vision_embed (ViT)', data: chunks.map(c => g(c,'cost_vision_embed')), borderColor: '#55efc4', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
        { label: 'vision_feed', data: chunks.map(c => g(c,'cost_vision_feed')), borderColor: '#a29bfe', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
        { label: 'audio_feed', data: chunks.map(c => g(c,'cost_audio_feed')), borderColor: '#74b9ff', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        title: { display: true, text: 'Prefill 耗时', color: '#9ba0b8', font: { size: 12 } },
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
      },
      scales: {
        x: { title: { display: true, text: 'Chunk' } },
        y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });

  // Generate chart
  new Chart(document.getElementById(`vchart-generate-${idx}`), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'generate_all', data: chunks.map(c => g(c,'cost_generate_all')), borderColor: '#74b9ff', backgroundColor: 'rgba(116,185,255,0.1)', fill: true, tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: 'prefill_all', data: chunks.map(c => g(c,'cost_prefill_all')), borderColor: '#fd79a8', borderDash: [6,3], tension: 0.3, pointRadius: 2, borderWidth: 2 },
        { label: 'cost_llm', data: chunks.map(c => g(c,'cost_llm')), borderColor: '#00cec9', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
        { label: 'cost_tts', data: chunks.map(c => g(c,'cost_tts')), borderColor: '#fdcb6e', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
        { label: 'cost_token2wav', data: chunks.map(c => g(c,'cost_token2wav')), borderColor: '#a29bfe', tension: 0.3, pointRadius: 2, borderWidth: 1.5 },
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        title: { display: true, text: 'Generate 耗时', color: '#9ba0b8', font: { size: 12 } },
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            afterBody: function(ctx) {
              const c = chunks[ctx[0].dataIndex];
              return `状态: ${c.is_listen ? 'Listen' : 'Speak'}\n文本: ${c.text || '(无)'}`;
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Chunk' } },
        y: { beginAtZero: true, title: { display: true, text: '秒 (s)' } }
      }
    }
  });
}

renderStats();
renderOverallCharts();
renderVideoList();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="生成 Duplex 评估结果静态 HTML 报告")
    parser.add_argument("--results", type=str, default="/root/test/eval_report/eval_results.json",
                        help="评估结果 JSON 文件路径")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = results_path.parent

    print(f"[INFO] 加载结果文件: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    json_str = json.dumps(eval_data, ensure_ascii=False)
    html_content = HTML_TEMPLATE.replace("__JSON_DATA__", json_str)

    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[INFO] 静态 HTML 已生成: {html_path}")
    print(f"[INFO] 目录结构:")
    print(f"       {output_dir}/")
    print(f"       ├── index.html")
    print(f"       ├── eval_results.json")
    print(f"       ├── videos/          (原始视频)")
    print(f"       └── rendered/        (推理结果视频)")
    print(f"[INFO] 启动服务: cd {output_dir} && python -m http.server 8765")


if __name__ == "__main__":
    main()
