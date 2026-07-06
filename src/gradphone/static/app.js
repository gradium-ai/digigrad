// gradphone — operator's & tenant's dashboard.
// Polls /ui/calls/live and /ui/history (tenant-scoped on the server side),
// posts to /ui/dial, opens result modal on completion.

const LIVE_POLL_MS = 3000;
const HISTORY_POLL_MS = 10000;
const RESULT_POLL_MS = 5000;
const RESULT_DEADLINE_MS = 10 * 60 * 1000;

const $ = (id) => document.getElementById(id);
const IS_OPERATOR = document.body.dataset.role === "operator";

// ─── Toast helper ────────────────────────────────────────
let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("visible"), 4000);
}

// ─── Modal ───────────────────────────────────────────────
function openResultModal(room, result) {
  $("modalSubtitle").textContent = room;
  const body = $("modalBody");
  const br = (result && result.business_result) || {};
  const rows = [
    ["status", br.status || "—"],
    ["answer", br.answer || "—"],
    ["confidence", br.confidence || "—"],
    ["duration", `${(result.duration_seconds || 0).toFixed(1)}s`],
    ["framework", result.framework || "—"],
  ];
  if (result.answered_by) rows.push(["answered by", result.answered_by]);
  if (result.twilio_call_status) rows.push(["twilio status", result.twilio_call_status]);
  body.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
  $("modalLatency").innerHTML = latencyHtml(result.latency);
  $("modalbackdrop").classList.add("visible");
}
function closeModal() { $("modalbackdrop").classList.remove("visible"); }
window.closeModal = closeModal;

// Open the result modal for any call by room (used by history rows).
async function openRoomResult(room) {
  const data = await getJson(`/ui/result/${encodeURIComponent(room)}`);
  if (!data) return;
  if (data.status === "complete") {
    openResultModal(room, data.result);
  } else {
    toast(`No result yet for ${room} (${data.status || "pending"})`);
  }
}
window.openRoomResult = openRoomResult;

// ─── Latency breakdown ───────────────────────────────────
// The cascade stages, in speaking order. STT + TTS are Gradium models;
// LLM is the configured text model; tool is bridge-measured I/O.
const LAT_STAGES = [
  { key: "stt",  label: "STT",  cls: "gradium" },
  { key: "llm",  label: "LLM",  cls: "model" },
  { key: "tool", label: "Tool", cls: "tool" },
  { key: "tts",  label: "TTS",  cls: "gradium" },
];

function fmtMs(v) {
  if (v === null || v === undefined) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`;
}

function latencyHtml(latency) {
  if (!latency || !latency.turns || !latency.turns.length) {
    return `<div class="lat-empty">No per-turn latency captured for this call.</div>`;
  }
  const agg = latency.aggregates || {};
  const med = (k) => (agg[k] && agg[k].median != null ? agg[k].median : null);
  // Stacked bar of median stage times (only stages with a value).
  const segs = LAT_STAGES.map((s) => ({ ...s, ms: med(s.key) })).filter((s) => s.ms != null);
  const barTotal = segs.reduce((a, s) => a + s.ms, 0) || 1;
  const bar = segs.map((s) =>
    `<span class="lat-seg ${s.cls}" style="width:${(s.ms / barTotal * 100).toFixed(1)}%"
           title="${s.label} median ${fmtMs(s.ms)}"></span>`
  ).join("");
  const legend = LAT_STAGES.map((s) =>
    `<span class="lat-key"><i class="lat-dot ${s.cls}"></i>${s.label}
       <b>${fmtMs(med(s.key))}</b></span>`
  ).join("");
  const respMed = med("response");

  const rows = latency.turns
    .filter((t) => t.turn >= 0)
    .map((t) => `
      <tr>
        <td>${t.turn}</td>
        <td>${fmtMs(t.stt_ms)}</td>
        <td>${fmtMs(t.llm_ms)}</td>
        <td>${fmtMs(t.tool_ms)}${toolNames(t.tools)}</td>
        <td>${fmtMs(t.tts_ms)}</td>
        <td class="lat-total">${fmtMs(t.response_ms)}</td>
      </tr>`).join("");

  return `
    <div class="lat-head">
      <span class="lat-title">Latency — median per turn</span>
      ${respMed != null ? `<span class="lat-resp">response ${fmtMs(respMed)}</span>` : ""}
    </div>
    <div class="lat-bar">${bar}</div>
    <div class="lat-legend">${legend}</div>
    <table class="lat-table">
      <thead><tr><th>Turn</th><th>STT</th><th>LLM</th><th>Tool</th><th>TTS</th><th>Response</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="lat-caption">
      Measured at the bridge from gradbot event arrival times. STT + TTS are
      Gradium; LLM is the configured text model; tool is dispatch→result.
    </div>`;
}

function toolNames(tools) {
  if (!tools || !tools.length) return "";
  const names = tools.map((t) => t.name).join(", ");
  return `<span class="lat-tools">${escapeHtml(names)}</span>`;
}

// Compact chip for a call's median response latency (live + history rows).
function latencyChip(latency) {
  const m = latency && latency.aggregates && latency.aggregates.response;
  if (!m || m.median == null) return "";
  return `<span class="lat-chip" title="median response (transcript→first audio)">⧗ ${fmtMs(m.median)}</span>`;
}

// ─── HTML escaping ───────────────────────────────────────
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ─── Fetch helpers ───────────────────────────────────────
async function getJson(path) {
  const r = await fetch(path, { credentials: "same-origin" });
  if (r.status === 401 || r.status === 307) { window.location.href = "/ui/login"; return null; }
  return r.json();
}
async function postJson(path, body) {
  const r = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status === 401 || r.status === 307) { window.location.href = "/ui/login"; return null; }
  return r.json();
}

// ─── Live calls ──────────────────────────────────────────
function renderLive(calls) {
  $("livecount").textContent = String(calls.length).padStart(2, "0");
  $("livedot").classList.toggle("live", calls.length > 0);

  const container = $("livecalls");
  if (!calls.length) {
    container.innerHTML = `<div class="empty">no calls in flight</div>`;
    return;
  }
  const rows = calls.map((c) => `
    <tr>
      <td><span class="dest">${escapeHtml(c.destination || "?")}</span>
          <span class="lang-tag">${escapeHtml(c.language || "en")}</span>
          <span class="task">${escapeHtml(c.business_name || c.room || "")}</span></td>
      <td><span class="phase">${escapeHtml(c.phase || "—")}</span></td>
      <td><span class="ts">${(c.age_seconds || 0).toFixed(0)}s</span> ${latencyChip(c.latency)}</td>
    </tr>
  `).join("");
  container.innerHTML = `
    <table>
      <thead><tr><th>Destination</th><th>Phase</th><th>Age</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function refreshLive() {
  const data = await getJson("/ui/calls/live");
  if (!data) return;
  renderLive(data.calls || []);
}

// ─── History ─────────────────────────────────────────────
function renderHistory(calls) {
  $("histcount").textContent = String(calls.length).padStart(2, "0");

  const container = $("history");
  if (!calls.length) {
    container.innerHTML = `<div class="empty">no calls yet — try one</div>`;
    return;
  }
  const rows = calls.map((c) => {
    const status = c.status || "pending";
    const ts = (c.started_at || "").slice(0, 16).replace("T", " ");
    const dur = c.duration_seconds ? `${c.duration_seconds.toFixed(0)}s` : "—";
    const answer = c.answer ? `<span class="answer">${escapeHtml(c.answer)}</span>` : "";
    const audioBtn = c.room
      ? `<a href="#" onclick="event.stopPropagation(); event.preventDefault(); playAudio('${escapeHtml(c.room)}')">▶ audio</a>`
      : "";
    const roomAttr = c.room ? `onclick="openRoomResult('${escapeHtml(c.room)}')" class="clickable" title="Latency + result"` : "";
    return `
      <tr ${roomAttr}>
        <td><span class="dest">${escapeHtml(c.destination || "?")}</span>
            <span class="lang-tag">${escapeHtml(c.language || "en")}</span>
            <span class="task">${escapeHtml(c.task || "")}</span>
            ${answer}</td>
        <td><span class="status ${status}">${escapeHtml(status)}</span></td>
        <td><span class="ts">${ts}</span><br><span class="ts">${dur}</span></td>
        <td>${audioBtn}</td>
      </tr>
    `;
  }).join("");
  container.innerHTML = `
    <table>
      <thead><tr><th>Destination · Task</th><th>Status</th><th>When</th><th>Audio</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function refreshHistory() {
  const data = await getJson("/ui/history");
  if (!data) return;
  renderHistory(data.calls || []);
}

// ─── Audio playback ──────────────────────────────────────
let activeAudio = null;
window.playAudio = function (room) {
  if (activeAudio) { activeAudio.pause(); activeAudio = null; }
  const audio = new Audio(`/ui/audio/${encodeURIComponent(room)}`);
  audio.controls = true;
  audio.play().catch((e) => toast(`Audio: ${e.message}`));
  activeAudio = audio;
};

// ─── Dial form ───────────────────────────────────────────
$("dialform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const btn = f.querySelector("button");
  const notice = $("dialnotice");
  notice.className = "dial-notice active";
  notice.textContent = "Dispatching…";
  btn.disabled = true;

  const payload = {
    to: f.to.value,
    reason: f.reason.value,
    language: f.language.value,
    business_name: f.business_name.value,
  };
  const out = await postJson("/ui/dial", payload);
  if (!out) { btn.disabled = false; return; }
  if (!out.ok) {
    notice.className = "dial-notice active error";
    notice.textContent = out.error || "Dispatch failed";
    btn.disabled = false;
    return;
  }
  const room = out.room;
  notice.textContent = `Ringing… room ${room}`;
  toast(`Calling ${payload.to}`);
  await refreshLive();
  btn.disabled = false;
  f.reason.value = "";

  pollResultThenModal(room);
});

async function pollResultThenModal(room) {
  const deadline = Date.now() + RESULT_DEADLINE_MS;
  while (Date.now() < deadline) {
    const data = await getJson(`/ui/result/${encodeURIComponent(room)}`);
    if (!data) return;
    if (data.status === "complete") {
      openResultModal(room, data.result);
      await refreshHistory();
      await refreshLive();
      return;
    }
    if (data.status === "missing" || data.status === "error") {
      toast(`Result error: ${data.error || data.status}`);
      return;
    }
    await new Promise((r) => setTimeout(r, RESULT_POLL_MS));
  }
  toast("Result poll timed out");
}

// ─── Voice clone (tenant only) ───────────────────────────
async function refreshVoice() {
  const el = $("voicestatus");
  if (!el) return;
  const data = await getJson("/ui/voice");
  if (!data || !data.ok) return;
  if (data.voice_id) {
    el.innerHTML = `<div class="voice-pill on">CLONED</div>
      <div class="voice-uid">${escapeHtml(data.voice_name || "—")}</div>
      <div class="voice-uid faint">${escapeHtml(data.voice_id)}</div>`;
  } else {
    el.innerHTML = `<div class="voice-pill off">DEFAULT</div>
      <div class="voice-uid faint">No clone — uses the language default</div>`;
  }
}

const voiceForm = $("voiceform");
if (voiceForm) {
  voiceForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = $("voicefile");
    const notice = $("voicenotice");
    const btn = voiceForm.querySelector("button.voice-btn");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      notice.className = "voice-notice active error";
      notice.textContent = "Pick an audio file first.";
      return;
    }
    notice.className = "voice-notice active";
    notice.textContent = `Uploading ${file.name} (${(file.size/1024).toFixed(0)} KB) and cloning…`;
    btn.disabled = true;

    const fd = new FormData();
    fd.append("audio", file);
    const r = await fetch("/ui/voice", {
      method: "POST",
      credentials: "same-origin",
      body: fd,
    });
    const data = await r.json().catch(() => ({}));
    btn.disabled = false;
    if (!data.ok) {
      notice.className = "voice-notice active error";
      notice.textContent = `Failed: ${data.error || r.statusText}`;
      return;
    }
    notice.className = "voice-notice active";
    notice.textContent = `Cloned. Future calls will use ${data.voice_name}.`;
    fileInput.value = "";
    refreshVoice();
  });
}

window.clearVoice = async function () {
  if (!confirm("Clear your custom voice and revert to the language default?")) return;
  const r = await postJson("/ui/voice/clear", {});
  if (!r) return;
  if (r.ok) {
    toast("Voice cleared.");
    refreshVoice();
  } else {
    toast(`Clear failed: ${r.error || ""}`);
  }
};

// ─── Boot ────────────────────────────────────────────────
refreshLive();
refreshHistory();
refreshVoice();
setInterval(refreshLive, LIVE_POLL_MS);
setInterval(refreshHistory, HISTORY_POLL_MS);
