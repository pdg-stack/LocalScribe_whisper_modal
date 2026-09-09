// Scan, preferences, preview estimates, and the transcription run itself
// are all wired to the real FastAPI backend (/api/scan, /api/preferences,
// /api/preview, /api/transcribe, /api/jobs/{id}/stream + /cancel). This
// file only owns rendering/DOM state -- backend/estimator.py and
// backend/config.py are the source of truth for GPU rates and RTF tables.

// GPU dropdown labels/rates still live here (backend/config.py is the
// source of truth; kept in sync manually -- small, static, rarely changes).
const GPU_OPTIONS = [
  { id: "T4", label: "T4", rate: 0.59 },
  { id: "L4", label: "L4", rate: 0.8 },
  { id: "A10G", label: "A10G", rate: 1.1 },
  { id: "A100", label: "A100", rate: 2.1 },
  { id: "H100", label: "H100", rate: 3.95 },
];
const RECOMMENDED_GPU = "L4";

// ---------- Formatting helpers ----------

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let val = bytes / 1024;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) { val /= 1024; i++; }
  return `${val.toFixed(1)} ${units[i]}`;
}

function formatDuration(totalSec) {
  totalSec = Math.round(totalSec);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatCost(dollars) {
  return `$${dollars.toFixed(2)}`;
}

// ---------- State ----------

const state = {
  scanData: null,
  scope: "current", // 'current' | 'all'
  expanded: { current: { video: false, audio: false }, all: { video: false, audio: false } },
  selected: {
    current: { video: new Set(), audio: new Set() },
    all: { video: new Set(), audio: new Set() },
  },
};

function scopeStats(scopeKey, type) {
  const files = state.scanData.scopes[scopeKey][type].files;
  const count = files.length;
  const size = files.reduce((s, f) => s + f.size_bytes, 0);
  const duration = files.reduce((s, f) => s + f.duration_sec, 0);
  return { count, size, duration };
}

function combinedScopeStats(scopeKey) {
  const v = scopeStats(scopeKey, "video");
  const a = scopeStats(scopeKey, "audio");
  return {
    count: v.count + a.count,
    size: v.size + a.size,
    duration: v.duration + a.duration,
  };
}

function selectedFilesInActiveScope() {
  const scope = state.scope;
  const result = [];
  for (const type of ["video", "audio"]) {
    const files = state.scanData.scopes[scope][type].files;
    for (const idx of state.selected[scope][type]) {
      result.push({ ...files[idx], type });
    }
  }
  return result;
}

// ---------- Rendering: scan table ----------

const scanTableBody = document.getElementById("scan-table-body");
const selectionSummary = document.getElementById("selection-summary");

function renderScanTable() {
  scanTableBody.innerHTML = "";
  for (const scopeKey of ["current", "all"]) {
    const active = state.scope === scopeKey;
    const label = scopeKey === "current" ? "Current folder" : "Including subfolders";
    const stats = combinedScopeStats(scopeKey);

    const scopeRow = document.createElement("tr");
    scopeRow.className = "scope-row" + (active ? "" : " greyed");
    scopeRow.innerHTML = `
      <td class="col-label">
        <input type="radio" name="scope" value="${scopeKey}" ${active ? "checked" : ""} />
        ${label}
      </td>
      <td>${stats.count}</td>
      <td>${formatBytes(stats.size)}</td>
      <td>${formatDuration(stats.duration)}</td>
    `;
    scopeRow.querySelector('input[type="radio"]').addEventListener("change", () => {
      state.scope = scopeKey;
      renderScanTable();
      renderOptionsVisibility();
    });
    scanTableBody.appendChild(scopeRow);

    for (const type of ["video", "audio"]) {
      const typeStats = scopeStats(scopeKey, type);
      const selectedSet = state.selected[scopeKey][type];
      const allSelected = typeStats.count > 0 && selectedSet.size === typeStats.count;
      const someSelected = selectedSet.size > 0 && !allSelected;
      const expanded = state.expanded[scopeKey][type];

      const typeRow = document.createElement("tr");
      typeRow.className = "type-row" + (active ? "" : " greyed");
      typeRow.innerHTML = `
        <td class="col-label">
          <span class="accordion-toggle">${expanded ? "▾" : "▸"}</span>
          <input type="checkbox" ${allSelected ? "checked" : ""} ${!active ? "disabled" : ""} />
          ${type === "video" ? "Video" : "Audio"}
        </td>
        <td>${typeStats.count}</td>
        <td>${formatBytes(typeStats.size)}</td>
        <td>${formatDuration(typeStats.duration)}</td>
      `;
      const checkbox = typeRow.querySelector('input[type="checkbox"]');
      checkbox.indeterminate = someSelected;

      typeRow.querySelector(".accordion-toggle").addEventListener("click", () => {
        // Accordion works regardless of whether this scope is active --
        // only selecting files requires switching to it first.
        state.expanded[scopeKey][type] = !state.expanded[scopeKey][type];
        renderScanTable();
      });
      checkbox.addEventListener("change", () => {
        if (!active) return;
        const files = state.scanData.scopes[scopeKey][type].files;
        if (checkbox.checked) {
          files.forEach((_, i) => selectedSet.add(i));
        } else {
          selectedSet.clear();
        }
        renderScanTable();
        renderOptionsVisibility();
      });
      scanTableBody.appendChild(typeRow);

      if (expanded) {
        const files = state.scanData.scopes[scopeKey][type].files;
        files.forEach((file, idx) => {
          const fileRow = document.createElement("tr");
          fileRow.className = "file-row" + (active ? "" : " greyed");
          fileRow.innerHTML = `
            <td class="col-label">
              <input type="checkbox" ${selectedSet.has(idx) ? "checked" : ""} ${!active ? "disabled" : ""} />
              ${file.path}
            </td>
            <td></td>
            <td>${formatBytes(file.size_bytes)}</td>
            <td>${formatDuration(file.duration_sec)}</td>
          `;
          fileRow.querySelector('input[type="checkbox"]').addEventListener("change", (e) => {
            if (!active) return;
            if (e.target.checked) selectedSet.add(idx); else selectedSet.delete(idx);
            renderScanTable();
            renderOptionsVisibility();
          });
          scanTableBody.appendChild(fileRow);
        });
      }
    }
  }

  const sel = selectedFilesInActiveScope();
  const size = sel.reduce((s, f) => s + f.size_bytes, 0);
  const duration = sel.reduce((s, f) => s + f.duration_sec, 0);
  selectionSummary.textContent = sel.length
    ? `Selected: ${sel.length} files — ${formatBytes(size)}, ${formatDuration(duration)}`
    : "Selected: 0 files";
}

// ---------- Folder step ----------

const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const folderPathInput = document.getElementById("folder-path");
const folderStatus = document.getElementById("folder-status");
const selectionStep = document.getElementById("selection-step");
const optionsStep = document.getElementById("options-step");
const actionStep = document.getElementById("action-step");

analyzeBtn.addEventListener("click", async () => {
  const path = folderPathInput.value.trim();
  if (!path) {
    folderStatus.textContent = "Enter a folder path first.";
    folderStatus.className = "status error";
    folderStatus.hidden = false;
    return;
  }
  folderStatus.textContent = "Scanning…";
  folderStatus.className = "status";
  folderStatus.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: path }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Scan failed (HTTP ${res.status})`);
    }
    state.scanData = await res.json();
    folderStatus.hidden = true;
    selectionStep.hidden = false;
    renderScanTable();
    await loadPreferences();
    renderOptionsVisibility();
  } catch (err) {
    folderStatus.textContent = err.message || "Could not scan that folder.";
    folderStatus.className = "status error";
    folderStatus.hidden = false;
  } finally {
    analyzeBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  folderPathInput.value = "";
  folderStatus.hidden = true;

  state.scanData = null;
  state.scope = "current";
  state.expanded = { current: { video: false, audio: false }, all: { video: false, audio: false } };
  state.selected = {
    current: { video: new Set(), audio: new Set() },
    all: { video: new Set(), audio: new Set() },
  };

  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  if (activeJobId) {
    fetch(`/api/jobs/${activeJobId}/cancel`, { method: "POST" }).catch(() => {});
    activeJobId = null;
  }

  scanTableBody.innerHTML = "";
  selectionSummary.textContent = "Selected: 0 files";
  selectionStep.hidden = true;
  optionsStep.hidden = true;
  actionStep.hidden = true;

  previewTableBody.innerHTML = "";
  previewSelectionSummary.textContent = "";
  invalidatePreview();
  resetRunAndDiagnostics();
});

// ---------- Options step ----------

const formatChecks = document.getElementById("format-checks");
const executionRadios = document.getElementsByName("execution");
const localFields = document.getElementById("local-fields");
const localResourceInfo = document.getElementById("local-resource-info");
const modalFields = document.getElementById("modal-fields");
const gpuSelect = document.getElementById("gpu-select");
const gpuRecommendation = document.getElementById("gpu-recommendation");
const modelSelect = document.getElementById("model-select");
const tokenIdInput = document.getElementById("modal-token-id");
const tokenSecretInput = document.getElementById("modal-token-secret");
const hfTokenInput = document.getElementById("hf-token");
const cleanupCheck = document.getElementById("cleanup-check");
const previewBtn = document.getElementById("preview-btn");

GPU_OPTIONS.forEach((gpu) => {
  const opt = document.createElement("option");
  opt.value = gpu.id;
  opt.textContent = `${gpu.label} — $${gpu.rate.toFixed(2)}/hr`;
  if (gpu.id === RECOMMENDED_GPU) opt.selected = true;
  gpuSelect.appendChild(opt);
});
gpuRecommendation.textContent = `Recommended: ${RECOMMENDED_GPU} — best cost/throughput balance for Whisper inference`;

// Placeholder hardware summary for the Local execution path -- Phase 2+
// replaces this with a real backend endpoint (e.g. psutil for CPU/RAM,
// torch.cuda.is_available()/nvidia-smi for GPU) reporting this machine's
// actual resources.
const MOCK_LOCAL_RESOURCES = {
  cpu: "Intel Core i7-12700K — 12 cores / 20 threads",
  ram: "32 GB RAM",
  gpu: "NVIDIA GeForce RTX 3080 (10 GB VRAM) — CUDA available",
};
localResourceInfo.textContent =
  `CPU: ${MOCK_LOCAL_RESOURCES.cpu} · RAM: ${MOCK_LOCAL_RESOURCES.ram} · GPU: ${MOCK_LOCAL_RESOURCES.gpu}`;

// ---------- Show/hide toggle for Modal token fields (copy/paste/cut are
// never blocked -- neither <input type=password> nor this toggle does
// anything that would prevent clipboard use) ----------

const EYE_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>';
const EYE_OFF_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 19c-7 0-11-7-11-7a21.86 21.86 0 0 1 5.06-6.06M9.9 4.24A10.94 10.94 0 0 1 12 4c7 0 11 7 11 7a21.86 21.86 0 0 1-2.16 3.19M14.12 14.12a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

document.querySelectorAll(".eye-toggle").forEach((btn) => {
  btn.innerHTML = EYE_ICON;
  btn.addEventListener("click", () => {
    const input = document.getElementById(btn.dataset.target);
    const reveal = input.type === "password";
    input.type = reveal ? "text" : "password";
    btn.innerHTML = reveal ? EYE_OFF_ICON : EYE_ICON;
    const labelEl = document.querySelector(`label[for="${btn.dataset.target}"]`);
    btn.setAttribute("aria-label", (reveal ? "Hide " : "Show ") + (labelEl ? labelEl.textContent : "value"));
  });
});

function currentExecutionMode() {
  for (const r of executionRadios) if (r.checked) return r.value;
  return "local";
}

function renderOptionsVisibility() {
  // Model/formats/execution/Preview are shown as soon as a scan has run,
  // regardless of whether any files are checked yet -- Preview's own
  // enabled/disabled state is what gates on selection (see below).
  const hasScan = !!state.scanData;
  optionsStep.hidden = !hasScan;
  actionStep.hidden = !hasScan;
  updatePreviewEnabled();
}

function updatePreviewEnabled() {
  const hasSelection = selectedFilesInActiveScope().length > 0;
  const hasFormat = [...formatChecks.querySelectorAll("input[type=checkbox]")].some((c) => c.checked);
  const mode = currentExecutionMode();
  let modalOk = true;
  if (mode === "modal") {
    modalOk = !!gpuSelect.value && tokenIdInput.value.trim() && tokenSecretInput.value.trim();
  }
  previewBtn.disabled = !(hasSelection && hasFormat && modalOk);
  // Any change to selection/options invalidates a preview already shown --
  // Begin only ever appears right after a fresh Preview run.
  invalidatePreview();
}

[formatChecks, tokenIdInput, tokenSecretInput, hfTokenInput, gpuSelect, modelSelect].forEach((el) => {
  el.addEventListener("input", () => { updatePreviewEnabled(); savePreferences(); });
  el.addEventListener("change", () => { updatePreviewEnabled(); savePreferences(); });
});
executionRadios.forEach((r) =>
  r.addEventListener("change", () => {
    const mode = currentExecutionMode();
    localFields.hidden = mode !== "local";
    modalFields.hidden = mode !== "modal";
    updatePreviewEnabled();
    savePreferences();
  })
);
cleanupCheck.addEventListener("change", () => { updatePreviewEnabled(); savePreferences(); });

// ---------- Preferences (GET/POST /api/preferences against
// user_prefs.json -- model/formats/execution/gpu, and (at the user's
// request) the Modal + Hugging Face credentials below, all in plaintext,
// gitignored, local-only) ----------

async function savePreferences() {
  const prefs = {
    model: modelSelect.value,
    formats: [...formatChecks.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value),
    execution: currentExecutionMode(),
    gpu: gpuSelect.value,
    cleanup: cleanupCheck.checked,
    // Saved at the user's request so they don't have to re-enter them --
    // stored in plaintext in user_prefs.json (gitignored, local-only).
    modal_token_id: tokenIdInput.value.trim() || null,
    modal_token_secret: tokenSecretInput.value.trim() || null,
    hf_token: hfTokenInput.value.trim() || null,
  };
  try {
    await fetch("/api/preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(prefs),
    });
  } catch (e) { /* best-effort -- a failed save just means prefs won't persist */ }
}

async function loadPreferences() {
  let prefs;
  try {
    const res = await fetch("/api/preferences");
    prefs = res.ok ? await res.json() : null;
  } catch (e) { prefs = null; }
  if (!prefs) return;
  if (prefs.model) modelSelect.value = prefs.model;
  if (prefs.formats) {
    formatChecks.querySelectorAll("input[type=checkbox]").forEach((c) => {
      c.checked = prefs.formats.includes(c.value);
    });
  }
  if (prefs.execution) {
    executionRadios.forEach((r) => { r.checked = r.value === prefs.execution; });
    localFields.hidden = prefs.execution !== "local";
    modalFields.hidden = prefs.execution !== "modal";
  }
  if (prefs.gpu) gpuSelect.value = prefs.gpu;
  if (typeof prefs.cleanup === "boolean") cleanupCheck.checked = prefs.cleanup;
  if (prefs.modal_token_id) tokenIdInput.value = prefs.modal_token_id;
  if (prefs.modal_token_secret) tokenSecretInput.value = prefs.modal_token_secret;
  if (prefs.hf_token) hfTokenInput.value = prefs.hf_token;
  updatePreviewEnabled();
}

// ---------- Preview / Begin / Cancel ----------

const previewPanel = document.getElementById("preview-panel");
const previewTableBody = document.getElementById("preview-table-body");
const beginRow = document.getElementById("begin-row");
const beginBtn = document.getElementById("begin-btn");
const runStep = document.getElementById("run-step");
const progressBar = document.getElementById("progress-bar");
const cancelBtn = document.getElementById("cancel-btn");
const logBox = document.getElementById("log-box");
const diagnosticsStep = document.getElementById("diagnostics-step");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsTableBody = document.getElementById("diagnostics-table-body");
const previewSelectionSummary = document.getElementById("preview-selection-summary");

function invalidatePreview() {
  // Begin only ever appears right after a fresh Preview -- any change to
  // selection/options hides both the estimate panel and Begin again.
  previewPanel.hidden = true;
  beginRow.hidden = true;
}

function resetRunAndDiagnostics() {
  runStep.hidden = true;
  diagnosticsStep.hidden = true;
  logBox.innerHTML = "";
  progressBar.value = 0;
  diagnosticsTableBody.innerHTML = "";
  cancelBtn.disabled = false;
  cancelBtn.textContent = "Cancel";
}

let activeJobId = null;
let activeEventSource = null;
let phaseStatus = {}; // { [phaseName]: { doneCount, failedCount, skippedCount, expectedCount } }, live-tracked during a run

// Mirrors backend/estimator.py's compute_steps() phase-assignment logic
// (which files get which phases) so the frontend knows how many
// (file, phase) instances to expect before marking a phase row "done".
function computeExpectedPhaseCounts(files, execution, cleanup) {
  const counts = {};
  const bump = (name) => { counts[name] = (counts[name] || 0) + 1; };
  files.forEach((f, i) => {
    if (f.type === "video") bump("Audio Extraction");
    if (execution !== "modal" && i === 0) bump("Whisper Model Setup");
    if (execution === "modal") bump("Modal.com Setup & Model Install");
    bump("Transcription");
    if (execution === "modal") bump("Download Transcript to Local");
    if (cleanup && f.type === "video") bump("Cleanup");
  });
  return counts;
}

function setPhaseIcon(phaseName, state) {
  const el = previewTableBody.querySelector(`.phase-icon[data-phase="${phaseName}"]`);
  if (!el) return;
  el.className = "phase-icon" + (state ? ` phase-icon--${state}` : "");
  el.textContent = state === "done" ? "✓" : state === "failed" ? "✗" : state === "warning" ? "!" : "";
}

// Once every (file, phase) instance for this phase has been accounted for
// (succeeded, failed, or skipped -- a file already failed in an earlier
// phase is skipped in this one), finalize its icon: green only if nothing
// failed, red only if nothing succeeded, yellow for a genuine mix of both
// (partial success) per the user's "partial completions = yellow warning"
// requirement. Skips alone (e.g. a phase every file happened to skip)
// don't count as either success or failure.
function finalizePhaseIfComplete(phaseName) {
  const st = phaseStatus[phaseName];
  if (!st) return;
  const accountedFor = st.doneCount + st.failedCount + st.skippedCount;
  if (accountedFor < st.expectedCount) return;
  if (st.failedCount > 0 && st.doneCount > 0) {
    setPhaseIcon(phaseName, "warning");
  } else if (st.failedCount > 0) {
    setPhaseIcon(phaseName, "failed");
  } else if (st.doneCount > 0) {
    setPhaseIcon(phaseName, "done");
  }
}

function buildTranscribeRequest() {
  const files = selectedFilesInActiveScope().map((f) => ({
    path: f.path, type: f.type, duration_sec: f.duration_sec,
  }));
  const mode = currentExecutionMode();
  return {
    folder_path: state.scanData.folder,
    files,
    model: modelSelect.value,
    formats: [...formatChecks.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value),
    execution: mode,
    gpu: mode === "modal" ? gpuSelect.value : null,
    cleanup: cleanupCheck.checked,
    // Also saved to user_prefs.json (plaintext) by savePreferences() so
    // the user doesn't have to re-enter them each time.
    modal_token_id: mode === "modal" ? tokenIdInput.value.trim() : null,
    modal_token_secret: mode === "modal" ? tokenSecretInput.value.trim() : null,
    // Applies to both execution modes -- Hugging Face rate-limits the
    // model download whether it happens locally or inside a Modal
    // container, not just for Modal execution.
    hf_token: hfTokenInput.value.trim() || null,
  };
}

previewBtn.addEventListener("click", async () => {
  // Re-running Preview means any earlier run's log/diagnostics are stale.
  resetRunAndDiagnostics();

  const req = buildTranscribeRequest();
  const selected = selectedFilesInActiveScope();
  const size = selected.reduce((s, f) => s + f.size_bytes, 0);
  const duration = selected.reduce((s, f) => s + f.duration_sec, 0);
  const modeLabel = req.execution === "modal" ? `Modal.com (${req.gpu})` : "Local";
  previewSelectionSummary.textContent =
    `${req.files.length} files — ${formatBytes(size)}, ${formatDuration(duration)} · ${req.model} model · ${req.formats.join(", ")} · ${modeLabel}`;

  previewBtn.disabled = true;
  try {
    const res = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Preview failed (HTTP ${res.status})`);
    }
    const { phases, total } = await res.json();

    previewTableBody.innerHTML = "";
    for (const p of phases) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="phase-icon" data-phase="${p.name}"></span>${p.name}</td><td>${formatDuration(p.sec)}</td><td>${formatCost(p.cost)}</td>`;
      previewTableBody.appendChild(tr);
    }
    const totalRow = document.createElement("tr");
    totalRow.innerHTML = `<td><strong>Total</strong></td><td><strong>${formatDuration(total.sec)}</strong></td><td><strong>${formatCost(total.cost)}</strong></td>`;
    previewTableBody.appendChild(totalRow);

    previewPanel.hidden = false;
    beginRow.hidden = false;
  } catch (err) {
    previewSelectionSummary.textContent = err.message || "Could not compute a preview.";
  } finally {
    previewBtn.disabled = false;
  }
});

function addLogLine(text, isFail) {
  const line = document.createElement("div");
  line.className = "log-line" + (isFail ? " fail" : "");
  line.textContent = text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

beginBtn.addEventListener("click", async () => {
  runStep.hidden = false;
  diagnosticsStep.hidden = true;
  logBox.innerHTML = "";
  progressBar.value = 0;
  cancelBtn.disabled = false;
  cancelBtn.textContent = "Cancel";
  beginBtn.disabled = true;

  const req = buildTranscribeRequest();
  const expected = computeExpectedPhaseCounts(req.files, req.execution, req.cleanup);
  phaseStatus = {};
  for (const [name, expectedCount] of Object.entries(expected)) {
    phaseStatus[name] = { doneCount: 0, failedCount: 0, skippedCount: 0, expectedCount };
    setPhaseIcon(name, null); // clear any icon left from a previous run
  }

  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Could not start the job (HTTP ${res.status})`);
    }
    const { job_id } = await res.json();
    activeJobId = job_id;

    const es = new EventSource(`/api/jobs/${job_id}/stream`);
    activeEventSource = es;
    es.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      if (event.event === "log") {
        addLogLine(event.text, !!event.fail);
      } else if (event.event === "step_start") {
        const st = phaseStatus[event.step];
        if (st && st.doneCount + st.failedCount + st.skippedCount < st.expectedCount) {
          setPhaseIcon(event.step, "spinner");
        }
      } else if (event.event === "step_done") {
        const st = phaseStatus[event.step];
        if (st) {
          st.doneCount++;
          finalizePhaseIfComplete(event.step);
        }
      } else if (event.event === "step_failed") {
        const st = phaseStatus[event.step];
        if (st) {
          st.failedCount++;
          finalizePhaseIfComplete(event.step);
        }
      } else if (event.event === "step_skipped") {
        const st = phaseStatus[event.step];
        if (st) {
          st.skippedCount++;
          finalizePhaseIfComplete(event.step);
        }
      } else if (event.event === "progress") {
        progressBar.value = event.percent;
      } else if (event.event === "done") {
        es.close();
        activeEventSource = null;
        activeJobId = null;
        cancelBtn.textContent = "Cancel";
        cancelBtn.disabled = true;
        beginBtn.disabled = false;
        showDiagnostics(event);
      }
    };
    es.onerror = () => {
      es.close();
      activeEventSource = null;
      activeJobId = null;
      cancelBtn.disabled = true;
      beginBtn.disabled = false;
      addLogLine("Lost connection to the server.", true);
    };
  } catch (err) {
    addLogLine(err.message || "Could not start the job.", true);
    cancelBtn.disabled = true;
    beginBtn.disabled = false;
  }
});

cancelBtn.addEventListener("click", async () => {
  if (!activeJobId) return;
  cancelBtn.disabled = true;
  cancelBtn.textContent = "Cancelling…";
  addLogLine("Cancelling… stopping the current step and cleaning up.");
  try {
    await fetch(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
  } catch (e) { /* the stream's onerror handler covers a dropped connection */ }
  // Button stays disabled/"Cancelling…" until the "done" SSE event lands --
  // the backend interrupts the in-flight step immediately, but Cleanup
  // still runs unconditionally afterward, so "done" (and this button
  // resetting to "Cancel") can lag a little behind the click.
});

function showDiagnostics(result) {
  diagnosticsStep.hidden = false;
  const total = result.succeeded + result.failed + result.skipped;
  diagnosticsSummary.textContent = result.cancelled
    ? `Cancelled — ${result.succeeded + result.failed} of ${total} files processed (${result.succeeded} succeeded, ${result.failed} failed)`
    : `${total} files processed — ${result.succeeded} succeeded, ${result.failed} failed`;

  diagnosticsTableBody.innerHTML = `
    <tr><td>Processing time</td><td>${formatDuration(result.total_sec)}</td><td>${formatDuration(result.per_file_sec)}</td><td>${formatDuration(result.per_minute_sec)}</td></tr>
    <tr><td>Cost</td><td>${formatCost(result.total_cost)}</td><td>${formatCost(result.per_file_cost)}</td><td>${formatCost(result.per_minute_cost)}</td></tr>
  `;
}
