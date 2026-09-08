// Phase 2+: scan and preferences are wired to the real FastAPI backend
// (/api/scan, /api/preferences). Phase 3/4 still replace the Preview/Begin
// estimate + run logic with real /api/preview, /api/transcribe,
// /api/jobs/{id}/stream calls.

const GPU_OPTIONS = [
  { id: "T4", label: "T4", rate: 0.59 },
  { id: "L4", label: "L4", rate: 0.8 },
  { id: "A10G", label: "A10G", rate: 1.1 },
  { id: "A100", label: "A100", rate: 2.1 },
  { id: "H100", label: "H100", rate: 3.95 },
];
const RECOMMENDED_GPU = "L4";

// Approximate real-time-factor (processing seconds per second of audio),
// calibrated for beam_size=5. Backend estimator.py (Phase 3/4) is the
// source of truth; these are placeholders for the Preview panel demo.
const RTF_LOCAL_CPU = {
  tiny: 0.1, base: 0.15, small: 0.25, medium: 0.45, "large-v3": 0.7, "large-v3-turbo": 0.35,
};
const RTF_MODAL_GPU = {
  tiny: 0.015, base: 0.02, small: 0.03, medium: 0.05, "large-v3": 0.08, "large-v3-turbo": 0.045,
};

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
  run: { cancelling: false, timer: null },
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
  state.run.cancelling = false;

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

[formatChecks, tokenIdInput, tokenSecretInput, gpuSelect, modelSelect].forEach((el) => {
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
// user_prefs.json — credentials are NEVER sent or persisted, only
// model/formats/execution/gpu) ----------

async function savePreferences() {
  const prefs = {
    model: modelSelect.value,
    formats: [...formatChecks.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value),
    execution: currentExecutionMode(),
    gpu: gpuSelect.value,
    cleanup: cleanupCheck.checked,
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

// Modal.com phases are per-file since each transcription call is an
// ephemeral instance (see transcription/modal_app.py in the plan) --
// setup/model-install happens fresh per file, same for the download back.
// These are placeholder constants for the Preview demo; Phase 3/4 replace
// them with backend estimator.py's real figures.
const MODAL_SETUP_SEC = 20; // cold start: spin instance + install/load model
const MODAL_DOWNLOAD_SEC = 2; // transcript result back to local

function computeEstimate() {
  const files = selectedFilesInActiveScope();
  const model = modelSelect.value;
  const mode = currentExecutionMode();
  const rtfTable = mode === "modal" ? RTF_MODAL_GPU : RTF_LOCAL_CPU;
  const rtf = rtfTable[model];
  const gpu = mode === "modal" ? GPU_OPTIONS.find((g) => g.id === gpuSelect.value) : null;
  const doCleanup = cleanupCheck.checked;

  const PHASE_ORDER = mode === "modal"
    ? ["Audio Extraction", "Modal.com Setup & Model Install", "Transcription", "Download Transcript to Local", "Cleanup"]
    : ["Audio Extraction", "Transcription", "Cleanup"];

  // Flat, file-major ordered list -- mirrors the real per-file pipeline
  // (extract -> [modal setup] -> transcribe -> [download] -> cleanup) so
  // it can drive both the Preview totals and the weighted progress bar.
  const steps = [];
  for (const f of files) {
    if (f.type === "video") {
      steps.push({ file: f, name: "Audio Extraction", sec: (f.duration_sec / 60) * 2, cost: 0 }); // ~2s/min source
    }
    if (mode === "modal") {
      steps.push({ file: f, name: "Modal.com Setup & Model Install", sec: MODAL_SETUP_SEC, cost: (MODAL_SETUP_SEC / 3600) * gpu.rate });
    }
    const transcriptionSec = f.duration_sec * rtf;
    steps.push({
      file: f,
      name: "Transcription",
      sec: transcriptionSec,
      cost: mode === "modal" ? (transcriptionSec / 3600) * gpu.rate : 0,
    });
    if (mode === "modal") {
      steps.push({ file: f, name: "Download Transcript to Local", sec: MODAL_DOWNLOAD_SEC, cost: 0 });
    }
    if (doCleanup && f.type === "video") {
      steps.push({ file: f, name: "Cleanup", sec: 1, cost: 0 });
    }
  }

  const totalsByName = {};
  for (const name of PHASE_ORDER) totalsByName[name] = { sec: 0, cost: 0 };
  for (const s of steps) { totalsByName[s.name].sec += s.sec; totalsByName[s.name].cost += s.cost; }

  const phases = PHASE_ORDER
    .filter((name) => name !== "Cleanup" || doCleanup)
    .map((name) => ({ name, sec: totalsByName[name].sec, cost: totalsByName[name].cost }));

  return { phases, steps, files };
}

function groupStepsByFile(steps) {
  const groups = [];
  let current = null;
  for (const s of steps) {
    if (!current || current.file !== s.file) {
      current = { file: s.file, steps: [] };
      groups.push(current);
    }
    current.steps.push(s);
  }
  return groups;
}

function stepLogLabel(step, model, mode) {
  switch (step.name) {
    case "Audio Extraction": return `Extracting audio: ${step.file.path}`;
    case "Modal.com Setup & Model Install": return `Setting up Modal.com & installing ${model} model: ${step.file.path}`;
    case "Transcription": return `Transcribing (${mode}, ${model}): ${step.file.path}`;
    case "Download Transcript to Local": return `Downloading transcript: ${step.file.path}`;
    case "Cleanup": return `Cleaning up intermediate audio: ${step.file.path}`;
    default: return `${step.name}: ${step.file.path}`;
  }
}

previewBtn.addEventListener("click", () => {
  // Phase 3/4 replace this with: await fetch('/api/preview', {...})
  // Re-running Preview means any earlier run's log/diagnostics are stale.
  resetRunAndDiagnostics();

  const estimate = computeEstimate();

  const size = estimate.files.reduce((s, f) => s + f.size_bytes, 0);
  const duration = estimate.files.reduce((s, f) => s + f.duration_sec, 0);
  const formats = [...formatChecks.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value).join(", ");
  const mode = currentExecutionMode();
  const modeLabel = mode === "modal" ? `Modal.com (${gpuSelect.value})` : "Local";
  previewSelectionSummary.textContent =
    `${estimate.files.length} files — ${formatBytes(size)}, ${formatDuration(duration)} · ${modelSelect.value} model · ${formats} · ${modeLabel}`;

  previewTableBody.innerHTML = "";
  let totalSec = 0, totalCost = 0;
  for (const p of estimate.phases) {
    totalSec += p.sec;
    totalCost += p.cost;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${p.name}</td><td>${formatDuration(p.sec)}</td><td>${formatCost(p.cost)}</td>`;
    previewTableBody.appendChild(tr);
  }
  const totalRow = document.createElement("tr");
  totalRow.innerHTML = `<td><strong>Total</strong></td><td><strong>${formatDuration(totalSec)}</strong></td><td><strong>${formatCost(totalCost)}</strong></td>`;
  previewTableBody.appendChild(totalRow);

  previewPanel.hidden = false;
  beginRow.hidden = false;
});

function addLogLine(text, isFail) {
  const line = document.createElement("div");
  line.className = "log-line" + (isFail ? " fail" : "");
  line.textContent = text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

beginBtn.addEventListener("click", () => {
  // Phase 3/4 replace this simulated run with real /api/transcribe +
  // /api/jobs/{id}/stream (SSE) + /api/jobs/{id}/cancel. The progress bar
  // is weighted by each step's estimated seconds (same figures as the
  // Preview table), not just a raw file count.
  const estimate = computeEstimate();
  const groups = groupStepsByFile(estimate.steps);
  const totalSec = estimate.steps.reduce((s, st) => s + st.sec, 0) || 1;

  runStep.hidden = false;
  diagnosticsStep.hidden = true;
  logBox.innerHTML = "";
  progressBar.value = 0;
  state.run.cancelling = false;
  cancelBtn.disabled = false;
  cancelBtn.textContent = "Cancel";

  const model = modelSelect.value;
  const mode = currentExecutionMode();
  let elapsedSec = 0;
  let gi = 0, si = 0;
  let succeeded = 0, failed = 0;

  function processNext() {
    if (state.run.cancelling) {
      addLogLine("Cancelled — remaining files were not started.");
      finishRun(true);
      return;
    }
    if (gi >= groups.length) {
      finishRun(false);
      return;
    }
    const group = groups[gi];
    if (si >= group.steps.length) {
      addLogLine(`Done: ${group.file.path}`);
      succeeded++;
      gi++; si = 0;
      processNext();
      return;
    }
    const step = group.steps[si];
    addLogLine(stepLogLabel(step, model, mode));

    const isLastFile = gi === groups.length - 1;
    const willFail = step.name === "Transcription" && isLastFile && groups.length > 2; // demo-only

    setTimeout(() => {
      elapsedSec += step.sec;
      progressBar.value = Math.min(100, Math.round((elapsedSec / totalSec) * 100));
      if (willFail) {
        addLogLine(`Failed: ${group.file.path} — ffmpeg could not read this file`, true);
        failed++;
        gi++; si = 0; // skip this file's remaining steps
      } else {
        si++;
      }
      processNext();
    }, 120);
  }

  function finishRun(wasCancelled) {
    // Job is over (finished or successfully cancelled) -- label reverts to
    // "Cancel" (not left reading "Cancelling…") and stays disabled since
    // there's nothing left to cancel.
    cancelBtn.textContent = "Cancel";
    cancelBtn.disabled = true;
    showDiagnostics(estimate, succeeded, failed, groups.length - succeeded - failed, wasCancelled);
  }

  processNext();
});

cancelBtn.addEventListener("click", () => {
  state.run.cancelling = true;
  cancelBtn.disabled = true;
  cancelBtn.textContent = "Cancelling…";
  addLogLine("Cancelling… the current file is finishing first.");
});

function showDiagnostics(estimate, succeeded, failed, skipped, wasCancelled) {
  diagnosticsStep.hidden = false;
  const total = succeeded + failed + skipped;
  diagnosticsSummary.textContent = wasCancelled
    ? `Cancelled — ${succeeded + failed} of ${total} files processed (${succeeded} succeeded, ${failed} failed)`
    : `${total} files processed — ${succeeded} succeeded, ${failed} failed`;

  const totalSec = estimate.phases.reduce((s, p) => s + p.sec, 0);
  const totalCost = estimate.phases.reduce((s, p) => s + p.cost, 0);
  const processed = succeeded + failed || 1;
  const totalMinutes = estimate.files.reduce((s, f) => s + f.duration_sec, 0) / 60 || 1;

  diagnosticsTableBody.innerHTML = `
    <tr><td>Processing time</td><td>${formatDuration(totalSec)}</td><td>${formatDuration(totalSec / processed)}</td><td>${formatDuration(totalSec / totalMinutes)}</td></tr>
    <tr><td>Cost</td><td>${formatCost(totalCost)}</td><td>${formatCost(totalCost / processed)}</td><td>${formatCost(totalCost / totalMinutes)}</td></tr>
  `;
}
