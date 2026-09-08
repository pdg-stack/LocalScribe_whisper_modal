// Phase 1 prototype: everything below runs against MOCK_SCAN / client-side
// estimate math. Phase 2 replaces the scan with a real /api/scan call;
// Phase 3/4 replace the Preview/Begin estimate + run logic with real
// /api/preview, /api/transcribe, /api/jobs/{id}/stream calls; Phase 2 also
// wires preferences to GET/POST /api/preferences instead of localStorage.

// ---------- Mock data ----------

function mockFile(path, durationSec, sizeBytes) {
  return { path, duration_sec: durationSec, size_bytes: sizeBytes };
}

const MOCK_SCAN = {
  folder: "D:\\Videos\\SampleProject",
  scopes: {
    current: {
      video: {
        files: [
          mockFile("intro.mp4", 92, 18_000_000),
          mockFile("subdir\\outro.mov", 64, 40_000_000),
          mockFile("raw_footage.mkv", 720, 900_000_000),
          mockFile("interview_pt1.mp4", 1980, 210_000_000),
        ],
      },
      audio: {
        files: [
          mockFile("notes.mp3", 480, 8_000_000),
          mockFile("call_recording.wav", 1620, 160_000_000),
        ],
      },
    },
    all: {
      video: {
        files: [
          mockFile("intro.mp4", 92, 18_000_000),
          mockFile("subdir\\outro.mov", 64, 40_000_000),
          mockFile("raw_footage.mkv", 720, 900_000_000),
          mockFile("interview_pt1.mp4", 1980, 210_000_000),
          mockFile("archive\\2024\\keynote.mp4", 3600, 1_200_000_000),
          mockFile("archive\\2024\\panel.mp4", 3900, 1_300_000_000),
          mockFile("archive\\2025\\demo.mov", 1500, 500_000_000),
        ],
      },
      audio: {
        files: [
          mockFile("notes.mp3", 480, 8_000_000),
          mockFile("call_recording.wav", 1620, 160_000_000),
          mockFile("archive\\2024\\podcast_ep1.mp3", 2700, 55_000_000),
          mockFile("archive\\2025\\podcast_ep2.mp3", 2800, 57_000_000),
        ],
      },
    },
  },
};

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
    scopeRow.className = "scope-row";
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
        if (!active) return;
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
const folderPathInput = document.getElementById("folder-path");
const folderStatus = document.getElementById("folder-status");
const selectionStep = document.getElementById("selection-step");
const optionsStep = document.getElementById("options-step");
const actionStep = document.getElementById("action-step");

analyzeBtn.addEventListener("click", () => {
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

  // Phase 2 replaces this with: await fetch('/api/scan', {method:'POST', body: JSON.stringify({folder_path: path})})
  setTimeout(() => {
    state.scanData = MOCK_SCAN;
    folderStatus.hidden = true;
    selectionStep.hidden = false;
    renderScanTable();
    loadPreferences();
    renderOptionsVisibility();
  }, 300);
});

// ---------- Options step ----------

const formatChecks = document.getElementById("format-checks");
const executionRadios = document.getElementsByName("execution");
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

function currentExecutionMode() {
  for (const r of executionRadios) if (r.checked) return r.value;
  return "local";
}

function renderOptionsVisibility() {
  const hasSelection = selectedFilesInActiveScope().length > 0;
  optionsStep.hidden = !hasSelection;
  actionStep.hidden = !hasSelection;
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
}

[formatChecks, tokenIdInput, tokenSecretInput, gpuSelect, modelSelect].forEach((el) => {
  el.addEventListener("input", () => { updatePreviewEnabled(); savePreferences(); });
  el.addEventListener("change", () => { updatePreviewEnabled(); savePreferences(); });
});
executionRadios.forEach((r) =>
  r.addEventListener("change", () => {
    modalFields.hidden = currentExecutionMode() !== "modal";
    updatePreviewEnabled();
    savePreferences();
  })
);
cleanupCheck.addEventListener("change", savePreferences);

// ---------- Preferences (localStorage placeholder for Phase 1; Phase 2
// swaps this for GET/POST /api/preferences against user_prefs.json —
// credentials are NEVER persisted, only model/formats/execution/gpu) ----------

const PREFS_KEY = "localscribe_prefs_v1";

function savePreferences() {
  const prefs = {
    model: modelSelect.value,
    formats: [...formatChecks.querySelectorAll("input[type=checkbox]:checked")].map((c) => c.value),
    execution: currentExecutionMode(),
    gpu: gpuSelect.value,
    cleanup: cleanupCheck.checked,
  };
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (e) { /* ignore */ }
}

function loadPreferences() {
  let prefs;
  try { prefs = JSON.parse(localStorage.getItem(PREFS_KEY) || "null"); } catch (e) { prefs = null; }
  if (!prefs) return;
  if (prefs.model) modelSelect.value = prefs.model;
  if (prefs.formats) {
    formatChecks.querySelectorAll("input[type=checkbox]").forEach((c) => {
      c.checked = prefs.formats.includes(c.value);
    });
  }
  if (prefs.execution) {
    executionRadios.forEach((r) => { r.checked = r.value === prefs.execution; });
    modalFields.hidden = prefs.execution !== "modal";
  }
  if (prefs.gpu) gpuSelect.value = prefs.gpu;
  if (typeof prefs.cleanup === "boolean") cleanupCheck.checked = prefs.cleanup;
}

// ---------- Preview / Begin / Cancel ----------

const previewPanel = document.getElementById("preview-panel");
const previewTableBody = document.getElementById("preview-table-body");
const beginBtn = document.getElementById("begin-btn");
const runStep = document.getElementById("run-step");
const progressBar = document.getElementById("progress-bar");
const cancelBtn = document.getElementById("cancel-btn");
const logBox = document.getElementById("log-box");
const diagnosticsStep = document.getElementById("diagnostics-step");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsTableBody = document.getElementById("diagnostics-table-body");

function computeEstimate() {
  const files = selectedFilesInActiveScope();
  const model = modelSelect.value;
  const mode = currentExecutionMode();
  const rtfTable = mode === "modal" ? RTF_MODAL_GPU : RTF_LOCAL_CPU;
  const rtf = rtfTable[model];

  let extractionSec = 0;
  let transcriptionSec = 0;
  for (const f of files) {
    if (f.type === "video") extractionSec += (f.duration_sec / 60) * 2; // ~2s/min source
    transcriptionSec += f.duration_sec * rtf;
  }
  const cleanupSec = cleanupCheck.checked ? files.filter((f) => f.type === "video").length * 1 : 0;

  let extractionCost = 0, transcriptionCost = 0, cleanupCost = 0;
  if (mode === "modal") {
    const gpu = GPU_OPTIONS.find((g) => g.id === gpuSelect.value);
    transcriptionCost = (transcriptionSec / 3600) * gpu.rate;
  }

  return {
    phases: [
      { name: "Audio Extraction", sec: extractionSec, cost: extractionCost },
      { name: "Transcription", sec: transcriptionSec, cost: transcriptionCost },
      ...(cleanupCheck.checked ? [{ name: "Cleanup", sec: cleanupSec, cost: cleanupCost }] : []),
    ],
    files,
  };
}

previewBtn.addEventListener("click", () => {
  // Phase 3/4 replace this with: await fetch('/api/preview', {...})
  const estimate = computeEstimate();
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
  // /api/jobs/{id}/stream (SSE) + /api/jobs/{id}/cancel.
  const estimate = computeEstimate();
  const files = estimate.files;
  runStep.hidden = false;
  diagnosticsStep.hidden = true;
  logBox.innerHTML = "";
  progressBar.value = 0;
  state.run.cancelling = false;
  cancelBtn.disabled = false;
  cancelBtn.textContent = "Cancel";

  const model = modelSelect.value;
  const mode = currentExecutionMode();
  let i = 0;
  let succeeded = 0, failed = 0;

  function processNext() {
    if (state.run.cancelling) {
      addLogLine("Cancelled — remaining files were not started.");
      finishRun(true);
      return;
    }
    if (i >= files.length) {
      finishRun(false);
      return;
    }
    const f = files[i];
    if (f.type === "video") addLogLine(`Extracting audio: ${f.path}`);
    addLogLine(`Transcribing (${mode}, ${model}): ${f.path}`);

    const willFail = i === files.length - 1 && files.length > 2; // demo: last file fails sometimes
    setTimeout(() => {
      if (willFail) {
        addLogLine(`Failed: ${f.path} — ffmpeg could not read this file`, true);
        failed++;
      } else {
        addLogLine(`Done: ${f.path}`);
        succeeded++;
      }
      i++;
      progressBar.value = Math.round((i / files.length) * 100);
      processNext();
    }, 250);
  }

  function finishRun(wasCancelled) {
    cancelBtn.disabled = true;
    showDiagnostics(estimate, succeeded, failed, files.length - succeeded - failed, wasCancelled);
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
