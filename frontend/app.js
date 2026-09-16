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
  const s = totalSec % 60;
  const totalMinutes = Math.floor(totalSec / 60);
  const m = totalMinutes % 60;
  const totalHours = Math.floor(totalMinutes / 60);
  const h = totalHours % 24;
  const totalDays = Math.floor(totalHours / 24);
  const d = totalDays % 365;
  const y = Math.floor(totalDays / 365);

  // Escalates unit as the value grows so the string always stays short
  // (two components max) and never needs to wrap in a table cell.
  if (y > 0) return `${y}y ${d}d`;
  if (totalDays > 0) return `${totalDays}d ${h}h`;
  if (totalHours > 0) return `${h}h ${m}m`;
  if (totalMinutes > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatCount(n) {
  const units = [{ v: 1e9, s: "B" }, { v: 1e6, s: "M" }, { v: 1e3, s: "K" }];
  for (const u of units) {
    if (n >= u.v) return `${(n / u.v).toFixed(1).replace(/\.0$/, "")}${u.s}`;
  }
  return `${n}`;
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
      // -1 is the "select all" sentinel for a group with 0 files (see
      // renderScanTable) -- it isn't a real file index, so it must never
      // turn into a phantom selected file here.
      if (idx < 0) continue;
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

    // Level-1 (scope) rows are never greyed -- the radio is always fully
    // interactive by design (see file header comment), and so is its
    // label now. Only the type/file rows nested under the inactive scope
    // stay greyed until its radio is selected.
    const scopeRow = document.createElement("tr");
    scopeRow.className = "scope-row";
    scopeRow.innerHTML = `
      <td class="col-label">
        <input type="radio" name="scope" value="${scopeKey}" ${active ? "checked" : ""} />
        <span class="row-label-text">${label}</span>
      </td>
      <td>${formatCount(stats.count)}</td>
      <td>${formatBytes(stats.size)}</td>
      <td>${formatDuration(stats.duration)}</td>
    `;
    const scopeRadio = scopeRow.querySelector('input[type="radio"]');
    scopeRadio.addEventListener("change", () => {
      state.scope = scopeKey;
      renderScanTable();
      renderOptionsVisibility();
    });
    scopeRow.querySelector(".row-label-text").addEventListener("click", () => {
      if (scopeRadio.checked) return;
      scopeRadio.checked = true;
      scopeRadio.dispatchEvent(new Event("change"));
    });
    scanTableBody.appendChild(scopeRow);

    for (const type of ["video", "audio"]) {
      const typeStats = scopeStats(scopeKey, type);
      const selectedSet = state.selected[scopeKey][type];
      // A group with 0 files has nothing for `selectedSet` to ever hold
      // (there's no index to add), so comparing sizes would either always
      // be vacuously true or always force this false -- neither lets the
      // checkbox actually respond to a click. -1 is used as a sentinel
      // "checked" marker specifically for the empty-group case (safe: it
      // can only ever be added when this group truly has 0 files, and a
      // fresh scan always clears selection before file counts can change).
      const allSelected = typeStats.count > 0 ? selectedSet.size === typeStats.count : selectedSet.has(-1);
      const someSelected = typeStats.count > 0 && selectedSet.size > 0 && !allSelected;
      const expanded = state.expanded[scopeKey][type];

      const typeRow = document.createElement("tr");
      typeRow.className = "type-row" + (active ? "" : " greyed");
      typeRow.innerHTML = `
        <td class="col-label">
          <span class="accordion-toggle">${expanded ? "▾" : "▸"}</span>
          <input type="checkbox" ${allSelected ? "checked" : ""} ${!active ? "disabled" : ""} />
          <span class="row-label-text">${type === "video" ? "Video" : "Audio"}</span>
        </td>
        <td>${formatCount(typeStats.count)}</td>
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
        if (typeStats.count === 0) {
          // Nothing to actually select -- just let the checkbox reflect
          // the click instead of silently doing nothing.
          if (checkbox.checked) selectedSet.add(-1); else selectedSet.delete(-1);
        } else {
          const files = state.scanData.scopes[scopeKey][type].files;
          if (checkbox.checked) {
            files.forEach((_, i) => selectedSet.add(i));
          } else {
            selectedSet.clear();
          }
        }
        renderScanTable();
        renderOptionsVisibility();
      });
      // Lets the "Video"/"Audio" text itself toggle select-all too -- the
      // checkbox alone is a small target to have to aim for with a mouse.
      typeRow.querySelector(".row-label-text").addEventListener("click", () => {
        if (!active) return;
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event("change"));
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
              <span class="row-label-text">${file.path}</span>
            </td>
            <td></td>
            <td>${formatBytes(file.size_bytes)}</td>
            <td>${formatDuration(file.duration_sec)}</td>
          `;
          const fileCheckbox = fileRow.querySelector('input[type="checkbox"]');
          fileCheckbox.addEventListener("change", (e) => {
            if (!active) return;
            if (e.target.checked) selectedSet.add(idx); else selectedSet.delete(idx);
            renderScanTable();
            renderOptionsVisibility();
          });
          // Lets the filename itself toggle its checkbox -- easier to hit
          // with a mouse than the checkbox alone, especially for long paths.
          fileRow.querySelector(".row-label-text").addEventListener("click", () => {
            if (!active) return;
            fileCheckbox.checked = !fileCheckbox.checked;
            fileCheckbox.dispatchEvent(new Event("change"));
          });
          scanTableBody.appendChild(fileRow);
        });
      }
    }
  }

  const sel = selectedFilesInActiveScope();
  // `|| 0` guards against a stale selection index left over from a
  // previous folder's (now differently-sized) file list turning into
  // `undefined` fields -- summing those would otherwise show "NaN"
  // instead of a real number.
  const size = sel.reduce((s, f) => s + (f.size_bytes || 0), 0);
  const duration = sel.reduce((s, f) => s + (f.duration_sec || 0), 0);
  selectionSummary.textContent = sel.length
    ? `Selected: ${sel.length} files — ${formatBytes(size)}, ${formatDuration(duration)}`
    : "Selected: 0 files";
}

// ---------- Folder step ----------

const folderStep = document.getElementById("folder-step");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const folderPathInput = document.getElementById("folder-path");
const folderStatus = document.getElementById("folder-status");
const browseFolderBtn = document.getElementById("browse-folder-btn");
const folderHistoryDropdown = document.getElementById("folder-history-dropdown");
const selectionStep = document.getElementById("selection-step");
const optionsStep = document.getElementById("options-step");
const actionStep = document.getElementById("action-step");

// Everything Reset clears, minus the folder path field and its status
// message -- shared with a failed Analyze, which needs to clear any
// stale scan/selection/preview/run state below it (an old successful
// scan's results shouldn't linger under a now-invalid path) while
// leaving the path itself and the error message visible.
function clearDownstreamState() {
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
}

// ---------- Folder history (persisted in user_prefs.json's
// folder_history field -- only folders that have been successfully
// analyzed at least once get added) ----------

const MAX_FOLDER_HISTORY = 20;
let folderHistory = [];

// Loaded immediately on page load (not gated on a scan, unlike the rest
// of loadPreferences() below) so the history dropdown works the moment
// the page opens, before the user has analyzed anything this session.
(async () => {
  try {
    const res = await fetch("/api/preferences");
    const prefs = res.ok ? await res.json() : null;
    if (prefs && Array.isArray(prefs.folder_history)) folderHistory = prefs.folder_history;
  } catch (e) { /* best effort */ }
})();

function addToFolderHistory(path) {
  folderHistory = [path, ...folderHistory.filter((p) => p !== path)].slice(0, MAX_FOLDER_HISTORY);
  fetch("/api/preferences", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder_history: folderHistory }),
  }).catch(() => { /* best-effort -- a failed save just means history won't persist */ });
}

function renderFolderHistoryDropdown(matches) {
  folderHistoryDropdown.innerHTML = "";
  if (matches.length === 0) {
    folderHistoryDropdown.hidden = true;
    return;
  }
  matches.forEach((path) => {
    const li = document.createElement("li");
    li.textContent = path;
    // mousedown (not click) fires before the input's blur handler would
    // otherwise hide the dropdown first and swallow the selection.
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      folderPathInput.value = path;
      folderHistoryDropdown.hidden = true;
      onFolderPathChanged();
    });
    folderHistoryDropdown.appendChild(li);
  });
  folderHistoryDropdown.hidden = false;
}

function showFolderHistoryDropdown() {
  const typed = folderPathInput.value.trim().toLowerCase();
  const matches = (typed ? folderHistory.filter((p) => p.toLowerCase().includes(typed)) : folderHistory).slice(0, 5);
  renderFolderHistoryDropdown(matches);
}

function clearFolderStatus() {
  folderStatus.hidden = true;
  folderStatus.textContent = "";
}

function updateAnalyzeButtonState() {
  analyzeBtn.disabled = !folderPathInput.value.trim();
}

// Shared by every way the field's value can change -- typing, picking a
// history entry, or the native browse dialog -- so a stale validation
// error never lingers after the path it was about is gone, and Analyze's
// enabled state always matches whether there's something to analyze.
function onFolderPathChanged() {
  clearFolderStatus();
  updateAnalyzeButtonState();
}

folderPathInput.addEventListener("focus", showFolderHistoryDropdown);
folderPathInput.addEventListener("input", () => {
  onFolderPathChanged();
  showFolderHistoryDropdown();
});
folderPathInput.addEventListener("blur", () => {
  setTimeout(() => { folderHistoryDropdown.hidden = true; }, 150);
});
updateAnalyzeButtonState(); // starts disabled -- the field is empty on page load

browseFolderBtn.addEventListener("click", async () => {
  browseFolderBtn.disabled = true;
  try {
    const res = await fetch("/api/browse-folder", { method: "POST" });
    if (res.ok) {
      const body = await res.json();
      if (body.path) {
        folderPathInput.value = body.path;
        folderHistoryDropdown.hidden = true;
        onFolderPathChanged();
      }
    }
  } catch (e) { /* best effort -- native dialog may not be available */ }
  finally {
    browseFolderBtn.disabled = false;
  }
});

analyzeBtn.addEventListener("click", async () => {
  const path = folderPathInput.value.trim();
  if (!path) return; // the button is disabled in this case, but guard anyway
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
    // A fresh scan's file list is a different (or resized) set of files --
    // any previously selected indices could point at something entirely
    // different now (or nothing at all), so selection never carries over
    // from one folder to the next.
    state.selected = {
      current: { video: new Set(), audio: new Set() },
      all: { video: new Set(), audio: new Set() },
    };
    folderStatus.hidden = true;
    selectionStep.hidden = false;
    renderScanTable();
    addToFolderHistory(path);
    await loadPreferences();
    renderOptionsVisibility();
  } catch (err) {
    folderStatus.textContent = err.message || "Could not scan that folder.";
    folderStatus.className = "status error";
    folderStatus.hidden = false;
    // An invalid folder shouldn't leave a previous successful scan's
    // results (selection, options, preview, run) visible below it.
    clearDownstreamState();
  } finally {
    updateAnalyzeButtonState();
  }
});

resetBtn.addEventListener("click", () => {
  folderPathInput.value = "";
  folderStatus.hidden = true;
  updateAnalyzeButtonState();
  clearDownstreamState();
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

// Real hardware summary for the Local execution path, from the machine
// actually running the server (backend/system_info.py, stdlib-only).
localResourceInfo.textContent = "Detecting…";
(async () => {
  try {
    const res = await fetch("/api/local-resources");
    if (!res.ok) throw new Error();
    const info = await res.json();
    localResourceInfo.textContent = `CPU: ${info.cpu} · RAM: ${info.ram} · ${info.note}`;
  } catch (e) {
    localResourceInfo.textContent = "Could not detect local hardware.";
  }
})();

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

// ---------- Locking the folder/selection/options steps once a job is
// running -- changing any of them mid-job would invalidate the very
// preview the running job was started from (and hide Begin/the run
// panel), so they're frozen (greyed out, all interaction blocked) for
// the duration of the run. They stay visible so the user can still refer
// to what the running job was actually configured with. Purely cosmetic
// controls that can't change the job's ingredients -- the accordion
// toggle, the password show/hide eye icon -- are explicitly exempted via
// the .job-locked CSS rule, not disabled here. ----------

function setPreRunSectionsLocked(locked) {
  [folderStep, selectionStep, optionsStep].forEach((el) => el.classList.toggle("job-locked", locked));
  const controls = [folderStep, selectionStep, optionsStep]
    .flatMap((el) => [...el.querySelectorAll("input, select, button")])
    .filter((el) => !el.classList.contains("eye-toggle"));
  controls.forEach((el) => { el.disabled = locked; });
  if (locked) {
    previewBtn.disabled = true;
  } else {
    // Re-derive each control's correct enabled/disabled state (e.g. a
    // checkbox in the inactive scope, or Analyze with an empty field)
    // rather than blindly re-enabling everything that was just unlocked.
    // Deliberately NOT updatePreviewEnabled() -- that also invalidates
    // (hides) the preview panel/Begin row, which would make the just-
    // finished job's Steps table disappear the moment it ends; nothing
    // about the selection/options actually changed here.
    renderScanTable();
    updateAnalyzeButtonState();
    previewBtn.disabled = !computePreviewButtonEnabled();
  }
}

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

function computePreviewButtonEnabled() {
  const hasSelection = selectedFilesInActiveScope().length > 0;
  const hasFormat = [...formatChecks.querySelectorAll("input[type=checkbox]")].some((c) => c.checked);
  const mode = currentExecutionMode();
  let modalOk = true;
  if (mode === "modal") {
    modalOk = !!gpuSelect.value && tokenIdInput.value.trim() && tokenSecretInput.value.trim();
  }
  return hasSelection && hasFormat && modalOk;
}

function updatePreviewEnabled() {
  previewBtn.disabled = !computePreviewButtonEnabled();
  // Any change to selection/options invalidates a preview already shown --
  // Begin only ever appears right after a fresh Preview run. Only called
  // from places where the selection/options genuinely just changed --
  // NOT from unlocking after a job ends (see setPreRunSectionsLocked),
  // where nothing actually changed and the just-finished job's Steps
  // table should stay visible, not disappear.
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
  if (Array.isArray(prefs.folder_history)) folderHistory = prefs.folder_history;
  updatePreviewEnabled();
}

// ---------- Preview / Begin / Cancel ----------

const previewPanel = document.getElementById("preview-panel");
const previewTableBody = document.getElementById("preview-table-body");
const beginRow = document.getElementById("begin-row");
const beginBtn = document.getElementById("begin-btn");
const runStep = document.getElementById("run-step");
const progressBar = document.getElementById("progress-bar");
const progressPercentLabel = document.getElementById("progress-percent-label");
const cancelBtn = document.getElementById("cancel-btn");
const logBox = document.getElementById("log-box");
const diagnosticsStep = document.getElementById("diagnostics-step");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsTableBody = document.getElementById("diagnostics-table-body");
const previewSelectionSummary = document.getElementById("preview-selection-summary");
const fatalErrorOverlay = document.getElementById("fatal-error-overlay");
const fatalErrorMessage = document.getElementById("fatal-error-message");
const fatalErrorOkBtn = document.getElementById("fatal-error-ok-btn");
const fatalErrorCancelBtn = document.getElementById("fatal-error-cancel-btn");

function showFatalErrorDialog(text) {
  fatalErrorMessage.textContent = text;
  fatalErrorOverlay.hidden = false;
}

function hideFatalErrorDialog() {
  fatalErrorOverlay.hidden = true;
}

// Both buttons just dismiss the dialog -- by the time this event arrives
// the backend has already decided the job can't continue (e.g. the
// batch's real upload size exceeds Modal.com's limit) and is already
// winding down on its own, so there's no different action for OK vs.
// Cancel to actually take beyond acknowledging the message.
fatalErrorOkBtn.addEventListener("click", hideFatalErrorDialog);
fatalErrorCancelBtn.addEventListener("click", hideFatalErrorDialog);

// Shows which job is running, above the progress bar -- so it can be
// matched against its server-side log file (named "<timestamp>_<job id
// prefix>.jsonl" under logs/) if it fails and needs investigating,
// without having to dig through server logs to find the id themselves.
const runJobIdDisplay = document.getElementById("run-job-id-display");
const runJobIdText = document.getElementById("run-job-id-text");
const runCopyJobIdBtn = document.getElementById("run-copy-job-id-btn");

function hideJobIdDisplay() {
  runJobIdDisplay.hidden = true;
  runJobIdText.textContent = "";
}

function showJobIdDisplay(jobId) {
  runJobIdText.textContent = jobId;
  runJobIdDisplay.hidden = false;
}

let copyJobIdResetTimer = null;

runCopyJobIdBtn.addEventListener("click", async () => {
  const id = runJobIdText.textContent;
  if (!id) return;
  try {
    await navigator.clipboard.writeText(id);
  } catch (e) {
    return; // clipboard API unavailable/denied -- nothing more we can do
  }
  runCopyJobIdBtn.classList.add("copied");
  runCopyJobIdBtn.title = "Copied!";
  clearTimeout(copyJobIdResetTimer);
  copyJobIdResetTimer = setTimeout(() => {
    runCopyJobIdBtn.classList.remove("copied");
    runCopyJobIdBtn.title = "Copy job ID";
  }, 1500);
});

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
  progressPercentLabel.textContent = "0%";
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
    if (i === 0) bump(execution === "modal" ? "Modal.com Setup & Model Install" : "Whisper Model Setup");
    if (execution === "modal") bump("Upload audio files to Modal.com");
    bump(execution === "modal" ? "Transcription & Download" : "Transcription");
    if (execution === "modal") bump("Cleanup - Modal.com Audio Uploads");
    if (i === 0) bump(execution === "modal" ? "Cleanup - Modal.com Teardown" : "Cleanup - Release Whisper Model");
    if (cleanup && f.type === "video") bump("Cleanup - Intermediate Files");
  });
  return counts;
}

function setPhaseIcon(phaseName, state) {
  const el = previewTableBody.querySelector(`.phase-icon[data-phase="${phaseName}"]`);
  if (!el) return;
  el.className = "phase-icon" + (state ? ` phase-icon--${state}` : "");
  el.textContent = state === "done" ? "✓" : state === "failed" ? "✗" : state === "warning" ? "!" : "";
}

// Shows live progress while a phase runs, clears on a clean finish, but
// is deliberately left in place if the phase was interrupted (a file
// failed/skipped/cancelled) so the row keeps showing exactly how far it
// got.
function setStepProgressSuffix(phaseName, text) {
  const el = previewTableBody.querySelector(`.step-progress-suffix[data-phase="${phaseName}"]`);
  if (el) el.textContent = text;
}

// Batch-wide "(NN%)" text per phase, set from step_progress events --
// kept separate from the file-count suffix below so the two can be
// combined/recombined independently as either one changes.
let stepPercentText = {};

// These phases process every file individually within the phase (unlike
// Audio Extraction, whose row shows a total size once done -- see the
// audio_extraction_summary handler below) -- while the phase is running,
// its row also shows "done / total" files, e.g. "5 / 10".
const FILE_COUNT_SUFFIX_PHASES = new Set([
  "Upload audio files to Modal.com",
  "Transcription",
  "Transcription & Download",
  "Cleanup - Modal.com Audio Uploads",
]);

function fileCountSuffixText(phaseName) {
  const st = phaseStatus[phaseName];
  if (!st || !FILE_COUNT_SUFFIX_PHASES.has(phaseName)) return "";
  const accountedFor = st.doneCount + st.failedCount + st.skippedCount + st.cancelledCount;
  return ` ${accountedFor} / ${st.expectedCount}`;
}

// Recombines and (re)renders a phase row's suffix from its latest known
// percent text and file-count text -- called any time either input
// changes (a step_progress event, or a file finishing the phase).
function updateStepSuffix(phaseName) {
  setStepProgressSuffix(phaseName, (stepPercentText[phaseName] || "") + fileCountSuffixText(phaseName));
}

// True once every file has cleanly finished this phase -- no failures or
// cancellations at all. Mirrors paintPhaseFromCounts' criteria for a
// green "done" icon; used to decide whether a phase's suffix should be
// cleared (clean finish) or left in place (interrupted, as a record of
// how far it got).
function isPhaseCleanlyDone(phaseName) {
  const st = phaseStatus[phaseName];
  if (!st) return false;
  const accountedFor = st.doneCount + st.failedCount + st.skippedCount + st.cancelledCount;
  return accountedFor >= st.expectedCount && st.failedCount === 0 && st.cancelledCount === 0 && st.doneCount > 0;
}

// Paints a phase's icon from its counts so far -- green only if nothing
// failed or was cancelled, red if this phase itself was ever interrupted
// or every file that reached it failed, yellow for a genuine mix of
// success and failure. cancelledCount always wins over doneCount: a phase
// where one file succeeded and another was cancelled mid-flight is an
// interrupted phase, not a successful one, regardless of the mix.
// skippedCount alone (every file cascade-skipped because an *earlier*
// phase already failed them -- this phase itself was never attempted)
// doesn't count as either success or failure.
function paintPhaseFromCounts(phaseName) {
  const st = phaseStatus[phaseName];
  if (!st) return;
  if (st.cancelledCount > 0) {
    setPhaseIcon(phaseName, "failed");
  } else if (st.failedCount > 0 && st.doneCount > 0) {
    setPhaseIcon(phaseName, "warning");
  } else if (st.failedCount > 0) {
    setPhaseIcon(phaseName, "failed");
  } else if (st.doneCount > 0) {
    setPhaseIcon(phaseName, "done");
  }
}

// Once every (file, phase) instance for this phase has been accounted for
// (succeeded, failed, cascade-skipped, or cancelled), paint its final icon.
function finalizePhaseIfComplete(phaseName) {
  const st = phaseStatus[phaseName];
  if (!st) return;
  const accountedFor = st.doneCount + st.failedCount + st.skippedCount + st.cancelledCount;
  if (accountedFor < st.expectedCount) return;
  paintPhaseFromCounts(phaseName);
}

// The job's "done" event (completed OR cancelled) can arrive while a phase
// row is still short of its expectedCount -- a cancel stops the pipeline
// before every file reaches every phase, so no further step_* events for
// those files are ever coming, and the row is left showing whatever it
// last was (almost always still spinning) forever without this. A phase
// with at least one recorded event that's still short of its count was
// genuinely interrupted partway through -- painted from its counts so
// far. A phase with *zero* events was never even reached (e.g. Download
// when Transcription itself got cancelled first) -- left untouched
// (still its original pending/blank icon) rather than marked failed,
// since nothing about it actually broke.
function forceFinalizeIncompletePhases() {
  for (const name of Object.keys(phaseStatus)) {
    const st = phaseStatus[name];
    const accountedFor = st.doneCount + st.failedCount + st.skippedCount + st.cancelledCount;
    if (accountedFor === 0) continue;
    if (accountedFor < st.expectedCount) {
      paintPhaseFromCounts(name);
    }
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
  const size = selected.reduce((s, f) => s + (f.size_bytes || 0), 0);
  const duration = selected.reduce((s, f) => s + (f.duration_sec || 0), 0);
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
      tr.innerHTML = `<td><span class="phase-icon" data-phase="${p.name}"></span>${p.name}<span class="step-progress-suffix" data-phase="${p.name}"></span></td><td>${formatDuration(p.sec)}</td><td>${formatCost(p.cost)}</td>`;
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
  return line;
}

// Audio Extraction and Transcription both report real progress (ffmpeg's
// -progress stream and faster-whisper's per-segment callback,
// respectively) -- whichever one is currently running has its log line
// kept updated in place with a live "(NN%)" suffix as step_progress
// events arrive, rather than only showing anything once the whole
// (sometimes long) step completes. Only one step is ever actually
// running at a time (the pipeline is sequential), so tracking "the"
// active one by name is enough; cleared whenever that step ends so a
// late/stray progress event can't rewrite an already-finished line.
let activeProgressStep = null;
let activeProgressLine = null;
let activeProgressBaseText = "";

beginBtn.addEventListener("click", async () => {
  setPreRunSectionsLocked(true);
  runStep.hidden = false;
  diagnosticsStep.hidden = true;
  logBox.innerHTML = "";
  progressBar.value = 0;
  progressPercentLabel.textContent = "0%";
  cancelBtn.disabled = false;
  cancelBtn.textContent = "Cancel";
  beginBtn.disabled = true;
  hideFatalErrorDialog();
  hideJobIdDisplay();

  const req = buildTranscribeRequest();
  const expected = computeExpectedPhaseCounts(req.files, req.execution, req.cleanup);
  phaseStatus = {};
  stepPercentText = {};
  for (const [name, expectedCount] of Object.entries(expected)) {
    phaseStatus[name] = { doneCount: 0, failedCount: 0, skippedCount: 0, cancelledCount: 0, expectedCount };
    setPhaseIcon(name, null); // clear any icon left from a previous run
  }
  // Clears every phase row's suffix, not just phases this run actually
  // expects -- e.g. a previous video-heavy run's Audio Extraction total
  // size, or a previous Modal run's Upload "done / total" count, must not
  // linger into a run (audio-only, or Local) where that row still exists
  // in the table but nothing will ever update it again.
  previewTableBody.querySelectorAll(".step-progress-suffix").forEach((el) => { el.textContent = ""; });
  activeProgressStep = null;
  activeProgressLine = null;
  activeProgressBaseText = "";

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
    showJobIdDisplay(job_id);

    const es = new EventSource(`/api/jobs/${job_id}/stream`);
    activeEventSource = es;
    es.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      if (event.event === "log") {
        const line = addLogLine(event.text, !!event.fail);
        activeProgressStep = event.step;
        activeProgressLine = line;
        activeProgressBaseText = event.text;
      } else if (event.event === "fatal_error") {
        // A job-ending error distinct from any single file/step failing
        // (e.g. the batch's real upload size exceeds Modal.com's limit,
        // checked right after Audio Extraction produces real file sizes)
        // -- already shown in the log above via its own "log" event;
        // this additionally surfaces it as a dialog over the page, since
        // it means the whole job is being abandoned, not just one file.
        showFatalErrorDialog(event.text);
      } else if (event.event === "audio_extraction_summary") {
        // Fires once, right after the whole Audio Extraction phase has
        // finished (or immediately, for an audio-only batch that never
        // extracted anything) -- replaces that row's now-cleared "(NN%)"
        // with the real total size of what's actually about to be
        // uploaded/transcribed.
        stepPercentText["Audio Extraction"] = ` (${formatBytes(event.total_bytes)})`;
        updateStepSuffix("Audio Extraction");
      } else if (event.event === "step_progress") {
        // step_progress arrives for Audio Extraction, Upload, and
        // Transcription (see pipeline.py). Two different numbers for two
        // different audiences: the Steps table has one row per phase for
        // the *whole batch*, so it shows batch_percent (a continuous
        // 0->100% sweep across every file in that phase); the scrolling
        // log is file-by-file, so its line shows percent (this file only,
        // correctly resetting to 0% per file) -- and only updates if it's
        // still the most recently logged step (guards against a stray
        // late event after the pipeline has already moved on).
        stepPercentText[event.step] = ` (${event.batch_percent}%)`;
        updateStepSuffix(event.step);
        if (event.step === activeProgressStep && activeProgressLine) {
          activeProgressLine.textContent = `${activeProgressBaseText} (${event.percent}%)`;
        }
      } else if (event.event === "step_start") {
        const st = phaseStatus[event.step];
        if (st && st.doneCount + st.failedCount + st.skippedCount + st.cancelledCount < st.expectedCount) {
          setPhaseIcon(event.step, "spinner");
        }
        // Shows "0 / N" the instant the phase starts, for Download
        // Transcript to Local in particular -- it never gets a
        // step_progress event of its own (it has nothing to report
        // progress on), so without this its file-count suffix would only
        // ever appear starting from its *second* file.
        updateStepSuffix(event.step);
      } else if (event.event === "step_done") {
        const st = phaseStatus[event.step];
        if (st) {
          st.doneCount++;
          finalizePhaseIfComplete(event.step);
        }
        // A clean finish across every file in the phase doesn't need the
        // suffix anymore -- only a phase interrupted by a failure/
        // cancellation (below) keeps it, as a record of how far it got.
        // Still short of every file, though: update in place (advances
        // the "done / total" count) rather than clearing, so the row
        // doesn't flicker blank between one file finishing and the next
        // one's progress arriving.
        if (isPhaseCleanlyDone(event.step)) {
          stepPercentText[event.step] = "";
          setStepProgressSuffix(event.step, "");
        } else {
          updateStepSuffix(event.step);
        }
        if (event.step === activeProgressStep) activeProgressLine = null;
      } else if (event.event === "step_failed") {
        const st = phaseStatus[event.step];
        if (st) {
          st.failedCount++;
          finalizePhaseIfComplete(event.step);
        }
        updateStepSuffix(event.step); // advances "done / total"; retained, never cleared
        if (event.step === activeProgressStep) activeProgressLine = null;
      } else if (event.event === "step_cancelled") {
        // This phase was itself interrupted mid-flight for this file --
        // distinct from step_skipped below (a file cascade-skipped
        // because an *earlier* phase already failed it, which says
        // nothing bad about THIS phase). paintPhaseFromCounts (via
        // finalizePhaseIfComplete) always shows a phase with any
        // cancellation as interrupted, never as a clean success.
        const st = phaseStatus[event.step];
        if (st) {
          st.cancelledCount++;
          finalizePhaseIfComplete(event.step);
        }
        updateStepSuffix(event.step);
        if (event.step === activeProgressStep) activeProgressLine = null;
      } else if (event.event === "step_skipped") {
        const st = phaseStatus[event.step];
        if (st) {
          st.skippedCount++;
          finalizePhaseIfComplete(event.step);
        }
        updateStepSuffix(event.step);
        if (event.step === activeProgressStep) activeProgressLine = null;
      } else if (event.event === "progress") {
        progressBar.value = event.percent;
        progressPercentLabel.textContent = `${event.percent}%`;
      } else if (event.event === "done") {
        es.close();
        activeEventSource = null;
        activeJobId = null;
        cancelBtn.textContent = "Cancel";
        cancelBtn.disabled = true;
        beginBtn.disabled = false;
        setPreRunSectionsLocked(false);
        forceFinalizeIncompletePhases();
        showDiagnostics(event);
      }
    };
    es.onerror = () => {
      es.close();
      activeEventSource = null;
      activeJobId = null;
      cancelBtn.disabled = true;
      beginBtn.disabled = false;
      setPreRunSectionsLocked(false);
      forceFinalizeIncompletePhases();
      addLogLine("Lost connection to the server.", true);
    };
  } catch (err) {
    addLogLine(err.message || "Could not start the job.", true);
    cancelBtn.disabled = true;
    beginBtn.disabled = false;
    setPreRunSectionsLocked(false);
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
