# PRD — LocalScribe_whisper_modal

## 1. Overview

A locally-hosted web app that scans a folder (recursively) for audio/video
files, lets the user pick which ones to transcribe with an open-source
Whisper model, and runs the transcription either on the user's own machine
(CPU) or on Modal.com GPUs. Output transcripts are written next to the
source files in the user's chosen formats.

**Problem:** transcribing a batch of local media files today means manually
running ffmpeg and a Whisper CLI per file, tracking costs/time by hand, and
having no single place to see progress or failures across a folder tree.

**Goal:** a single local page where the user points at a folder, picks
files and settings, previews estimated time/cost, and gets transcripts with
a clear progress view and an end-of-run diagnostics summary.

## 2. Users & scope

Single local user running the app on their own machine (no auth, no
multi-tenant concerns). Out of scope: cloud hosting of the app itself,
support for engines other than faster-whisper, translation (transcription
only), streaming/live audio.

## 3. User flow

1. Paste a folder path, click **Analyze**.
2. Review file counts/size/duration for **Current folder** vs **Including
   subfolders**, pick a scope, expand Video/Audio to select individual
   files (or select-all per type).
3. Choose Whisper model, output formats, execution mode (Local or
   Modal.com — with GPU type + Modal credentials if remote), and whether to
   delete intermediate audio files afterward.
4. Click **Preview** to see estimated time/cost per phase, then **Begin**.
5. Watch a progress bar and concise status log; **Cancel** if needed —
   interrupts the in-flight step immediately, then Cleanup still runs for
   any file with an extracted intermediate WAV.
6. Review the diagnostics summary (succeeded/failed counts, actual
   time/cost totals) once the run ends.

## 4. Functional requirements

### 4.1 Folder scan
- Recursive walk of the given folder; classify files as video/audio by
  extension.
- Report count, total size, total duration **both** for files directly in
  the folder and for the folder + all subfolders combined.
- Duration/size sourced via `ffprobe`/filesystem stat.

### 4.2 File selection UI
- One table, heading "Select the files to transcribe".
- Two folder-level scopes ("Current folder" / "Including subfolders") as
  **radio buttons** — always both enabled, so the user is never stuck
  unable to switch scope. Only one scope's file lists are selectable at a
  time; the other scope's Video/Audio rows (and their files, once
  expanded) are visibly greyed out and non-interactive.
- Under the active scope, Video and Audio each get an **accordion toggle**
  (left of a **tri-state checkbox** that selects/deselects all files of
  that type). Expanding lists individual files with their path relative to
  the scanned folder, duration, and size, each with its own checkbox.
- A live summary line below the table: "Selected: N files — ⟨size⟩,
  ⟨duration⟩".

### 4.3 Options
- Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`,
  `large-v3-turbo`.
- Output formats (multi-select): `txt`, `srt`, `vtt`, `json`, `tsv`.
- Execution: **Local** (CPU) or **Modal.com** (GPU).
  - Modal.com reveals a GPU type dropdown (T4/L4/A10G/A100/H100, each
    showing $/hr) with **L4 recommended and pre-selected by default**, and
    two credential fields — **Modal Token ID** and **Modal Token Secret**
    (password inputs).
- "Delete intermediate audio files when done" checkbox.
- Model/formats/execution/GPU/credential selections persist across
  sessions in `user_prefs.json`, at the user's explicit request (revised
  from the original "credentials never persist" design) — the Token
  ID/Secret are stored there in **plaintext**; the file stays gitignored
  and the credentials are never transmitted anywhere but Modal itself.

### 4.4 Preview → Begin → Cancel
- **Preview** is enabled only once all required inputs are present; it
  renders an inline 3-phase estimate table (Audio Extraction /
  Transcription / Cleanup) with time + cost per phase and a total, then
  reveals **Begin**.
- **Begin** starts the job: overall progress bar, capped-height
  auto-scrolling status log, and a **Cancel** button.
- **Cancel** interrupts the in-flight step immediately rather than
  letting it run to completion: it kills an in-flight ffmpeg extraction,
  stops a local Whisper transcription between segments (the earliest
  point a lazily-computed segment generator can be checked), and cancels
  an in-flight Modal RPC (terminating the remote container, which also
  stops billing for it). Cleanup then still runs unconditionally
  afterward for every file that has an extracted intermediate WAV on
  disk, whether or not that file's later phase completed or failed — so
  "Cancelling…" can take a few seconds to settle while Cleanup finishes,
  even though the step that was actually running stopped immediately.

### 4.5 Per-file pipeline
1. Extract audio via ffmpeg to a 16kHz mono WAV (same basename/folder as
   source), skipped for files that are already audio.
2. Transcribe (local faster-whisper, or an ephemeral Modal.com GPU
   function) → segments.
3. Write one output file per selected format (same basename/folder).
4. If cleanup is checked, delete the WAV generated in step 1 only (never a
   pre-existing original audio file).

Each stage emits a progress event; failures are caught per-file and shown
as a short plain-language message, with full detail written to the log
file.

### 4.6 Diagnostics
- On completion (finished or cancelled): a one-line summary ("N files
  processed — X succeeded, Y failed") plus a small Total / Per file / Per
  minute-of-media table for Processing time and Cost (actual measured
  values, not estimates). No per-file breakdown in the UI — full detail
  lives in the log files.

### 4.7 Logging
- JSON-line process logs written locally under `logs/` (gitignored,
  private — never committed).

## 5. Non-functional requirements
- No build step for the frontend (plain HTML/CSS/JS) — keeps the app
  simple to run (`pip install` + `uvicorn`) with nothing else to install.
- No secrets committed to the repo — `user_prefs.json` is gitignored.
  Modal credentials *are* written there in plaintext (see 4.3), at the
  user's request, for convenience; never transmitted anywhere but Modal.
- Runs fully locally; the only outbound network call is to Modal.com when
  that execution mode is explicitly chosen.

## 6. Non-goals
- Multi-user access control, cloud deployment of the app itself,
  translation, live/streaming transcription, engines other than
  faster-whisper.

## 7. Open questions / risks
- `ctranslate2`/`faster-whisper` wheel availability on very recent Python
  versions (mitigated by pinning the project venv to Python 3.12 via
  `uv`).
- Modal.com GPU pricing changes over time — the in-app rate table is
  labeled "approximate, verify at modal.com/pricing".

## 8. Delivery plan

Six phases, each ending in a local git commit (after a secret/PII sweep),
tracked in Linear as project **LocalScribe_whisper_modal**
(1 phase = 1 milestone, since Linear Cycles can't be created via the
available API — milestones are the closest creatable "sprint" analog):

1. Frontend Prototype (mock data) — PDG-5..PDG-10
2. Backend Foundation & Folder Scan — PDG-11..PDG-15
3. Local Transcription Pipeline — PDG-16..PDG-22
4. Modal.com GPU Execution — PDG-23..PDG-26
5. Preferences, Diagnostics & Logging Polish — PDG-27..PDG-30
6. Packaging, Docs & QA — PDG-31..PDG-34

Full architecture, API surface, and rationale for each design decision
live in the engineering plan (see project history / Linear project
description); this PRD tracks the product-level requirements those
decisions satisfy.
