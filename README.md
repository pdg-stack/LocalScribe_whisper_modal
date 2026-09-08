# LocalScribe_whisper_modal

A locally-hosted web app that scans a folder for audio/video files and
transcribes the ones you pick, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
either on your own machine or on [Modal.com](https://modal.com) GPUs.

> **Status:** under active development (Phase 1 of 6 — see the plan/Linear
> project for the full roadmap). Setup instructions below will fill in as
> the backend lands.

## Setup

_Coming in Phase 2._ Setup will require:

1. An isolated virtual environment (never install into system Python) —
   `uv venv --python 3.12 .venv` (Python 3.12 is pinned since `ctranslate2`/
   `faster-whisper` wheel availability for very recent Python versions is
   not guaranteed).
2. `ffmpeg`/`ffprobe` available on `PATH`.
3. `pip install -r requirements.txt`.
4. `uvicorn backend.main:app --reload`, then open `http://localhost:8000`.

## Modal.com (optional GPU execution)

To run transcription on Modal.com instead of locally, you'll need a Modal
account and a **Token ID** and **Token Secret** (both entered directly in
the app UI — never stored on disk). Details on GPU type selection and
pricing will be documented here once Phase 4 lands.

## License

MIT — see [LICENSE](LICENSE).
