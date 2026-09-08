"""Static configuration: media extensions, model/GPU choices, estimate tables."""

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "json", "tsv"]

# faster-whisper's own default is 5; matches reference Whisper's decoding
# accuracy. Not exposed in the UI -- see plan for rationale.
BEAM_SIZE = 5

GPU_OPTIONS = [
    {"id": "T4", "label": "T4", "rate": 0.59},
    {"id": "L4", "label": "L4", "rate": 0.80},
    {"id": "A10G", "label": "A10G", "rate": 1.10},
    {"id": "A100", "label": "A100", "rate": 2.10},
    {"id": "H100", "label": "H100", "rate": 3.95},
]
RECOMMENDED_GPU = "L4"

# Approximate real-time-factor (processing seconds per second of audio),
# calibrated for beam_size=5. Mirrors frontend/app.js's placeholder tables
# used for the Preview panel before this backend existed.
RTF_LOCAL_CPU = {
    "tiny": 0.10, "base": 0.15, "small": 0.25,
    "medium": 0.45, "large-v3": 0.70, "large-v3-turbo": 0.35,
}
RTF_MODAL_GPU = {
    "tiny": 0.015, "base": 0.02, "small": 0.03,
    "medium": 0.05, "large-v3": 0.08, "large-v3-turbo": 0.045,
}

MODAL_SETUP_SEC = 20  # cold start: spin instance + install/load model
MODAL_DOWNLOAD_SEC = 2  # transcript result back to local
