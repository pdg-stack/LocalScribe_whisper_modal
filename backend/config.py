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
# calibrated for beam_size=5. RTF_MODAL_GPU is the baseline for our
# recommended default GPU (L4); GPU_SPEED_MULTIPLIER scales it for the
# others (see estimator.py). No public benchmark table is fully reliable
# here -- exact throughput depends on hardware/driver/batch/load -- so
# these are rough starting points refined by backend/calibration.py's
# self-calibration from real completed jobs, which estimator.py prefers
# once samples exist for a given (model, device) pair.
RTF_LOCAL_CPU = {
    "tiny": 0.10, "base": 0.15, "small": 0.25,
    "medium": 0.45, "large-v3": 0.70, "large-v3-turbo": 0.35,
}
RTF_MODAL_GPU = {
    "tiny": 0.015, "base": 0.02, "small": 0.03,
    "medium": 0.05, "large-v3": 0.08, "large-v3-turbo": 0.045,
}

# Relative processing speed vs. L4 (1.0 = same speed as the RTF_MODAL_GPU
# baseline above), derived from published FP16 TFLOPS ratios as a rough
# starting point -- e.g. T4 ~65 TFLOPS vs L4 ~121 TFLOPS. Real-world
# Whisper speedup is usually somewhat less than the raw TFLOPS ratio since
# decode is partly memory-bandwidth-bound, so these lean conservative.
GPU_SPEED_MULTIPLIER = {
    "T4": 0.55,
    "L4": 1.0,
    "A10G": 1.05,
    "A100": 2.3,
    "H100": 3.8,
}

MODAL_SETUP_SEC = 20  # cold start: spin instance + install/load model
MODAL_DOWNLOAD_SEC = 2  # transcript result back to local

# Rough estimate assuming the model is already cached locally (fast path).
# A genuine first-time download depends entirely on the user's internet
# speed and model size, so this is a starting point only -- real observed
# load time (which naturally includes any download) gets calibrated via
# backend/calibration.py after the first local job with each model, same
# self-correcting mechanism as the RTF tables above.
LOCAL_MODEL_SETUP_SEC = {
    "tiny": 3, "base": 4, "small": 6,
    "medium": 10, "large-v3": 18, "large-v3-turbo": 14,
}
