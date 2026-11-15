"""Static configuration: media extensions, model/GPU choices, estimate tables."""

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
OUTPUT_FORMATS = ["txt", "srt", "vtt", "json", "tsv"]

# faster-whisper's own default is 5; matches reference Whisper's decoding
# accuracy. Not exposed in the UI -- see plan for rationale.
BEAM_SIZE = 5

# ---------------------------------------------------------------------------
# Language / task
# ---------------------------------------------------------------------------

# The 100 language codes faster-whisper 1.2.1 accepts for `language=`
# (faster_whisper.tokenizer._LANGUAGE_CODES). Whisper writes each language
# in its own native script, so this list doubles as the "transcription
# script" choice. "auto" is not a Whisper code -- it means "let Whisper
# detect it" (language=None) and is the only value that can mis-detect.
DEFAULT_LANGUAGE = "en"
AUTO_LANGUAGE = "auto"
WHISPER_LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao",
    "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Myanmar (Burmese)", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk", "no": "Norwegian",
    "oc": "Occitan", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala",
    "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
    "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

# Whisper's translate task only ever outputs English. large-v3-turbo was
# fine-tuned on transcription data only, so with task="translate" it tends
# to ignore the task and emit source-language text -- silently wrong, so
# it's blocked rather than allowed.
TRANSLATE_UNSUPPORTED_MODELS = {"large-v3-turbo"}


def validate_language_options(model: str, language: str, translate: bool) -> None:
    """Raises ValueError with a user-facing message for an invalid
    language/translate combination. Used by the API layer as the backend
    twin of the frontend's disabled checkbox."""
    if language != AUTO_LANGUAGE and language not in WHISPER_LANGUAGES:
        raise ValueError(f"Unsupported language code: {language!r}.")
    if translate and model in TRANSLATE_UNSUPPORTED_MODELS:
        raise ValueError(
            f"The {model} model does not support translation to English -- choose another "
            "model (e.g. medium or large-v3) or untick 'Translate to English'."
        )


def build_decode_options(language: str, translate: bool) -> dict:
    """Keyword arguments for faster-whisper's model.transcribe(), shared by
    the local and Modal paths so both decode identically. Plain
    JSON-friendly values only -- this dict is shipped to the Modal
    container as an RPC argument.

    The "silent" accuracy settings (not exposed in the UI):
    - vad_filter: drops silence/music/noise before decoding. Cuts
      hallucinated text over quiet stretches and keeps language detection
      from being skewed by a silent/music intro.
    - condition_on_previous_text=False: stops one bad segment (wrong
      language, repetition loop) from being fed forward as context and
      poisoning every segment after it. Slightly less cross-segment
      consistency, far fewer runaway failures on long recordings.
    - temperature fallback + compression_ratio/log_prob/no_speech
      thresholds: Whisper re-decodes a segment at higher temperature when
      it looks repetitive or low-confidence (these are faster-whisper's
      defaults, pinned here so an upstream default change can't silently
      alter accuracy).
    - hallucination_silence_threshold: with word timestamps on, skips
      segments that are hallucinated over >2s of silence.
    - language_detection_segments/threshold: for Auto only, detect from up
      to 4 windows (not just the first 30s) and only trust a result with
      >=0.5 probability.
    - A pinned language skips detection entirely -- the only fully
      reliable setting.
    """
    # Translate: the UI fixes the language box to "English" (the *output*
    # language), but passing language="en" would tell Whisper the audio is
    # already English and it would skip translating. So for translate the
    # source language is always auto-detected.
    is_auto = language == AUTO_LANGUAGE or translate
    return {
        "language": None if is_auto else language,
        "task": "translate" if translate else "transcribe",
        "beam_size": BEAM_SIZE,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": False,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "hallucination_silence_threshold": 2.0,
        "language_detection_segments": 4,
        "language_detection_threshold": 0.5,
    }


# ---------------------------------------------------------------------------
# Model weight storage
# ---------------------------------------------------------------------------

# Local execution stores downloaded Whisper weights here (override with the
# LOCALSCRIBE_MODEL_DIR env var), passed to faster-whisper as download_root
# -- a fixed, project-local folder (gitignored) that outlives the process,
# so a model is downloaded once and reused by every later job and restart.
# Modal execution has its own, separate Volume (modal_app.MODEL_CACHE_VOLUME).
import os as _os
from pathlib import Path as _Path

LOCAL_MODEL_DIR = _Path(
    _os.environ.get("LOCALSCRIBE_MODEL_DIR") or _Path(__file__).resolve().parent.parent / "models"
)

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

MODAL_SETUP_SEC = 20  # cold start: spin instance + install/load model -- fallback only; see calibration.py
MODAL_AUDIO_CLEANUP_SEC = 2  # delete + commit one file's audio off AUDIO_UPLOAD_VOLUME

# ffmpeg audio-extraction time, in seconds per minute of source media --
# fallback only, before any real samples exist (extraction is fast: no
# video re-encoding, just demuxing + decoding + writing WAV, so real
# observed throughput on real hardware has been under a tenth of this).
# backend/calibration.py prefers actually-measured throughput from
# completed jobs once available, same self-correcting mechanism as the
# RTF/setup/upload estimates above.
AUDIO_EXTRACTION_SEC_PER_MIN = 2.0

# The extracted WAV's fixed bitrate (16kHz, mono, 16-bit -- matches
# ffmpeg_utils.extract_audio's output format exactly), used to estimate
# how many bytes a file's "Upload audio files to Modal.com" step will
# need to send.
WAV_BYTES_PER_SEC = 32_000

# Conservative assumed upload bandwidth (~8 Mbps) for estimating how long
# uploading that WAV will take -- real speed varies enormously by
# connection, so this is only the fallback used before any real samples
# exist; backend/calibration.py prefers actually-measured throughput from
# completed jobs once available, same self-correcting mechanism as the
# RTF/setup tables above.
MODAL_UPLOAD_BYTES_PER_SEC = 1_000_000

# Modal's own per-container ephemeral disk quota defaults to 512 GiB
# (confirmed via Modal's docs: "a per-container disk quota that defaults
# to 512 GiB", raisable up to 3 TiB via the `ephemeral_disk` Function
# parameter). Every file in a job uploads to that same shared container's
# /tmp *before* any of them are transcribed and deleted (Upload is its
# own pass, ahead of Transcription -- see estimator.py/pipeline.py), so
# the true peak usage is the *whole batch's* total size, not any single
# file's. 500 GB (decimal, not GiB -- deliberately the more conservative
# of the two units) leaves a real margin below the 512 GiB default quota
# rather than cutting it close.
MAX_MODAL_UPLOAD_BYTES = 500 * 1_000_000_000

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
