"""Modal.com GPU execution: an ephemeral App/Function built fresh per
request (no `modal deploy` needed) so the GPU type and credentials can
vary per call. Credentials are set as env vars right before the call --
they're never written to disk (see main.py: only ever taken from the
request body, never persisted to user_prefs.json).
"""

from __future__ import annotations

import os

import modal

# Pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 on bare debian_slim
# proved unreliable (missing plain-named .so symlinks ctranslate2 dlopen()s
# by exact filename, regardless of LD_LIBRARY_PATH/import-order fixes) --
# NVIDIA's own CUDA+cuDNN runtime image has these correctly registered via
# ldconfig already, which is the standard fix for this class of issue.
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("faster-whisper")
)


def _transcribe_remote(audio_bytes: bytes, model_name: str, beam_size: int) -> list[dict]:
    # Runs inside the Modal container -- imports are local to it.
    import tempfile

    from faster_whisper import WhisperModel

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, _info = model.transcribe(path, beam_size=beam_size)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def transcribe_on_modal(
    audio_bytes: bytes,
    model_name: str,
    gpu_id: str,
    beam_size: int,
    token_id: str,
    token_secret: str,
) -> list[dict]:
    # Env vars take priority over any ~/.modal.toml active profile (see
    # Modal's SDK config docs) -- this call always uses the credentials
    # supplied for this request, regardless of what's configured locally.
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret

    app = modal.App("localscribe-whisper-modal")
    remote_fn = app.function(image=IMAGE, gpu=gpu_id, timeout=600)(_transcribe_remote)
    with app.run():
        return remote_fn.remote(audio_bytes, model_name, beam_size)
