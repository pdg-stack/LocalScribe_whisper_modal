"""Modal.com GPU execution: an ephemeral App/Function built fresh per
request (no `modal deploy` needed) so the GPU type and credentials can
vary per call. Credentials are set as env vars right before the call --
never sent anywhere but Modal itself (see main.py: they're also saved in
plaintext to user_prefs.json at the user's request, purely for
convenience -- that's a separate concern from this module).
"""

from __future__ import annotations

import os
from typing import Callable

import modal

from backend.jobs import StepCancelled

# Pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 on bare debian_slim
# proved unreliable (missing plain-named .so symlinks ctranslate2 dlopen()s
# by exact filename, regardless of LD_LIBRARY_PATH/import-order fixes) --
# NVIDIA's own CUDA+cuDNN runtime image has these correctly registered via
# ldconfig already, which is the standard fix for this class of issue.
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("faster-whisper")
)

# Each call builds a fresh ephemeral container (no `modal deploy`), which
# by itself has no persistent disk -- without a Volume, faster-whisper
# would re-download the model from Hugging Face on every single Modal
# run, which is both slow and what triggers HF's unauthenticated-request
# rate-limit warning. Mounting a Volume at the model cache path makes the
# download happen once; every call after that reuses the cached weights.
MODEL_CACHE_DIR = "/cache/huggingface"
MODEL_CACHE_VOLUME = modal.Volume.from_name("localscribe-whisper-model-cache", create_if_missing=True)


def _transcribe_remote(audio_bytes: bytes, model_name: str, beam_size: int) -> dict:
    # Runs inside the Modal container -- imports are local to it.
    import tempfile
    import time

    from faster_whisper import WhisperModel

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    model = WhisperModel(model_name, device="cuda", compute_type="float16", download_root=MODEL_CACHE_DIR)
    # model.transcribe() returns a lazy generator -- inference actually
    # happens while iterating it below, so the timer has to wrap that, not
    # the call itself, to measure real inference time (excluding model
    # load, which isn't proportional to audio length).
    started = time.perf_counter()
    segments_iter, _info = model.transcribe(path, beam_size=beam_size)
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments_iter]
    inference_sec = time.perf_counter() - started
    return {"segments": segments, "inference_sec": inference_sec}


def transcribe_on_modal(
    audio_bytes: bytes,
    model_name: str,
    gpu_id: str,
    beam_size: int,
    token_id: str,
    token_secret: str,
    hf_token: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Returns {"segments": [...], "inference_sec": float}. inference_sec
    is real GPU inference time only (excludes cold start/model load) --
    used to calibrate future time estimates for this (model, GPU) pair;
    billing cost is still based on the full RPC wall time by the caller,
    since that's what Modal actually charges for.

    Uses .spawn() + polled .get(timeout=...) rather than a single blocking
    .remote() call, so should_cancel() can be checked every second; on
    cancellation .cancel(terminate_containers=True) actually kills the
    remote container immediately (stopping billing), not just abandons
    the local wait."""
    # Env vars take priority over any ~/.modal.toml active profile (see
    # Modal's SDK config docs) -- this call always uses the credentials
    # supplied for this request, regardless of what's configured locally.
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret

    # modal.Secret injects HF_TOKEN into the *remote container's*
    # environment (unlike MODAL_TOKEN_ID/SECRET above, which only affect
    # this local process talking to Modal) -- huggingface_hub picks it up
    # automatically from there when downloading model weights.
    secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []

    app = modal.App("localscribe-whisper-modal")
    remote_fn = app.function(
        image=IMAGE, gpu=gpu_id, timeout=600,
        volumes={"/cache": MODEL_CACHE_VOLUME},
        secrets=secrets,
    )(_transcribe_remote)
    with app.run():
        call = remote_fn.spawn(audio_bytes, model_name, beam_size)
        while True:
            try:
                return call.get(timeout=1.0)
            except (TimeoutError, modal.exception.TimeoutError):
                # A poll timeout just means "not done yet" -- FunctionCall.get()
                # raises Python's plain built-in TimeoutError for this in the
                # installed modal SDK (confirmed by reading its source), not
                # modal.exception.TimeoutError as the name would suggest; the
                # two are unrelated classes, so both are caught here to be
                # safe across SDK versions. Anything else (auth, network,
                # real errors) propagates normally.
                if should_cancel is not None and should_cancel():
                    call.cancel(terminate_containers=True)
                    raise StepCancelled("Modal.com transcription cancelled") from None
