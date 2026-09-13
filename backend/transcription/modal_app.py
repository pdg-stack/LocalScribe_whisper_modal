"""Modal.com GPU execution: an ephemeral App/container class built fresh
per job (no `modal deploy` needed) so the GPU type and credentials can vary
per request. Credentials are set as env vars right before the call -- never
sent anywhere but Modal itself (see main.py: they're also saved in
plaintext to user_prefs.json at the user's request, purely for
convenience -- that's a separate concern from this module).

One ModalTranscriber is opened per job (not per file): it keeps a single
`app.run()` session and a single warm container/model instance for every
file in the job. The model loads once, in the container's `@modal.enter()`
lifecycle hook, and every file's Transcription step re-uses that same
loaded-in-GPU-memory model via a `.spawn()` call on the same instance --
rather than the old per-file design, which built a brand new ephemeral App
and re-loaded the model into a fresh container for every single file.
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

# Without a Volume, faster-whisper would re-download the model from
# Hugging Face into every fresh container, which is both slow and what
# triggers HF's unauthenticated-request rate-limit warning. Mounting a
# Volume at the model cache path makes the download happen once; every
# container after that reuses the cached weights.
MODEL_CACHE_DIR = "/cache/huggingface"
MODEL_CACHE_VOLUME = modal.Volume.from_name("localscribe-whisper-model-cache", create_if_missing=True)

# Uploading the whole WAV as a single function argument (the old design)
# is one blocking network call with no visibility into progress and no
# way to check should_cancel() until it's fully done -- for a long file
# that can be 60-200+ seconds with the UI unable to tell "uploading" from
# "hung" (and for anything over Modal's ~100MB per-call gRPC payload
# limit, it fails outright). Chunking it into many small RPCs instead
# gives should_cancel() a check between every chunk and a real progress
# callback after each one.
#
# 16 MB is a deliberate middle ground, not a measured optimum: a bigger
# chunk means fewer RPC round-trips (less fixed per-call overhead) but
# worse cancel/progress granularity, since a chunk already in flight
# can't be interrupted. At the conservative ~1MB/s assumed upload
# bandwidth (see config.MODAL_UPLOAD_BYTES_PER_SEC), 16MB keeps
# worst-case cancel latency around 16s while cutting a 200MB file's
# chunk count ~4x versus a smaller 4MB chunk (50 calls -> ~13) -- and
# still leaves a healthy ~6x margin under the 100MB hard limit.
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


def _build_transcriber_cls(model_name: str):
    # Built dynamically (per job) so model_name can be closed over without
    # needing a modal.parameter() -- a single job always transcribes with
    # one model, so one class per job is all that's needed.
    class _RemoteTranscriber:
        # A plain (non-@modal.method()) helper -- bundled into the same
        # cloudpickled class as everything else below, unlike a
        # module-level function would be. serialized=True ships this
        # class by pickling it directly (see ModalTranscriber.__init__),
        # and cloudpickle resolves a *module-level* function reference
        # "by reference" (module path + name) rather than by value --
        # which, before this was made a method, made the container try
        # to `import backend.transcription.modal_app` to resolve it and
        # fail with "No module named 'backend'", since only
        # faster-whisper is installed in the remote image, never our own
        # package. A method defined right here has no such problem: it's
        # part of the one self-contained blob that gets shipped over.
        def _upload_path(self, upload_id: str) -> str:
            return f"/tmp/upload_{upload_id}.wav"

        @modal.enter()
        def load_model(self):
            # Runs once per container, when it starts -- not per file.
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_name, device="cuda", compute_type="float16", download_root=MODEL_CACHE_DIR)

        @modal.method()
        def ping(self) -> bool:
            # Trivial remote call with no real work of its own -- calling
            # it is just a way to force @modal.enter() (container start +
            # model load) to happen now, as its own explicit "Setup" step,
            # rather than lazily on whichever file's transcribe() call
            # happens to be first.
            return True

        @modal.method()
        def start_upload(self, upload_id: str) -> None:
            # Truncates/creates the file this upload's chunks will be
            # appended to -- guards against ever accidentally appending
            # onto a stale file left by an earlier, unrelated upload_id.
            with open(self._upload_path(upload_id), "wb"):
                pass

        @modal.method()
        def upload_chunk(self, upload_id: str, chunk: bytes) -> None:
            with open(self._upload_path(upload_id), "ab") as f:
                f.write(chunk)

        @modal.method()
        def transcribe(self, upload_id: str, beam_size: int, progress_queue: modal.Queue | None = None) -> dict:
            import os
            import time

            path = self._upload_path(upload_id)

            # model.transcribe() returns a lazy generator -- inference
            # actually happens while iterating it below, so the timer has
            # to wrap that, not the call itself, to measure real inference
            # time (excluding model load, which isn't proportional to
            # audio length, and already happened once in load_model()).
            started = time.perf_counter()
            segments_iter, _info = self.model.transcribe(path, beam_size=beam_size)
            segments = []
            for s in segments_iter:
                segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
                if progress_queue is not None:
                    # Best-effort progress signal back to the local caller
                    # (polling the same queue) -- a full queue would mean
                    # progress isn't being drained, which shouldn't ever
                    # happen here, but this must never be what fails a
                    # transcription, so swallow a full queue rather than
                    # blocking or propagating.
                    try:
                        progress_queue.put(s.end, block=False)
                    except Exception:
                        pass
            inference_sec = time.perf_counter() - started
            # Frees the container's local disk now rather than waiting for
            # teardown -- matters since every file in the job shares this
            # same warm container/disk (see ModalTranscriber's docstring).
            try:
                os.remove(path)
            except OSError:
                pass
            return {"segments": segments, "inference_sec": inference_sec}

    return _RemoteTranscriber


class ModalTranscriber:
    """Holds one ephemeral Modal App, one warm container class, and one
    `app.run()` session for an entire job's worth of files. Use as a
    context manager around the job's Upload/Transcription steps:

        with ModalTranscriber(gpu, model, token_id, token_secret, hf_token) as t:
            for file in files:
                t.upload(audio_bytes, upload_id, should_cancel=...)
                t.transcribe(upload_id, beam_size, should_cancel=...)

    Every file's calls within that `with` block reuse the same container
    and the same already-loaded model -- Modal only cold-starts once for
    the whole job (or not at all, if a container from a previous job is
    still warm), instead of once per file.
    """

    def __init__(
        self,
        gpu_id: str,
        model_name: str,
        token_id: str,
        token_secret: str,
        hf_token: str | None = None,
    ) -> None:
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

        self._app = modal.App("localscribe-whisper-modal")
        # serialized=True is required here: _RemoteTranscriber is built
        # fresh per job (see _build_transcriber_cls) so it can close over
        # this job's model_name, which makes it a locally-scoped class --
        # Modal's default cls handling only supports classes importable by
        # reference from module scope, and raises LocalFunctionError
        # otherwise. serialized=True tells it to cloudpickle the class
        # directly instead, which is exactly what a dynamically built
        # class needs.
        remote_cls = self._app.cls(
            image=IMAGE, gpu=gpu_id, timeout=600,
            volumes={"/cache": MODEL_CACHE_VOLUME},
            secrets=secrets,
            serialized=True,
        )(_build_transcriber_cls(model_name))
        self._instance = remote_cls()
        self._run_ctx = None

    @property
    def is_active(self) -> bool:
        """True once __enter__ has run and __exit__ hasn't -- lets a
        caller tell whether a later __exit__() call would do real work or
        just be the idempotent no-op (see __exit__ below)."""
        return self._run_ctx is not None

    def __enter__(self) -> "ModalTranscriber":
        self._run_ctx = self._app.run()
        self._run_ctx.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        # Idempotent: the job's "Modal.com Teardown" step calls this
        # explicitly (as a visible, catchable pipeline step), and run_job's
        # own finally block calls it again unconditionally as a safety net
        # for early/abnormal exits (e.g. cancelled before that step ran) --
        # a no-op the second time keeps that safety net from double-exiting
        # an already-closed session.
        if self._run_ctx is None:
            return
        self._run_ctx.__exit__(*exc_info)
        self._run_ctx = None

    def warm_up(self, should_cancel: Callable[[], bool] | None = None) -> None:
        """Forces the container to start and the model to load now, as
        the job's single, explicit Setup step -- run once, before any
        file's Transcription -- instead of paying that cold-start cost
        implicitly on whichever file happens to run first."""
        call = self._instance.ping.spawn()
        while True:
            try:
                call.get(timeout=1.0)
                return
            except (TimeoutError, modal.exception.TimeoutError):
                if should_cancel is not None and should_cancel():
                    call.cancel(terminate_containers=True)
                    raise StepCancelled("Modal.com setup cancelled") from None

    def upload(
        self,
        audio_bytes: bytes,
        upload_id: str,
        should_cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Uploads audio_bytes to the container in UPLOAD_CHUNK_BYTES
        pieces, each its own blocking RPC, rather than one call carrying
        the whole payload -- so should_cancel() can be checked between
        chunks (cancelling mid-upload only ever waits for the current
        chunk, a few seconds at most, not the whole file) and on_progress
        gets real bytes-uploaded-so-far. Call once per file before
        transcribe(upload_id, ...) for that same upload_id."""
        self._instance.start_upload.remote(upload_id)
        total = len(audio_bytes)
        sent = 0
        for offset in range(0, total, UPLOAD_CHUNK_BYTES):
            if should_cancel is not None and should_cancel():
                raise StepCancelled("Modal.com upload cancelled")
            chunk = audio_bytes[offset:offset + UPLOAD_CHUNK_BYTES]
            self._instance.upload_chunk.remote(upload_id, chunk)
            sent += len(chunk)
            if on_progress is not None:
                on_progress(sent / total if total > 0 else 1.0)

    def transcribe(
        self,
        upload_id: str,
        beam_size: int,
        should_cancel: Callable[[], bool] | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Returns {"segments": [...], "inference_sec": float}. Transcribes
        whatever upload(audio_bytes, upload_id, ...) already placed on the
        container for this upload_id -- this call itself only ever sends
        a small id string, not the audio, so it starts immediately rather
        than blocking on a large upload first.

        inference_sec is real GPU inference time only (excludes model
        load, which already happened once for the container) -- used to
        calibrate future time estimates for this (model, GPU) pair;
        billing cost is still based on the full RPC wall time by the
        caller, since that's what Modal actually charges for.

        Uses .spawn() + polled .get(timeout=...) rather than a single
        blocking .remote() call, so should_cancel() can be checked every
        second; on cancellation .cancel(terminate_containers=True)
        actually kills the remote container immediately (stopping
        billing), not just abandons the local wait. An ephemeral Queue is
        passed into the remote call purely so the container can report
        per-segment progress back to on_progress while it's still
        running -- otherwise this RPC returns nothing at all until the
        entire file (which can be very long) is fully transcribed."""
        with modal.Queue.ephemeral() as progress_queue:
            call = self._instance.transcribe.spawn(upload_id, beam_size, progress_queue)
            while True:
                if on_progress is not None:
                    for processed_sec in progress_queue.get_many(1000, block=False):
                        on_progress(processed_sec)
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
