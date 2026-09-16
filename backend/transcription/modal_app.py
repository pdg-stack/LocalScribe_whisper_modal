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

# Uploaded audio used to be written to the container's own local /tmp
# disk -- fine as long as the same container serves both the upload and
# the later transcribe() call, which is only a best-effort optimization,
# not a guarantee: Modal's own docs confirm every GPU Function is subject
# to preemption ("the `nonpreemptible` parameter is not supported for GPU
# Functions"), with likelihood rising the longer a Function runs -- and
# this job's Upload pass, uploading every file before transcribing any of
# them, can itself run the better part of an hour. A real run hit exactly
# this: 13 of 20 files' uploads landed on a container that was gone by
# the time Transcription got to them (see logs/ for that run; see
# pipeline.py's _is_missing_remote_upload for the retry that recovers
# from it after the fact). A Volume fixes the actual cause instead of
# only recovering from it: writes committed here are durable, external
# storage, not tied to any one container's lifetime -- whichever
# container is current when transcribe() runs can read a file committed
# by a *different*, since-replaced container, once it reload()s. Kept as
# its own Volume (not reused from MODEL_CACHE_VOLUME) since this one
# holds short-lived per-job data that gets deleted right after each
# file's Transcription, not a long-lived cache.
AUDIO_UPLOAD_DIR = "/audio_uploads"
AUDIO_UPLOAD_VOLUME = modal.Volume.from_name("localscribe-audio-uploads", create_if_missing=True)

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
            return f"{AUDIO_UPLOAD_DIR}/upload_{upload_id}.wav"

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
        def commit_upload(self, upload_id: str) -> None:
            # Called once, after every chunk for this file has been
            # written -- makes the file durable, external-storage data
            # instead of only existing on this container's local view of
            # the Volume mount. Without this, a file that's fully written
            # but never committed is just as vulnerable to this
            # container being replaced as it was on plain local /tmp.
            AUDIO_UPLOAD_VOLUME.commit()

        @modal.method()
        def transcribe(self, upload_id: str, beam_size: int, progress_queue: modal.Queue | None = None) -> dict:
            import os
            import time

            # Picks up any commits made since this container's Volume
            # mount was first attached -- specifically covers the case
            # this Volume exists for: the file committed by a *different*
            # (since-replaced) container. A no-op, cheap call when
            # nothing's changed (e.g. the common case of the same
            # container that just uploaded it also transcribing it).
            AUDIO_UPLOAD_VOLUME.reload()
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
            # Deletion is its own explicit step (delete_upload() below,
            # called from pipeline.py as "Cleanup - Modal.com Audio
            # Uploads", after Download) rather than happening silently
            # here -- gives the user a visible step/progress for it, the
            # same as every other phase, instead of hiding real work
            # (and a real network round-trip for the commit) inside what
            # the UI shows as "Transcription."
            return {"segments": segments, "inference_sec": inference_sec}

        @modal.method()
        def delete_upload(self, upload_id: str) -> None:
            # Explicit, visible cleanup of this file's audio from the
            # Volume -- matters more here than it ever did for a
            # container's own /tmp: unlike /tmp (which just disappears
            # when the container goes away), this Volume is a named,
            # persistent resource that outlives the job, so every file's
            # audio would otherwise pile up here indefinitely across
            # every job ever run. Committed so the deletion itself is
            # durable too, not just removed from this container's local
            # view of the mount. A missing file (e.g. cleanup running
            # twice, or the file was never actually written) is not an
            # error -- there's nothing left to clean up either way.
            import os
            try:
                os.remove(self._upload_path(upload_id))
                AUDIO_UPLOAD_VOLUME.commit()
            except OSError:
                pass

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
        #
        # max_containers=1 is load-bearing, not an optimization -- and
        # solves a *different* failure mode than AUDIO_UPLOAD_VOLUME
        # above, even though both were diagnosed from similar-looking
        # symptoms. Modal's default pool-based scheduling only reuses a
        # warm container as a best-effort optimization; it does not pin a
        # class instance to one physical container across separate
        # .remote()/.spawn() calls, and will spin up an extra *concurrent*
        # replica if it decides one is warranted. A real job hit exactly
        # that: 15 of 23 files' Transcription calls got routed to a
        # container that never received their upload_chunk() calls,
        # failing with PyAV FileNotFoundError/InvalidDataError reading
        # the missing/partial WAV (see logs/ for the run this was
        # diagnosed from). Capping the pool at 1 makes that impossible --
        # every call queues for the one container instead. (The Volume
        # switch above independently protects against the *sequential*
        # version of the same symptom -- one container replacing another
        # over time, e.g. from Modal.com's GPU preemption, which
        # max_containers=1 alone does nothing to prevent.) scaledown_window
        # is raised well past any realistic gap between this job's calls
        # (a slow per-file extraction/upload can easily exceed Modal's
        # ~60s default) so an idle lull mid-job can't get the single
        # container recycled either; this doesn't add cost, since the
        # container's real lifetime is still bounded by our own explicit
        # teardown either way (see __exit__).
        # timeout is the max wall time Modal allows for a SINGLE call to
        # any one method on this class (self-imposed, not Modal's own
        # platform ceiling of 24h) -- transcribe() is the one that can
        # actually run long, and it needs real headroom: a slow
        # model/GPU pairing on a long lecture recording can genuinely
        # take the better part of an hour (e.g. large-v3 on a T4, RTF
        # ~0.145, over a 6h file, is ~29 real minutes of inference alone)
        # -- 600s (10 min, the previous value) was too tight for that,
        # and a call that legitimately needs more than its configured
        # timeout gets killed by Modal exactly like a genuinely stuck one
        # would (see transcribe()'s FunctionTimeoutError handling below).
        # 2 hours is generous headroom for any realistic single file
        # while still bounding a truly hung call rather than waiting
        # forever.
        remote_cls = self._app.cls(
            image=IMAGE, gpu=gpu_id, timeout=7200,
            volumes={"/cache": MODEL_CACHE_VOLUME, AUDIO_UPLOAD_DIR: AUDIO_UPLOAD_VOLUME},
            secrets=secrets,
            serialized=True,
            max_containers=1,
            scaledown_window=1800,
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
            except modal.exception.FunctionTimeoutError:
                # A *real*, terminal failure -- see transcribe()'s except
                # block below for why this must never be caught by the
                # "still polling" branch it would otherwise fall into.
                raise
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
        transcribe(upload_id, ...) for that same upload_id.

        The final commit_upload() call is what actually makes this file's
        data durable (see AUDIO_UPLOAD_VOLUME's docstring) -- deliberately
        made only once, after every chunk, rather than per-chunk: a
        cancelled/interrupted upload has nothing worth committing anyway
        (transcribe() would just fail on a partial file, same as if it
        were never committed at all), so there's no reason to pay a
        commit's cost after every single chunk."""
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
        self._instance.commit_upload.remote(upload_id)

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
                except modal.exception.FunctionTimeoutError:
                    # A *real*, terminal failure: the remote transcribe()
                    # call itself hit this class's own timeout=600-ish
                    # config (see ModalTranscriber.__init__) and Modal
                    # killed it server-side -- confirmed by reading
                    # modal's _process_result(), which raises exactly
                    # this for a GENERIC_STATUS_TIMEOUT result. This is
                    # NOT the same thing as the plain poll timing out
                    # below (the 1.0s `call.get(timeout=...)` just not
                    # having a result *yet*) -- it's the actual
                    # function-level deadline being exceeded, and the
                    # call is now dead; nothing more will ever come back
                    # from it. FunctionTimeoutError IS a subclass of
                    # modal.exception.TimeoutError (confirmed by reading
                    # its source), so without this branch coming first,
                    # the broader except below would swallow it as just
                    # "not done yet" and this loop would poll forever,
                    # in practice hanging the job on this file with no
                    # error ever surfacing -- exactly what happened in a
                    # real run once a file's Transcription ran long
                    # enough to hit it (see this project's logs/ for that
                    # run). Must propagate so _run_single_step's except
                    # Exception block can fail this step visibly instead.
                    raise
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

    def delete_upload(self, upload_id: str, should_cancel: Callable[[], bool] | None = None) -> None:
        """Removes this file's audio from AUDIO_UPLOAD_VOLUME once its
        Transcription (and Download, a formality) are done with it --
        called once per file, as its own "Cleanup - Modal.com Audio
        Uploads" step (see pipeline.py), the same way Cleanup -
        Intermediate Files explicitly cleans up the local WAV. should_cancel
        is checked before the call (not mid-call -- a single delete+commit
        is not meaningfully interruptible partway through) purely so a
        cancel click doesn't dispatch further cleanup calls once the user
        has asked to stop."""
        if should_cancel is not None and should_cancel():
            raise StepCancelled("Modal.com cleanup cancelled")
        self._instance.delete_upload.remote(upload_id)
