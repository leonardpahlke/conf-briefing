"""Video transcription using faster-whisper."""

import json
import logging
import time
from pathlib import Path

from conf_briefing.config import Config
from conf_briefing.console import console, tag


def transcribe_video(
    video_path: Path,
    output_dir: Path,
    model,
    model_name: str,
    initial_prompt: str = "",
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON."""
    segments_iter, info = model.transcribe(
        str(video_path),
        beam_size=5,
        initial_prompt=initial_prompt or None,
    )

    segments = []
    full_parts = []
    for seg in segments_iter:
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text})
        full_parts.append(seg.text)

    video_id = video_path.stem

    # Load title from metadata sidecar written during download
    title = ""
    meta_path = video_path.parent / f"{video_id}.meta.json"
    if meta_path.exists():
        try:
            title = json.loads(meta_path.read_text()).get("title", "")
        except (json.JSONDecodeError, OSError):
            pass

    transcript = {
        "video_id": video_id,
        "title": title,
        "language": info.language,
        "duration_sec": round(info.duration, 1),
        "model": model_name,
        "full_text": " ".join(full_parts),
        "segments": segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return out_path


def _build_whisperx_output(
    result: dict,
    video_path: Path,
    output_dir: Path,
    model_name: str,
    language: str,
    diarized: bool,
) -> Path:
    """Build transcript JSON from WhisperX result. Shared by single/batch paths."""
    import contextlib

    # Build speaker label map (SPEAKER_00 → Speaker 1, etc.)
    speaker_map: dict[str, str] = {}
    if diarized:
        raw_speakers = sorted(
            {
                seg.get("speaker", "")
                for seg in result.get("segments", [])
                if seg.get("speaker")
            }
        )
        speaker_map = {s: f"Speaker {i + 1}" for i, s in enumerate(raw_speakers)}

    segments = []
    full_parts = []
    for seg in result.get("segments", []):
        entry = {
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", ""),
        }
        raw_speaker = seg.get("speaker", "")
        if raw_speaker:
            entry["speaker"] = speaker_map.get(raw_speaker, raw_speaker)
        segments.append(entry)

        text = seg.get("text", "").strip()
        if text:
            label = speaker_map.get(raw_speaker, raw_speaker) if raw_speaker else ""
            full_parts.append(f"[{label}] {text}" if label else text)

    video_id = video_path.stem

    title = ""
    meta_path = video_path.parent / f"{video_id}.meta.json"
    if meta_path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            title = json.loads(meta_path.read_text()).get("title", "")

    duration = segments[-1]["end"] if segments else 0

    transcript = {
        "video_id": video_id,
        "title": title,
        "language": language,
        "duration_sec": round(duration, 1),
        "model": model_name,
        "backend": "whisperx",
        "diarized": diarized,
        "full_text": " ".join(full_parts),
        "segments": segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.json"
    out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return out_path


# Default batch size for WhisperX inference. 16 works well for >=8 GB VRAM;
# reduced automatically to 8 on CPU to limit memory pressure.
_WHISPERX_BATCH_SIZE_GPU = 16
_WHISPERX_BATCH_SIZE_CPU = 8

# Conference talks typically have 1-5 speakers (presenter + Q&A audience).
_DIARIZE_MIN_SPEAKERS = 1
_DIARIZE_MAX_SPEAKERS = 5


def transcribe_all(
    config: Config,
    device: str,
    compute_type: str,
    initial_prompt: str = "",
) -> list[Path]:
    """Transcribe all downloaded videos. Returns list of transcript file paths."""
    data_dir = config.data_dir
    videos_dir = data_dir / "videos"
    output_dir = data_dir / "transcripts"

    if not videos_dir.exists():
        console.print(f"{tag('whisper')} No videos directory found, skipping transcription.")
        return []

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        console.print(f"{tag('whisper')} No video files found.")
        return []

    # Filter out already-transcribed videos
    to_process = []
    existing = []
    for vf in video_files:
        transcript_path = output_dir / f"{vf.stem}.json"
        if transcript_path.exists():
            existing.append(transcript_path)
        else:
            to_process.append(vf)

    if existing:
        console.print(f"{tag('whisper')} Skipping {len(existing)} already-transcribed video(s).")

    if not to_process:
        console.print(f"{tag('whisper')} All videos already transcribed.")
        return existing

    model_name = config.extract.whisper_model

    console.print(
        f"{tag('whisper')} Transcribing {len(to_process)} video(s) "
        f"with {model_name} on {device} ({compute_type})."
    )

    # Suppress HuggingFace Hub token warning — model is already cached locally
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    from faster_whisper import WhisperModel

    with console.status(f"{tag('whisper')} Loading model {model_name}..."):
        t0 = time.monotonic()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        elapsed = time.monotonic() - t0
    console.print(f"{tag('whisper')} Model loaded in {elapsed:.1f}s.")

    results = list(existing)
    total = len(to_process)
    for i, vf in enumerate(to_process, 1):
        with console.status(f"{tag('whisper')} [{i}/{total}] Transcribing {vf.name}..."):
            t0 = time.monotonic()
            out = transcribe_video(vf, output_dir, model, model_name, initial_prompt)
            elapsed = time.monotonic() - t0
        console.print(f"{tag('whisper')} [{i}/{total}] {vf.name} ({elapsed:.0f}s)")
        results.append(out)

    console.print(f"{tag('whisper')} Transcribed {total} video(s).")
    return results


def transcribe_all_whisperx(
    config: Config,
    device: str,
    compute_type: str,
    initial_prompt: str = "",
    diarize: bool = True,
) -> list[Path]:
    """Transcribe all downloaded videos using WhisperX with optional diarization.

    Models are loaded once and reused across all videos. GPU memory is managed
    between pipeline stages to reduce peak VRAM usage.
    """
    import gc
    import os
    import warnings

    # Suppress tqdm bars and noisy warnings BEFORE importing whisperx/torch,
    # since pyannote and torch fire warnings/progress at import time.
    # NOTE: Python's C-level warning filter uses pattern.match() (anchored at
    # start of string). pyannote's torchcodec warning starts with '\n', so we
    # need (?s) (DOTALL) to let '.' match newlines, otherwise the filter fails.
    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
    warnings.filterwarnings("ignore", message=r"(?s).*libnvrtc.*")
    warnings.filterwarnings("ignore", message=r"(?s).*libtorchcodec.*")
    warnings.filterwarnings("ignore", message=r"(?s).*amdgpu.ids.*")
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # pyannote \\s regex
    warnings.filterwarnings("ignore", message=r"(?s).*weights_only.*")
    warnings.filterwarnings("ignore", message=r"(?s).*upgrade_checkpoint.*")
    warnings.filterwarnings("ignore", message=r"(?s).*Bad things might happen.*")
    warnings.filterwarnings("ignore", message=r"(?s).*TF32.*")  # pyannote ReproducibilityWarning
    logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning_utilities").setLevel(logging.ERROR)
    logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)

    import torch

    # Disable MIOpen (ROCm's cuDNN equivalent) to avoid HIPRTC JIT compilation.
    # MIOpen's runtime compiler can't find headers on NixOS (/nix/store paths).
    # GPU operations still work — just without MIOpen's optimized kernels.
    # Performance impact is negligible for inference workloads.
    if device == "rocm":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    import whisperx

    # whisperx.log_utils.get_logger() auto-calls setup_logging(level="info")
    # on first use, overriding any level we set beforehand. Reconfigure it.
    from whisperx.log_utils import setup_logging as _wxlog_setup
    _wxlog_setup(level="error")

    data_dir = config.data_dir
    videos_dir = data_dir / "videos"
    output_dir = data_dir / "transcripts"

    if not videos_dir.exists():
        console.print(f"{tag('whisperx')} No videos directory found, skipping transcription.")
        return []

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        console.print(f"{tag('whisperx')} No video files found.")
        return []

    # Filter out already-transcribed videos
    to_process = []
    existing = []
    for vf in video_files:
        transcript_path = output_dir / f"{vf.stem}.json"
        if transcript_path.exists():
            existing.append(transcript_path)
        else:
            to_process.append(vf)

    if existing:
        console.print(
            f"{tag('whisperx')} Skipping {len(existing)} already-transcribed video(s)."
        )

    if not to_process:
        console.print(f"{tag('whisperx')} All videos already transcribed.")
        return existing

    model_name = config.extract.whisper_model
    hf_token = os.environ.get("HF_TOKEN", "")
    can_diarize = diarize and bool(hf_token)

    # ROCm: ctranslate2 (used by faster-whisper for ASR) has no ROCm support,
    # so ASR runs on CPU. Alignment + diarization use PyTorch which works on
    # ROCm via the HIP compatibility layer (exposed as "cuda" in torch).
    is_rocm = device == "rocm"
    asr_device = "cpu" if is_rocm else device
    asr_compute = "int8" if is_rocm else compute_type
    torch_device = "cuda" if is_rocm else device  # PyTorch ROCm uses "cuda" API
    batch_size = _WHISPERX_BATCH_SIZE_CPU if asr_device == "cpu" else _WHISPERX_BATCH_SIZE_GPU

    device_desc = "rocm (ASR on CPU, alignment/diarization on GPU)" if is_rocm \
        else f"{device} ({compute_type})"
    console.print(
        f"{tag('whisperx')} Transcribing {len(to_process)} video(s) "
        f"with {model_name} on {device_desc}, "
        f"batch_size={batch_size}"
        f"{' + diarization' if can_diarize else ''}."
    )

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    def _free_gpu():
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- Load models once ---
    with console.status(f"{tag('whisperx')} Loading ASR model {model_name}..."):
        t0 = time.monotonic()
        asr_model = whisperx.load_model(
            model_name,
            asr_device,
            compute_type=asr_compute,
            language="en",
            asr_options={
                "beam_size": 5,
                "initial_prompt": initial_prompt or None,
                "hotwords": initial_prompt or None,
            },
            vad_options={"vad_onset": 0.5, "vad_offset": 0.363},
        )
        elapsed = time.monotonic() - t0
    console.print(f"{tag('whisperx')} ASR model loaded in {elapsed:.1f}s.")

    with console.status(f"{tag('whisperx')} Loading alignment model..."):
        align_model, align_metadata = whisperx.load_align_model(
            language_code="en", device=torch_device
        )
    console.print(f"{tag('whisperx')} Alignment model loaded.")

    diarize_model = None
    if can_diarize:
        with console.status(f"{tag('whisperx')} Loading diarization model..."):
            try:
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    token=hf_token, device=torch_device
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load diarization model. This usually means:\n"
                    "  1. You haven't accepted the pyannote model licenses. Visit:\n"
                    "     - https://hf.co/pyannote/speaker-diarization-3.1\n"
                    "     - https://hf.co/pyannote/segmentation-3.0\n"
                    "     and click 'Agree and access repository' while logged in.\n"
                    "  2. Your HF_TOKEN is missing or invalid. Set it in .env\n"
                    f"\n  Original error: {exc}"
                ) from exc
        console.print(f"{tag('whisperx')} Diarization model loaded.")

    # --- Process videos ---
    results = list(existing)
    total = len(to_process)
    for i, vf in enumerate(to_process, 1):
        with console.status(f"{tag('whisperx')} [{i}/{total}] Transcribing {vf.name}..."):
            t0 = time.monotonic()

            audio = whisperx.load_audio(str(vf))

            # 1. Transcribe with batched inference
            result = asr_model.transcribe(audio, batch_size=batch_size)
            language = result.get("language", "en")

            # 2. Word-level alignment
            result = whisperx.align(
                result["segments"], align_model, align_metadata, audio, torch_device
            )

            # 3. Speaker diarization
            diarized = False
            if diarize_model is not None:
                diarize_segments = diarize_model(
                    audio,
                    min_speakers=_DIARIZE_MIN_SPEAKERS,
                    max_speakers=_DIARIZE_MAX_SPEAKERS,
                )
                result = whisperx.assign_word_speakers(
                    diarize_segments, result, fill_nearest=True
                )
                diarized = True

            # 4. Build and save output
            out = _build_whisperx_output(
                result, vf, output_dir, model_name, language, diarized
            )

            elapsed = time.monotonic() - t0

        console.print(f"{tag('whisperx')} [{i}/{total}] {vf.name} ({elapsed:.0f}s)")
        results.append(out)

    # Cleanup
    del asr_model, align_model, diarize_model
    _free_gpu()

    console.print(f"{tag('whisperx')} Transcribed {total} video(s).")
    return results
