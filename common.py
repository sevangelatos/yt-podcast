#!/usr/bin/env python3
"""
Common utilities shared by translate.py and translate_expressive.py.

Provides silence detection, audio chunking, audio download/conversion helpers,
and device resolution used by both translation scripts.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SAMPLE_RATE = 16_000  # SeamlessM4T / SeamlessExpressive expects 16 kHz input
SUPPORTED_FORMATS = ("wav", "mp3", "ogg")

# Chunking defaults
DEFAULT_MIN_CHUNK = 15   # seconds — don't split before this
DEFAULT_MAX_CHUNK = 30   # seconds — force-split if no silence found
DEFAULT_SILENCE_THRESH = -25  # dBFS
MIN_SILENCE_DURATION_MS = 150  # minimum silence length to count as a split point
FRAME_MS = 20  # RMS energy analysis frame size in milliseconds


# ---------------------------------------------------------------------------
# Silence detection & chunking
# ---------------------------------------------------------------------------

def find_silence_regions(
    waveform: torch.Tensor,
    thresh_dbfs: float = DEFAULT_SILENCE_THRESH,
    min_silence_ms: int = MIN_SILENCE_DURATION_MS,
) -> list[tuple[int, int]]:
    mono = waveform.squeeze(0)
    frame_samples = FRAME_MS * INPUT_SAMPLE_RATE // 1000

    n_frames = mono.shape[0] // frame_samples
    trimmed = mono[: n_frames * frame_samples]
    frames = trimmed.view(n_frames, frame_samples)
    rms = frames.float().pow(2).mean(dim=1).sqrt()

    peak = mono.abs().max().float()
    if peak < 1e-8:
        return [(0, mono.shape[0])]
    thresh_linear = peak * (10.0 ** (thresh_dbfs / 20.0))

    is_quiet = rms < thresh_linear
    min_quiet_frames = max(1, min_silence_ms * INPUT_SAMPLE_RATE // (1000 * frame_samples))

    regions: list[tuple[int, int]] = []
    run_start: int | None = None
    for i in range(n_frames):
        if is_quiet[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_quiet_frames:
                regions.append((run_start * frame_samples, i * frame_samples))
            run_start = None

    if run_start is not None and (n_frames - run_start) >= min_quiet_frames:
        regions.append((run_start * frame_samples, mono.shape[0]))

    return regions


def chunk_audio_at_silences(
    waveform: torch.Tensor,
    silence_regions: list[tuple[int, int]],
    min_chunk_s: int = DEFAULT_MIN_CHUNK,
    max_chunk_s: int = DEFAULT_MAX_CHUNK,
) -> list[torch.Tensor]:
    total_samples = waveform.shape[-1]
    min_samples = min_chunk_s * INPUT_SAMPLE_RATE
    max_samples = max_chunk_s * INPUT_SAMPLE_RATE

    chunks: list[torch.Tensor] = []
    chunk_start = 0
    si = 0

    while chunk_start < total_samples:
        remaining = total_samples - chunk_start

        if remaining <= max_samples:
            if remaining >= INPUT_SAMPLE_RATE // 2:
                chunks.append(waveform[..., chunk_start:total_samples])
            break

        while si < len(silence_regions) and silence_regions[si][1] <= chunk_start:
            si += 1

        split_at: int | None = None
        for j in range(si, len(silence_regions)):
            s_start, s_end = silence_regions[j]
            mid = (s_start + s_end) // 2
            if mid <= chunk_start + min_samples:
                continue
            if mid > chunk_start + max_samples:
                break
            split_at = mid
            break

        if split_at is None:
            split_at = chunk_start + max_samples

        chunks.append(waveform[..., chunk_start:split_at])
        chunk_start = split_at

    total_chunked = sum(c.shape[-1] for c in chunks)
    if total_chunked != total_samples:
        raise ValueError(
            f"Audio loss detected: {total_samples} samples in, {total_chunked} samples chunked "
            f"(delta={total_samples - total_chunked})"
        )

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    for prog in ("yt-dlp", "ffmpeg"):
        if shutil.which(prog) is None:
            print(f"Error: '{prog}' not found on PATH. Please install it.", file=sys.stderr)
            sys.exit(1)


def download_audio(url: str, work_dir: Path) -> Path:
    output_path = work_dir / "source_audio.wav"
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {INPUT_SAMPLE_RATE} -ac 1",
        "-o", str(work_dir / "source_audio.%(ext)s"),
        url,
    ]
    print(f"Downloading audio from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    if not output_path.exists():
        print("Error: expected audio file not found after download.", file=sys.stderr)
        sys.exit(1)

    print("Audio downloaded.")
    return output_path


def convert_audio(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg conversion failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def resolve_device(device_arg: str | None) -> tuple[torch.device, torch.dtype]:
    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"Using device: {device} ({dtype})")
    return device, dtype
