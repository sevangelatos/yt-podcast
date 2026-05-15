#!/usr/bin/env python3
"""
yt-podcast: Translate YouTube video audio with expressive AI voice.

Downloads audio from a YouTube video using yt-dlp and translates it
using Meta's SeamlessExpressive — a speech-to-speech translation model
that preserves the speaker's prosody, emotion, and pacing.

Audio is split at detected silence boundaries (between words/sentences)
to avoid cutting mid-utterance, while keeping chunks large enough for
the translation model to produce high-quality output.

Requirements:
    pip install seamless_communication
    yt-dlp and ffmpeg must be available on PATH.
    SeamlessExpressive weights require Meta approval — see README.

Usage:
    python translate.py "https://youtube.com/watch?v=..." -o output.mp3
    python translate.py "https://youtube.com/watch?v=..." --tgt-lang fra
    python translate.py "https://youtube.com/watch?v=..." --duration-factor 1.1
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("yt-podcast")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SAMPLE_RATE = 16_000  # SeamlessExpressive expects 16 kHz input
SUPPORTED_FORMATS = ("wav", "mp3", "ogg")

# Target languages supported by SeamlessExpressive
SUPPORTED_LANGUAGES = {
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "cmn": "Mandarin (experimental)",
    "ita": "Italian (experimental)",
}

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
    """Find silence regions in the waveform.

    Returns a list of (start_sample, end_sample) tuples for each region
    where the RMS energy stays below *thresh_dbfs* (relative to peak
    amplitude) for at least *min_silence_ms* milliseconds.
    """
    mono = waveform.squeeze(0)  # (samples,)
    frame_samples = FRAME_MS * INPUT_SAMPLE_RATE // 1000  # 320 samples at 16 kHz

    # Compute per-frame RMS energy
    n_frames = mono.shape[0] // frame_samples
    trimmed = mono[: n_frames * frame_samples]
    frames = trimmed.view(n_frames, frame_samples)
    rms = frames.float().pow(2).mean(dim=1).sqrt()

    # Convert threshold from dBFS (relative to peak) to linear amplitude
    peak = mono.abs().max().float()
    if peak < 1e-8:
        return [(0, mono.shape[0])]
    thresh_linear = peak * (10.0 ** (thresh_dbfs / 20.0))

    # Find contiguous runs of quiet frames
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
        regions.append((run_start * frame_samples, n_frames * frame_samples))

    return regions


def chunk_audio_at_silences(
    waveform: torch.Tensor,
    silence_regions: list[tuple[int, int]],
    min_chunk_s: int = DEFAULT_MIN_CHUNK,
    max_chunk_s: int = DEFAULT_MAX_CHUNK,
) -> list[torch.Tensor]:
    """Split waveform at silence boundaries, respecting min/max chunk sizes."""
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

    # Validate: no audio was lost
    total_chunked = sum(c.shape[-1] for c in chunks)
    assert total_chunked == total_samples, (
        f"Audio loss detected: {total_samples} samples in, {total_chunked} samples chunked "
        f"(delta={total_samples - total_chunked})"
    )

    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_prerequisites() -> None:
    for prog in ("yt-dlp", "ffmpeg"):
        if shutil.which(prog) is None:
            print(f"Error: '{prog}' not found on PATH. Please install it.", file=sys.stderr)
            sys.exit(1)


def download_audio(url: str, work_dir: Path) -> Path:
    """Download audio from *url* as 16 kHz mono WAV using yt-dlp."""
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
    """Convert audio from *src* to *dst* using ffmpeg (format from extension)."""
    cmd = ["ffmpeg", "-y", "-i", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg conversion failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# SeamlessExpressive translation
# ---------------------------------------------------------------------------

def load_expressive_model(device: torch.device, dtype: torch.dtype, gated_model_dir: Path | None = None):
    """Load SeamlessExpressive Translator + PretsselGenerator.

    Returns (translator, pretssel_generator, prosody_fbank_extractor, gcmvn_mean, gcmvn_std).
    """
    from fairseq2.data.audio import WaveformToFbankConverter

    from seamless_communication.cli.expressivity.predict.pretssel_generator import (
        PretsselGenerator,
    )
    from seamless_communication.inference import Translator
    from seamless_communication.models.unity import (
        load_gcmvn_stats,
        load_unity_unit_tokenizer,
    )
    from seamless_communication.store import add_gated_assets

    model_name = "seamless_expressivity"
    vocoder_name = "vocoder_pretssel"

    if gated_model_dir:
        add_gated_assets(gated_model_dir)

    print("Loading SeamlessExpressive models …")

    unit_tokenizer = load_unity_unit_tokenizer(model_name)

    translator = Translator(
        model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )

    pretssel_generator = PretsselGenerator(
        vocoder_name,
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype,
    )

    # Fbank extractor for prosody input only (no standardization — GCMVN is applied instead).
    # The Translator handles its own fbank extraction internally (with standardize=True)
    # when we pass raw audio tensors as input.
    prosody_fbank_extractor = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )

    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    print("Models loaded.\n")

    return translator, pretssel_generator, prosody_fbank_extractor, gcmvn_mean, gcmvn_std


def _prepare_prosody_input(
    chunk: torch.Tensor,
    prosody_fbank_extractor,
    gcmvn_mean: torch.Tensor,
    gcmvn_std: torch.Tensor,
    device: torch.device,
):
    """Prepare GCMVN-normalized fbank features for the prosody encoder.

    The main encoder input is handled by Translator.predict() internally
    when we pass raw audio tensors. This function only produces the
    GCMVN-normalized prosody input needed by both predict() and the
    PretsselGenerator.

    Returns src_gcmvn as a SequenceData dict.
    """
    from fairseq2.data import SequenceData

    # chunk shape: (1, samples) → (samples, 1) for fbank extractor (channel_last=True)
    wav = chunk.squeeze(0).unsqueeze(1).to(device)

    data = prosody_fbank_extractor({"waveform": wav, "sample_rate": INPUT_SAMPLE_RATE})
    fbank = data["fbank"]

    # GCMVN normalization for prosody encoder
    gcmvn_fbank = fbank.subtract(gcmvn_mean).divide(gcmvn_std)

    src_gcmvn = SequenceData(
        seqs=gcmvn_fbank.unsqueeze(0),
        seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
        is_ragged=False,
    )

    return src_gcmvn


def translate_chunks(
    chunks: list[torch.Tensor],
    translator,
    pretssel_generator,
    prosody_fbank_extractor,
    gcmvn_mean: torch.Tensor,
    gcmvn_std: torch.Tensor,
    tgt_lang: str,
    duration_factor: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    """Translate audio chunks using SeamlessExpressive.

    Returns (list_of_waveform_tensors, output_sample_rate).
    """
    translated: list[torch.Tensor] = []
    total_src = 0.0
    total_tgt = 0.0
    output_sr = INPUT_SAMPLE_RATE  # updated from pretssel output

    for i, chunk in enumerate(chunks, 1):
        src_dur = chunk.shape[-1] / INPUT_SAMPLE_RATE
        total_src += src_dur

        print(f"  Translating chunk {i}/{len(chunks)} (source: {src_dur:.1f}s) … ", end="", flush=True)

        # Prepare GCMVN-normalized prosody input for the prosody encoder.
        # The main encoder fbank input is computed internally by Translator.predict()
        # when we pass the raw audio tensor.
        src_gcmvn = _prepare_prosody_input(
            chunk, prosody_fbank_extractor, gcmvn_mean, gcmvn_std, device
        )

        # Pass raw audio tensor — Translator converts to fbank internally
        # with standardize=True, matching its training config.
        text_output, unit_output = translator.predict(
            chunk.squeeze(0),  # (samples,) — predict expects 1D or 2D tensor
            "s2st",
            tgt_lang,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,
        )

        assert unit_output is not None
        speech_output = pretssel_generator.predict(
            unit_output.units,
            tgt_lang=tgt_lang,
            prosody_encoder_input=src_gcmvn,
        )

        output_sr = speech_output.sample_rate
        wav = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
        tgt_dur = wav.shape[-1] / output_sr
        total_tgt += tgt_dur
        print(f"translated: {tgt_dur:.1f}s")

        translated.append(wav)

    print(f"\n  Total source: {total_src:.1f}s → total translated: {total_tgt:.1f}s")

    return translated, output_sr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    lang_list = ", ".join(f"{k} ({v})" for k, v in SUPPORTED_LANGUAGES.items())
    parser = argparse.ArgumentParser(
        description="Translate YouTube video audio with expressive AI voice (SeamlessExpressive).",
        epilog=f"Supported target languages: {lang_list}",
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output",
        default="translated.mp3",
        help="Output file path; format from extension: wav, mp3, ogg (default: translated.mp3)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="eng",
        choices=sorted(SUPPORTED_LANGUAGES.keys()),
        help="Target language code (default: eng)",
    )
    parser.add_argument(
        "--duration-factor",
        type=float,
        default=1.0,
        help="Duration factor to tune speech rate; >1 = slower, <1 = faster (default: 1.0)",
    )
    parser.add_argument(
        "--gated-model-dir",
        type=Path,
        default=None,
        help="Path to locally downloaded SeamlessExpressive model directory (if not using HF Hub)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: cuda, cpu, mps (auto-detected if omitted)",
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=DEFAULT_MIN_CHUNK,
        help=f"Minimum chunk duration in seconds (default: {DEFAULT_MIN_CHUNK})",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=DEFAULT_MAX_CHUNK,
        help=f"Maximum chunk duration in seconds — force-split if no silence found (default: {DEFAULT_MAX_CHUNK})",
    )
    parser.add_argument(
        "--silence-thresh",
        type=float,
        default=DEFAULT_SILENCE_THRESH,
        help=f"Silence threshold in dBFS (default: {DEFAULT_SILENCE_THRESH})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.url:
        parser.error("the following arguments are required: url")

    if args.min_chunk >= args.max_chunk:
        parser.error("--min-chunk must be less than --max-chunk")

    out = Path(args.output)
    fmt = out.suffix.lstrip(".").lower()
    if fmt not in SUPPORTED_FORMATS:
        parser.error(f"Unsupported format '.{fmt}'. Use one of: {', '.join(SUPPORTED_FORMATS)}")

    _check_prerequisites()

    # Device ----------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device} ({dtype})")

    # Load model ------------------------------------------------------------
    translator, pretssel_generator, prosody_fbank_extractor, gcmvn_mean, gcmvn_std = (
        load_expressive_model(device, dtype, gated_model_dir=args.gated_model_dir)
    )

    # Output paths ----------------------------------------------------------
    original_path = out.with_stem(out.stem + ".original")

    # Download audio --------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="yt-podcast-") as tmp:
        work_dir = Path(tmp)
        audio_path = download_audio(args.url, work_dir)

        # Load audio --------------------------------------------------------
        waveform, _ = torchaudio.load(str(audio_path))
        total_seconds = waveform.shape[-1] / INPUT_SAMPLE_RATE
        print(f"Audio duration: {total_seconds:.1f}s")

        # Detect silence & chunk --------------------------------------------
        print(f"Detecting silence regions (threshold: {args.silence_thresh} dBFS) …")
        silence_regions = find_silence_regions(waveform, thresh_dbfs=args.silence_thresh)
        print(f"  Found {len(silence_regions)} silence region(s)")

        chunks = chunk_audio_at_silences(
            waveform, silence_regions,
            min_chunk_s=args.min_chunk,
            max_chunk_s=args.max_chunk,
        )
        chunk_durs = [c.shape[-1] / INPUT_SAMPLE_RATE for c in chunks]
        print(
            f"  Split into {len(chunks)} chunk(s): "
            f"min={min(chunk_durs):.1f}s, max={max(chunk_durs):.1f}s, "
            f"avg={sum(chunk_durs)/len(chunk_durs):.1f}s\n"
        )

        # Translate ---------------------------------------------------------
        df_str = f", duration_factor={args.duration_factor}" if args.duration_factor != 1.0 else ""
        print(f"Translating to '{args.tgt_lang}'{df_str} …")
        translated, output_sr = translate_chunks(
            chunks,
            translator, pretssel_generator, prosody_fbank_extractor,
            gcmvn_mean, gcmvn_std,
            tgt_lang=args.tgt_lang,
            duration_factor=args.duration_factor,
            device=device,
        )

        # Concatenate & save ------------------------------------------------
        full_wav = torch.cat(translated, dim=-1).unsqueeze(0)

        if fmt == "wav":
            torchaudio.save(str(out), full_wav, output_sr)
        else:
            wav_tmp = work_dir / "translated.wav"
            torchaudio.save(str(wav_tmp), full_wav, output_sr)
            convert_audio(wav_tmp, out)

        # Save original audio in target format
        if fmt == "wav":
            shutil.copy2(audio_path, original_path)
        else:
            convert_audio(audio_path, original_path)

    print(f"\nDone — saved to {out}")
    print(f"Original audio: {original_path}")


if __name__ == "__main__":
    main()
