#!/usr/bin/env python3
"""
yt-podcast: Translate YouTube video audio to English with an AI voice.

Downloads audio from a YouTube video using yt-dlp and translates it to
English using Meta's SeamlessM4T v2 — a direct speech-to-speech translation
model (no intermediate text step). Supports multiple output voices.

Audio is split at detected silence boundaries (between words/sentences)
to avoid cutting mid-utterance, while keeping chunks large enough for
the translation model to produce high-quality output.

Requirements:
    pip install -r requirements.txt
    yt-dlp and ffmpeg must be available on PATH.

Usage:
    python translate.py "https://youtube.com/watch?v=..." -o output.wav
    python translate.py "https://youtube.com/watch?v=..." -v ben -o output.wav
    python translate.py --list-voices
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

# ---------------------------------------------------------------------------
# Voice presets — each maps to a SeamlessM4T v2 vocoder speaker embedding.
# The vocoder supports IDs 0-199; these are curated presets with distinct
# characteristics.  Use --speaker-id N to try any ID directly.
# ---------------------------------------------------------------------------
VOICES: dict[str, int] = {
    "aria": 0,
    "ben": 1,
    "caleb": 2,
    "diana": 3,
    "elena": 4,
    "felix": 5,
    "grace": 6,
    "henry": 7,
    "iris": 8,
    "james": 9,
}

MODEL_ID = "facebook/seamless-m4t-v2-large"
SAMPLE_RATE = 16_000  # SeamlessM4T v2 input and vocoder output sample rate
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
    """Find silence regions in the waveform.

    Returns a list of (start_sample, end_sample) tuples for each region
    where the RMS energy stays below *thresh_dbfs* (relative to peak
    amplitude) for at least *min_silence_ms* milliseconds.
    """
    mono = waveform.squeeze(0)  # (samples,)
    frame_samples = FRAME_MS * SAMPLE_RATE // 1000  # 320 samples at 16 kHz

    # Compute per-frame RMS energy using unfold
    # Trim tail samples that don't fill a complete frame
    n_frames = mono.shape[0] // frame_samples
    trimmed = mono[: n_frames * frame_samples]
    frames = trimmed.view(n_frames, frame_samples)
    rms = frames.float().pow(2).mean(dim=1).sqrt()

    # Convert threshold from dBFS (relative to peak) to linear amplitude
    peak = mono.abs().max().float()
    if peak < 1e-8:
        # Entire waveform is silence
        return [(0, mono.shape[0])]
    thresh_linear = peak * (10.0 ** (thresh_dbfs / 20.0))

    # Find contiguous runs of quiet frames
    is_quiet = rms < thresh_linear
    min_quiet_frames = max(1, min_silence_ms * SAMPLE_RATE // (1000 * frame_samples))

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

    # Handle trailing silence
    if run_start is not None and (n_frames - run_start) >= min_quiet_frames:
        regions.append((run_start * frame_samples, n_frames * frame_samples))

    return regions


def chunk_audio_at_silences(
    waveform: torch.Tensor,
    silence_regions: list[tuple[int, int]],
    min_chunk_s: int = DEFAULT_MIN_CHUNK,
    max_chunk_s: int = DEFAULT_MAX_CHUNK,
) -> list[torch.Tensor]:
    """Split waveform at silence boundaries, respecting min/max chunk sizes.

    Greedy algorithm:
    - Accumulate audio from the current chunk start.
    - Once past *min_chunk_s*, split at the midpoint of the next silence region.
    - If no silence is found by *max_chunk_s*, force-split there.
    - Never produce chunks shorter than 0.5 s.
    """
    total_samples = waveform.shape[-1]
    min_samples = min_chunk_s * SAMPLE_RATE
    max_samples = max_chunk_s * SAMPLE_RATE

    chunks: list[torch.Tensor] = []
    chunk_start = 0

    # Index into silence_regions — advance as we move through the waveform
    si = 0

    while chunk_start < total_samples:
        remaining = total_samples - chunk_start

        # If whatever is left is shorter than max_chunk, take it all
        if remaining <= max_samples:
            if remaining >= SAMPLE_RATE // 2:  # at least 0.5 s
                chunks.append(waveform[..., chunk_start:total_samples])
            break

        # Advance silence index past chunk_start
        while si < len(silence_regions) and silence_regions[si][1] <= chunk_start:
            si += 1

        # Look for the first silence region whose midpoint is past min_chunk
        split_at: int | None = None
        for j in range(si, len(silence_regions)):
            s_start, s_end = silence_regions[j]
            mid = (s_start + s_end) // 2

            # Silence must be within our window
            if mid <= chunk_start + min_samples:
                continue
            if mid > chunk_start + max_samples:
                break

            split_at = mid
            break

        if split_at is None:
            # No suitable silence found — force-split at max_chunk
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
    """Verify that yt-dlp and ffmpeg are installed."""
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
        "--postprocessor-args", f"ffmpeg:-ar {SAMPLE_RATE} -ac 1",
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


def translate_chunks(
    chunks: list[torch.Tensor],
    processor: AutoProcessor,
    model: SeamlessM4Tv2ForSpeechToSpeech,
    tgt_lang: str,
    speaker_id: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Run S2ST on each chunk and return a list of translated waveforms."""
    translated: list[torch.Tensor] = []
    total_src = 0.0
    total_tgt = 0.0

    for i, chunk in enumerate(chunks, 1):
        src_dur = chunk.shape[-1] / SAMPLE_RATE
        total_src += src_dur

        print(f"  Translating chunk {i}/{len(chunks)} (source: {src_dur:.1f}s) … ", end="", flush=True)
        inputs = processor(
            audio=chunk.squeeze(0).numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            waveform, _ = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id,
            )

        wav = waveform.cpu().squeeze()
        tgt_dur = wav.shape[-1] / SAMPLE_RATE
        total_tgt += tgt_dur
        print(f"translated: {tgt_dur:.1f}s")

        translated.append(wav)

    print(f"\n  Total source: {total_src:.1f}s → total translated: {total_tgt:.1f}s")

    return translated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate YouTube video audio to English using an AI voice.",
        epilog="Use --list-voices to see available voice presets.",
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output",
        default="translated.mp3",
        help="Output file path; format from extension: wav, mp3, ogg (default: translated.mp3)",
    )
    parser.add_argument(
        "-v", "--voice",
        default="aria",
        choices=sorted(VOICES.keys()),
        help="Named voice preset (default: aria)",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="Raw vocoder speaker ID 0-199 (overrides --voice)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="eng",
        help="Target language code (default: eng)",
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
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Print available voices and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --list-voices ---------------------------------------------------------
    if args.list_voices:
        print("Available voice presets:")
        for name, sid in sorted(VOICES.items()):
            print(f"  {name:12s}  speaker_id={sid}")
        print("\nPass --speaker-id N (0-199) to try any vocoder speaker directly.")
        sys.exit(0)

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
    print(f"Using device: {device}")

    # Speaker ID ------------------------------------------------------------
    speaker_id = args.speaker_id if args.speaker_id is not None else VOICES[args.voice]

    # Load model ------------------------------------------------------------
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    from huggingface_hub import try_to_load_from_cache
    cached = isinstance(try_to_load_from_cache(MODEL_ID, "config.json"), str)

    if cached:
        print(f"Loading model {MODEL_ID} ({dtype}) from cache …")
    else:
        print(f"Downloading model {MODEL_ID} (~9 GB, {dtype}) …")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    # Output paths ----------------------------------------------------------
    original_path = out.with_stem(out.stem + ".original")

    # Download audio --------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="yt-podcast-") as tmp:
        work_dir = Path(tmp)
        audio_path = download_audio(args.url, work_dir)

        # Load audio --------------------------------------------------------
        waveform, _ = torchaudio.load(str(audio_path))
        total_seconds = waveform.shape[-1] / SAMPLE_RATE
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
        chunk_durs = [c.shape[-1] / SAMPLE_RATE for c in chunks]
        print(
            f"  Split into {len(chunks)} chunk(s): "
            f"min={min(chunk_durs):.1f}s, max={max(chunk_durs):.1f}s, "
            f"avg={sum(chunk_durs)/len(chunk_durs):.1f}s\n"
        )

        # Translate ---------------------------------------------------------
        print(f"Translating to '{args.tgt_lang}' with voice '{args.voice}' (speaker_id={speaker_id}) …")
        translated = translate_chunks(
            chunks, processor, model,
            tgt_lang=args.tgt_lang,
            speaker_id=speaker_id,
            device=device,
        )

        # Concatenate & save ------------------------------------------------
        full_wav = torch.cat(translated, dim=-1).unsqueeze(0).to(torch.float32)

        if fmt == "wav":
            torchaudio.save(str(out), full_wav, SAMPLE_RATE)
        else:
            wav_tmp = work_dir / "translated.wav"
            torchaudio.save(str(wav_tmp), full_wav, SAMPLE_RATE)
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
