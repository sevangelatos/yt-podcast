#!/usr/bin/env python3
"""
yt-podcast: Translate YouTube video audio to English with an AI voice.

Downloads audio from a YouTube video using yt-dlp and translates it to
English using Meta's SeamlessM4T v2 — a direct speech-to-speech translation
model (no intermediate text step). Supports multiple output voices.

Requirements:
    pip install -r requirements.txt
    yt-dlp and ffmpeg must be available on PATH.

Usage:
    python translate.py "https://youtube.com/watch?v=..." -o output.wav
    python translate.py "https://youtube.com/watch?v=..." -v ben -o output.mp3
    python translate.py --list-voices
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
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
MODEL_SAMPLE_RATE = 16_000  # SeamlessM4T v2 expects 16 kHz input
OUTPUT_SAMPLE_RATE = 16_000  # vocoder output sample rate
MAX_CHUNK_SECONDS = 20  # safe upper bound for one forward pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_prerequisites() -> None:
    """Verify that yt-dlp and ffmpeg are installed."""
    for prog in ("yt-dlp", "ffmpeg"):
        if shutil.which(prog) is None:
            print(f"Error: '{prog}' not found on PATH. Please install it.", file=sys.stderr)
            sys.exit(1)


def download_audio(url: str, work_dir: str) -> str:
    """Download audio from *url* as 16 kHz mono WAV using yt-dlp."""
    output_template = os.path.join(work_dir, "source_audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {MODEL_SAMPLE_RATE} -ac 1",
        "-o", output_template,
        url,
    ]
    print(f"Downloading audio from: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    output_path = os.path.join(work_dir, "source_audio.wav")
    if not os.path.exists(output_path):
        print("Error: expected audio file not found after download.", file=sys.stderr)
        sys.exit(1)

    print("Audio downloaded.")
    return output_path


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio and ensure it is 16 kHz mono float32."""
    waveform, sr = torchaudio.load(path)
    if sr != MODEL_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, MODEL_SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, MODEL_SAMPLE_RATE


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_seconds: int = MAX_CHUNK_SECONDS,
) -> list[torch.Tensor]:
    """Split waveform into fixed-duration chunks, dropping tail < 0.5 s."""
    chunk_samples = chunk_seconds * sample_rate
    total = waveform.shape[-1]
    chunks: list[torch.Tensor] = []
    for start in range(0, total, chunk_samples):
        end = min(start + chunk_samples, total)
        chunk = waveform[..., start:end]
        if chunk.shape[-1] >= sample_rate // 2:  # at least 0.5 s
            chunks.append(chunk)
    return chunks


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
    for i, chunk in enumerate(chunks, 1):
        print(f"  Translating chunk {i}/{len(chunks)} …")
        audio_np = chunk.squeeze(0).numpy()
        inputs = processor(
            audio=audio_np,
            sampling_rate=MODEL_SAMPLE_RATE,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            waveform, waveform_lengths = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id,
            )

        # waveform shape: (batch, frames, 1)
        wav = waveform.cpu().squeeze()
        translated.append(wav)

    return translated


def save_audio(waveform: torch.Tensor, path: str) -> None:
    """Save waveform to *path* (format inferred from extension)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # torchaudio requires float32 (or integer) for saving
    waveform = waveform.to(torch.float32)
    torchaudio.save(path, waveform, OUTPUT_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate YouTube video audio to English using an AI voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available voices:\n"
            + "".join(f"  {n:12s} (speaker_id={sid})\n" for n, sid in sorted(VOICES.items()))
            + "\nUse --speaker-id N (0-199) to try any vocoder speaker directly."
        ),
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output",
        default="translated.wav",
        help="Output file path. Supports .wav and .flac (default: translated.wav)",
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
        "--chunk-seconds",
        type=int,
        default=MAX_CHUNK_SECONDS,
        help=f"Max seconds per chunk (default: {MAX_CHUNK_SECONDS})",
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
    print(f"Loading model {MODEL_ID} ({dtype}) …")
    print("(First run downloads ~9 GB of weights.)")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    # Derive path for the original audio copy
    out = Path(args.output)
    original_path = out.with_stem(out.stem + ".original")

    # Download audio --------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="yt-podcast-") as work_dir:
        audio_path = download_audio(args.url, work_dir)

        # Keep a copy of the original audio ---------------------------------
        shutil.copy2(audio_path, original_path)
        print(f"Original audio saved to: {original_path}")

        # Load & chunk ------------------------------------------------------
        print("Preparing audio …")
        waveform, sr = load_audio(audio_path)
        total_seconds = waveform.shape[-1] / sr
        chunks = chunk_audio(waveform, sr, args.chunk_seconds)
        print(f"  Total duration : {total_seconds:.1f}s")
        print(f"  Chunks         : {len(chunks)} × ≤{args.chunk_seconds}s\n")

        # Translate ---------------------------------------------------------
        print(f"Translating to '{args.tgt_lang}' with voice '{args.voice}' (speaker_id={speaker_id}) …")
        translated = translate_chunks(
            chunks, processor, model,
            tgt_lang=args.tgt_lang,
            speaker_id=speaker_id,
            device=device,
        )

    # Concatenate & save ----------------------------------------------------
    full_wav = torch.cat(translated, dim=-1)
    save_audio(full_wav, args.output)
    print(f"\nDone — saved to {args.output}")
    print(f"Original audio: {original_path}")


if __name__ == "__main__":
    main()
