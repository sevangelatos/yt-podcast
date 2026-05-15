#!/usr/bin/env python3
"""
yt-podcast: Translate YouTube video audio with AI voice.

Downloads audio from a YouTube video using yt-dlp and translates it
using Meta's SeamlessM4T v2 — a speech-to-speech translation model.

Audio is split at detected silence boundaries (between words/sentences)
to avoid cutting mid-utterance, while keeping chunks large enough for
the translation model to produce high-quality output.

Requirements:
    uv pip install "seamless_communication @ git+https://github.com/facebookresearch/seamless_communication.git"
    uv pip install sentencepiece
    yt-dlp and ffmpeg must be available on PATH.

Usage:
    python translate.py "https://youtube.com/watch?v=..." -o output.mp3
    python translate.py "https://youtube.com/watch?v=..." --tgt-lang fra
    python translate.py "https://youtube.com/watch?v=..." --duration-factor 1.1
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SAMPLE_RATE = 16_000  # SeamlessM4T expects 16 kHz input
SUPPORTED_FORMATS = ("wav", "mp3", "ogg")

# Target languages supported by SeamlessM4T v2
SUPPORTED_LANGUAGES = {
    "arb": "Arabic",
    "ben": "Bengali",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Chinese (Mandarin)",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "eng": "English",
    "est": "Estonian",
    "fin": "Finnish",
    "fra": "French",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "kan": "Kannada",
    "kor": "Korean",
    "mlt": "Maltese",
    "nld": "Dutch",
    "pes": "Persian (Farsi)",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "spa": "Spanish",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Uzbek",
    "vie": "Vietnamese",
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
        regions.append((run_start * frame_samples, mono.shape[0]))

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
    if total_chunked != total_samples:
        raise ValueError(
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
# SeamlessM4T v2 translation
# ---------------------------------------------------------------------------

def load_m4t_model(device: torch.device, dtype: torch.dtype):
    """Load SeamlessM4T v2 Translator with vocoder."""
    from seamless_communication.inference import Translator

    print("Loading SeamlessM4T v2 model …")

    translator = Translator(
        model_name_or_card="seamlessM4T_v2_large",
        vocoder_name_or_card="vocoder_v2",
        device=device,
        dtype=dtype,
    )

    print("Model loaded.\n")

    return translator


def _get_speaker_idx(translator, tgt_lang: str) -> int:
    """Get the default speaker index for the target language."""
    lang_spkr_map = getattr(translator.vocoder, "lang_spkr_idx_map", None)
    if lang_spkr_map:
        multi_spkr = lang_spkr_map.get("multispkr", {})
        if tgt_lang in multi_spkr and multi_spkr[tgt_lang]:
            return multi_spkr[tgt_lang][0]
    return -1


def _build_voice_list(translator) -> dict[int, dict]:
    """Build a mapping from speaker IDs to their language info.

    Returns dict: {id: {"lang_codes": [str], "lang_names": [str]}}
    Speaker IDs are globally unique (speaker embedding indices from multispkr).
    """
    lang_spkr_map = getattr(translator.vocoder, "lang_spkr_idx_map", None)
    if not lang_spkr_map:
        return {}

    voices: dict[int, dict] = {}

    # Only multispkr entries are speaker embedding indices
    multi_spkr = lang_spkr_map.get("multispkr", {})
    for lang_code, spkr_indices in multi_spkr.items():
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
        for spkr_idx in spkr_indices:
            if spkr_idx not in voices:
                voices[spkr_idx] = {"lang_codes": [], "lang_names": []}
            if lang_code not in voices[spkr_idx]["lang_codes"]:
                voices[spkr_idx]["lang_codes"].append(lang_code)
                voices[spkr_idx]["lang_names"].append(lang_name)

    return voices


def _get_all_voice_ids(translator) -> set[int]:
    """Collect all valid speaker indices into a set for O(1) lookup."""
    lang_spkr_map = getattr(translator.vocoder, "lang_spkr_idx_map", None)
    if not lang_spkr_map:
        return set()
    ids: set[int] = set()
    for spkr_indices in lang_spkr_map.get("multispkr", {}).values():
        ids.update(spkr_indices)
    return ids


def _resolve_voice(translator, voice_id: int) -> int:
    """Resolve a voice ID to a speaker index. Returns -1 if not found."""
    if voice_id in _get_all_voice_ids(translator):
        return voice_id
    return -1


def translate_chunks(
    chunks: list[torch.Tensor],
    translator,
    tgt_lang: str,
    duration_factor: float,
    spkr_idx: int,
) -> tuple[list[torch.Tensor], int]:
    """Translate audio chunks using SeamlessM4T v2 S2ST.

    Returns (list_of_waveform_tensors, output_sample_rate).
    """
    translated: list[torch.Tensor] = []
    total_src = 0.0
    total_tgt = 0.0
    output_sr = INPUT_SAMPLE_RATE

    for i, chunk in enumerate(chunks, 1):
        src_dur = chunk.shape[-1] / INPUT_SAMPLE_RATE
        total_src += src_dur

        print(f"  Translating chunk {i}/{len(chunks)} (source: {src_dur:.1f}s) … ", end="", flush=True)

        # Translate chunk — pass raw audio tensor, model handles fbank internally
        _, speech_output = translator.predict(
            input=chunk.squeeze(0),  # (samples,) — predict expects 1D or 2D tensor
            task_str="S2ST",
            tgt_lang=tgt_lang,
            sample_rate=INPUT_SAMPLE_RATE,
            duration_factor=duration_factor,
            spkr=spkr_idx,
        )

        if speech_output is None:
            print("warning: no speech output for this chunk, skipping")
            continue

        wav = speech_output.audio_wavs[0][0].flatten().to(torch.float32).cpu()
        output_sr = speech_output.sample_rate
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
        description="Translate YouTube video audio with AI voice (SeamlessM4T v2).",
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
        "--voice",
        type=int,
        default=None,
        help="Speaker ID for translated output (globally unique). Use --list-voices to see available IDs.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available speaker IDs (globally unique) and exit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.url and not args.list_voices:
        parser.error("the following arguments are required: url")

    if args.min_chunk >= args.max_chunk:
        parser.error("--min-chunk must be less than --max-chunk")

    if not args.list_voices:
        out = Path(args.output)
        fmt = out.suffix.lstrip(".").lower()
        if fmt not in SUPPORTED_FORMATS:
            parser.error(f"Unsupported format '.{fmt}'. Use one of: {', '.join(SUPPORTED_FORMATS)}")
        _check_prerequisites()
    else:
        out = Path(args.output)
        fmt = out.suffix.lstrip(".").lower()

    # Device ----------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    print(f"Using device: {device} ({dtype})")

    # Load model ------------------------------------------------------------
    translator = load_m4t_model(device, dtype)

    # List voices -----------------------------------------------------------
    if args.list_voices:
        voice_map = _build_voice_list(translator)
        if not voice_map:
            print("No voices available.")
            return
        current_lang = args.tgt_lang
        lang_spkr_map = getattr(translator.vocoder, "lang_spkr_idx_map", None)
        default_id = -1
        if lang_spkr_map:
            multi_spkr = lang_spkr_map.get("multispkr", {})
            if current_lang in multi_spkr and multi_spkr[current_lang]:
                default_id = multi_spkr[current_lang][0]
        print(f"\n{'ID':<5} {'Languages':<40} Default")
        print(f"{'-'*5} {'-'*40} {'-'*7}")
        for voice_id in sorted(voice_map):
            info = voice_map[voice_id]
            lang_names = ", ".join(info["lang_names"])
            marker = " <-- target" if voice_id == default_id else ""
            print(f"  {voice_id:<3} {lang_names:<40}{marker}")
        return

    # Resolve voice ---------------------------------------------------------
    if args.voice is not None:
        spkr_idx = _resolve_voice(translator, args.voice)
        if spkr_idx < 0:
            print(f"Error: voice ID {args.voice} not found.", file=sys.stderr)
            print("Use --list-voices to see available speaker IDs.", file=sys.stderr)
            sys.exit(1)
        print(f"Using speaker ID {args.voice}")
    else:
        spkr_idx = _get_speaker_idx(translator, args.tgt_lang)

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
        voice_info = f" (voice ID: {args.voice})" if args.voice is not None else ""
        print(f"Translating to '{args.tgt_lang}'{voice_info}{df_str} …")
        translated, output_sr = translate_chunks(
            chunks,
            translator,
            tgt_lang=args.tgt_lang,
            duration_factor=args.duration_factor,
            spkr_idx=spkr_idx,
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
