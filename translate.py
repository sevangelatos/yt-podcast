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
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

import shutil

from common import (
    DEFAULT_MAX_CHUNK,
    DEFAULT_MIN_CHUNK,
    DEFAULT_SILENCE_THRESH,
    INPUT_SAMPLE_RATE,
    SUPPORTED_FORMATS,
    chunk_audio_at_silences,
    check_prerequisites,
    convert_audio,
    download_audio,
    find_silence_regions,
    resolve_device,
)

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
        check_prerequisites()
    else:
        out = Path(args.output)
        fmt = out.suffix.lstrip(".").lower()

    device, dtype = resolve_device(args.device)

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
