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
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_SAMPLE_RATE = 24_000  # Vocoder outputs 24 kHz (pretssel)

# Target languages supported by SeamlessExpressive
SUPPORTED_LANGUAGES = {
    "eng": "English",
    "fra": "French",
    "deu": "German",
    "spa": "Spanish",
    "cmn": "Mandarin (experimental)",
    "ita": "Italian (experimental)",
}


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
    beam_size: int,
    len_penalty: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], int]:
    """Translate audio chunks using SeamlessExpressive.

    Returns (list_of_waveform_tensors, output_sample_rate).
    """
    from seamless_communication.inference.generator import SequenceGeneratorOptions

    translated: list[torch.Tensor] = []
    total_src = 0.0
    total_tgt = 0.0
    output_sr = INPUT_SAMPLE_RATE  # updated from pretssel output

    text_gen_opts = SequenceGeneratorOptions(
        beam_size=beam_size,
        len_penalty=len_penalty,
        soft_max_seq_len=(1, 200),
    )

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
            text_generation_opts=text_gen_opts,
        )

        if unit_output is None:
            print("warning: no speech output for this chunk, skipping")
            continue
        speech_output = pretssel_generator.predict(
            unit_output.units,
            tgt_lang=tgt_lang,
            prosody_encoder_input=src_gcmvn,
        )

        output_sr = speech_output.sample_rate
        wav = speech_output.audio_wavs[0].to(torch.float32).cpu()
        while wav.dim() > 1:
            wav = wav.squeeze(0)
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
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for text decoder; higher = better quality but slower (default: 5)",
    )
    parser.add_argument(
        "--len-penalty",
        type=float,
        default=1.0,
        help="Length penalty for text decoder; <1 = shorter output, >1 = longer output (default: 1.0)",
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

    check_prerequisites()

    device, dtype = resolve_device(args.device)

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
        extra_str = ""
        if args.beam_size != 5:
            extra_str += f", beam_size={args.beam_size}"
        if args.len_penalty != 1.0:
            extra_str += f", len_penalty={args.len_penalty}"
        print(f"Translating to '{args.tgt_lang}'{df_str}{extra_str} …")
        translated, output_sr = translate_chunks(
            chunks,
            translator, pretssel_generator, prosody_fbank_extractor,
            gcmvn_mean, gcmvn_std,
            tgt_lang=args.tgt_lang,
            duration_factor=args.duration_factor,
            beam_size=args.beam_size,
            len_penalty=args.len_penalty,
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
