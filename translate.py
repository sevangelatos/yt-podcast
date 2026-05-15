#!/usr/bin/env python3
"""
yt-podcast: Translate YouTube video audio using SeamlessStreaming (S2ST).

Downloads audio from a YouTube video using yt-dlp and translates it
using Meta's SeamlessStreaming — a simultaneous speech-to-speech
translation model from facebook/seamless-streaming.

The model processes audio as a continuous stream using SimulEval's
agent pipeline. Audio is fed in 1-second segments (vs 20ms for true
realtime) because the w2v-BERT encoder re-encodes all accumulated
frames on every push — larger segments reduce push count and avoid
O(n^2) slowdown. The monotonic attention and early-stop mechanism
handle natural segmentation at sentence boundaries.

Requirements:
    pip install fairseq2==0.2.1  # pulls torch==2.2.2
    pip install seamless_communication  # from GitHub
    pip install simuleval~=1.1.3 torchaudio sentencepiece
    yt-dlp and ffmpeg must be available on PATH.

Usage:
    python translate.py "https://youtube.com/watch?v=..." -o output.mp3
    python translate.py "https://youtube.com/watch?v=..." --tgt-lang fra
    python translate.py "https://youtube.com/watch?v=..." --tgt-lang deu --speaker-id 3
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import torch
import torchaudio
from simuleval.data.segments import SpeechSegment, EmptySegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger("yt-podcast")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SAMPLE_RATE = 16_000  # SeamlessStreaming expects 16 kHz input
OUTPUT_SAMPLE_RATE = 16_000  # Vocoder outputs 16 kHz
SUPPORTED_FORMATS = ("wav", "mp3", "ogg")

# Source segment size in samples.
# The OfflineWav2VecBertEncoderAgent re-encodes ALL accumulated fbank frames
# on every push(), so fewer, larger pushes are dramatically faster for offline
# use. 16000 samples = 1 second is a good balance (vs 320 = 20ms for realtime).
SOURCE_SEGMENT_SIZE = 16_000

# Safety limit: max fbank frames before forcing a pipeline reset to prevent OOM.
# The w2v-BERT encoder accumulates fbanks between early-stop resets.
# At 10ms per frame, 6000 frames = ~60 seconds. On an 8GB GPU this uses ~4-5GB.
MAX_FBANK_FRAMES_BEFORE_RESET = 6000

# Target languages supported by SeamlessStreaming S2ST
SUPPORTED_LANGUAGES = {
    "arb": "Arabic",
    "ben": "Bengali",
    "cat": "Catalan",
    "ces": "Czech",
    "cmn": "Mandarin Chinese",
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
    "pes": "Persian",
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
# Helpers
# ---------------------------------------------------------------------------


def _check_prerequisites() -> None:
    for prog in ("yt-dlp", "ffmpeg"):
        if shutil.which(prog) is None:
            print(
                f"Error: '{prog}' not found on PATH. Please install it.",
                file=sys.stderr,
            )
            sys.exit(1)


def download_audio(url: str, work_dir: Path) -> Path:
    """Download audio from *url* as 16 kHz mono WAV using yt-dlp."""
    output_path = work_dir / "source_audio.wav"
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {INPUT_SAMPLE_RATE} -ac 1",
        "-o",
        str(work_dir / "source_audio.%(ext)s"),
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
# Streaming pipeline
# ---------------------------------------------------------------------------


def build_streaming_args(
    device: str,
    dtype: str,
    tgt_lang: str,
    speaker_id: int,
) -> Namespace:
    """Build the args namespace required by SeamlessStreamingS2STAgent."""
    return Namespace(
        # Task
        task="s2st",
        # Model names
        unity_model_name="seamless_streaming_unity",
        monotonic_decoder_model_name="seamless_streaming_monotonic_decoder",
        # Device / precision
        device=device,
        dtype=dtype,
        fp16=(dtype == "fp16"),
        # Feature extractor
        sample_rate=INPUT_SAMPLE_RATE,
        shift_size=10,
        window_size=25,
        feature_dim=80,
        denormalize=True,
        # Wav2Vec-BERT encoder
        min_starting_wait_w2vbert=192,
        # Text decoder (monotonic attention)
        max_len_a=0,
        max_len_b=100,
        max_consecutive_write=50,
        min_starting_wait=1,
        no_early_stop=True,
        tgt_lang=tgt_lang,
        decision_threshold=0.5,
        decision_method="min",
        p_choose_start_layer=0,
        block_ngrams=False,
        # Unit decoder (NAR T2U)
        min_unit_chunk_size=50,
        d_factor=1.0,
        # Vocoder
        vocoder_name="vocoder_v2",
        vocoder_speaker_id=speaker_id,
        # SimulEval
        source_segment_size=SOURCE_SEGMENT_SIZE,
    )


def load_pipeline(args: Namespace):
    """Load the SeamlessStreamingS2STAgent pipeline."""
    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STAgent,
    )

    print("Loading SeamlessStreaming pipeline...")
    t0 = time.time()
    agent = SeamlessStreamingS2STAgent(args)
    elapsed = time.time() - t0
    print(f"Pipeline loaded in {elapsed:.1f}s\n")
    return agent


def _get_encoder_fbank_count(agent) -> int:
    """Return the number of accumulated fbank frames in the w2v-BERT encoder."""
    # module_list[1] is OfflineWav2VecBertEncoderAgent
    try:
        return len(agent.module_list[1].states.source)
    except (IndexError, AttributeError):
        return 0


def translate_streaming(
    waveform: torch.Tensor,
    agent,
    tgt_lang: str,
) -> torch.Tensor:
    """
    Stream audio through the SeamlessStreaming pipeline.

    Feeds audio in 1-second segments (16000 samples at 16 kHz) for efficient
    offline processing. The pipeline's monotonic attention and early-stop
    mechanism handle natural segmentation at sentence boundaries.

    The OfflineWav2VecBertEncoderAgent re-encodes ALL accumulated fbank frames
    on every push(). Using larger segments dramatically reduces the number of
    encoder calls and avoids the O(n^2) slowdown from tiny 20ms chunks.

    A safety reset prevents OOM if the encoder accumulates too many frames
    without an early-stop reset (e.g., during long silences or music).

    Returns the full translated waveform (1-D float32 CPU tensor at 16 kHz).
    """
    # Flatten to 1-D
    samples = waveform.squeeze().tolist()
    total_samples = len(samples)
    total_dur = total_samples / INPUT_SAMPLE_RATE

    chunk_ms = SOURCE_SEGMENT_SIZE / INPUT_SAMPLE_RATE * 1000
    print(f"  Streaming {total_dur:.1f}s of audio ({total_samples} samples)")
    print(f"  Segment size: {SOURCE_SEGMENT_SIZE} samples ({chunk_ms:.0f}ms)")

    # Reset pipeline state
    agent.reset()

    collected_audio: list[list[float]] = []
    num_segments_pushed = 0
    num_output_chunks = 0
    num_safety_resets = 0
    t0 = time.time()

    # Progress interval: report every ~10 seconds of source audio
    progress_interval = max(1, int(10 * INPUT_SAMPLE_RATE / SOURCE_SEGMENT_SIZE))

    # Phase 1: Feed audio segments
    for offset in range(0, total_samples, SOURCE_SEGMENT_SIZE):
        end = min(offset + SOURCE_SEGMENT_SIZE, total_samples)
        chunk = samples[offset:end]
        is_last = end >= total_samples

        # Safety check: if the encoder has accumulated too many fbank frames
        # without an early-stop reset, force a reset to prevent OOM.
        fbank_count = _get_encoder_fbank_count(agent)
        if fbank_count > MAX_FBANK_FRAMES_BEFORE_RESET:
            logger.info(
                f"Safety reset at src={offset / INPUT_SAMPLE_RATE:.0f}s "
                f"(encoder had {fbank_count} fbank frames)"
            )
            agent.reset()
            num_safety_resets += 1

        segment = SpeechSegment(
            content=chunk,
            sample_rate=INPUT_SAMPLE_RATE,
            finished=is_last,
            tgt_lang=tgt_lang,
        )

        agent.push(segment)
        num_segments_pushed += 1

        # Pop any available output
        output = agent.pop()
        if not output.is_empty and hasattr(output, "content") and len(output.content) > 0:
            collected_audio.append(output.content)
            num_output_chunks += 1

        # Progress reporting
        if num_segments_pushed % progress_interval == 0:
            src_time = end / INPUT_SAMPLE_RATE
            out_samples = sum(len(c) for c in collected_audio)
            out_time = out_samples / OUTPUT_SAMPLE_RATE
            elapsed = time.time() - t0
            speed = src_time / elapsed if elapsed > 0 else 0
            print(
                f"  [{src_time:.0f}s/{total_dur:.0f}s] "
                f"output: {out_time:.1f}s, "
                f"speed: {speed:.1f}x realtime"
            )

    # Phase 2: Drain — keep popping until pipeline signals finished
    max_drain_iters = 5000
    for drain_i in range(max_drain_iters):
        output = agent.pop()
        if not output.is_empty and hasattr(output, "content") and len(output.content) > 0:
            collected_audio.append(output.content)
            num_output_chunks += 1
        if output.finished:
            break
        # Push empty finished segments to drive the pipeline forward
        empty = SpeechSegment(
            content=[],
            sample_rate=INPUT_SAMPLE_RATE,
            finished=True,
            tgt_lang=tgt_lang,
        )
        agent.push(empty)
    else:
        logger.warning(
            f"Drain did not finish after {max_drain_iters} iterations — "
            "some audio may be truncated."
        )

    elapsed = time.time() - t0

    if not collected_audio:
        print("Error: no audio output was produced.", file=sys.stderr)
        sys.exit(1)

    # Flatten all output chunks into a single tensor
    all_samples: list[float] = []
    for chunk in collected_audio:
        all_samples.extend(chunk)

    result = torch.tensor(all_samples, dtype=torch.float32)
    out_dur = len(all_samples) / OUTPUT_SAMPLE_RATE

    print(f"\n  Total: {total_dur:.1f}s source -> {out_dur:.1f}s translated")
    print(f"  Chunks produced: {num_output_chunks}")
    if num_safety_resets > 0:
        print(f"  Safety resets: {num_safety_resets}")
    print(f"  Wall time: {elapsed:.1f}s ({total_dur / elapsed:.1f}x realtime)")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    lang_list = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
    parser = argparse.ArgumentParser(
        description="Translate YouTube video audio using SeamlessStreaming (S2ST).",
        epilog=f"Supported target languages: {lang_list}",
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output",
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
        "--speaker-id",
        type=int,
        default=-1,
        help="Vocoder speaker ID; -1 uses the default voice for the language (default: -1)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: cuda, cpu (auto-detected if omitted)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.url:
        parser.error("the following arguments are required: url")

    out = Path(args.output)
    fmt = out.suffix.lstrip(".").lower()
    if fmt not in SUPPORTED_FORMATS:
        parser.error(
            f"Unsupported format '.{fmt}'. Use one of: {', '.join(SUPPORTED_FORMATS)}"
        )

    _check_prerequisites()

    # Device ----------------------------------------------------------------
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    dtype = "fp16" if "cuda" in device else "fp32"
    print(f"Using device: {device} ({dtype})")

    # Build pipeline args ---------------------------------------------------
    pipeline_args = build_streaming_args(
        device=device,
        dtype=dtype,
        tgt_lang=args.tgt_lang,
        speaker_id=args.speaker_id,
    )

    # Load pipeline ---------------------------------------------------------
    agent = load_pipeline(pipeline_args)

    # Output paths ----------------------------------------------------------
    original_path = out.with_stem(out.stem + ".original")

    # Download audio --------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="yt-podcast-") as tmp:
        work_dir = Path(tmp)
        audio_path = download_audio(args.url, work_dir)

        # Load audio --------------------------------------------------------
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != INPUT_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=INPUT_SAMPLE_RATE
            )
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        total_seconds = waveform.shape[-1] / INPUT_SAMPLE_RATE
        print(f"Audio duration: {total_seconds:.1f}s")

        # Translate ---------------------------------------------------------
        print(f"Translating to '{args.tgt_lang}' ...")

        translated_wav = translate_streaming(
            waveform,
            agent,
            tgt_lang=args.tgt_lang,
        )

        # Ensure 2D for torchaudio.save: (channels, samples)
        if translated_wav.dim() == 1:
            translated_wav = translated_wav.unsqueeze(0)

        # Save output -------------------------------------------------------
        if fmt == "wav":
            torchaudio.save(str(out), translated_wav, OUTPUT_SAMPLE_RATE)
        else:
            wav_tmp = work_dir / "translated.wav"
            torchaudio.save(str(wav_tmp), translated_wav, OUTPUT_SAMPLE_RATE)
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
