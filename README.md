# yt-podcast

A CLI tool that downloads audio from YouTube and translates it into another language using [Meta's SeamlessM4T v2](https://github.com/facebookresearch/seamless_communication) — a speech-to-speech translation model.

The result is a translated audio file in the target language.

## Features

- **YouTube Audio Download** -- Extracts audio from any YouTube video URL using `yt-dlp`
- **Speech-to-Speech Translation** -- Direct audio-to-audio translation (no intermediate text)
- **Silence-Aware Chunking** -- Intelligently splits audio at silence boundaries to avoid mid-utterance cuts
- **36 Languages** -- English, French, German, Spanish, Mandarin, Japanese, Korean, Arabic, and more
- **Duration Control** -- Tune speech rate with `--duration-factor`
- **Voice Selection** -- Choose from multiple voices per language by speaker ID
- **Multiple Output Formats** -- WAV, MP3, and OGG
- **Device Auto-Detection** -- Automatically uses CUDA (GPU), MPS (Apple Silicon), or CPU
- **Side-by-Side Original** -- Automatically saves the original audio alongside the translated output

## Prerequisites

- Python 3.11 (fairseq2n 0.2.x has no cp312 wheels)
- `yt-dlp` -- installed and on PATH
- `ffmpeg` -- installed and on PATH

## Installation

```bash
# Create a Python 3.11 virtual environment:
uv venv .venv --python python3.11
source .venv/bin/activate

# Install torch+torchaudio with CUDA 12.1 support:
uv pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only:
uv pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install seamless_communication from GitHub (also pulls in fairseq2, numpy):
uv pip install "seamless_communication @ git+https://github.com/facebookresearch/seamless_communication.git"

# Install remaining dependencies:
uv pip install sentencepiece yt-dlp
```

## Usage

```bash
# Basic translation to English
python translate.py "https://youtube.com/watch?v=..." -o output.mp3

# Translate to French
python translate.py "https://youtube.com/watch?v=..." --tgt-lang fra

# Adjust speech rate (1.1 = 10% slower)
python translate.py "https://youtube.com/watch?v=..." --duration-factor 1.1

# List available speaker IDs (no URL required, loads model)
python translate.py --list-voices

# List speakers for a specific target language
python translate.py --list-voices --tgt-lang fra

# Select a specific speaker
python translate.py "https://youtube.com/watch?v=..." --tgt-lang ind --voice 24
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `url` | (required unless `--list-voices`) | YouTube video URL |
| `-o, --output` | `translated.mp3` | Output file (wav, mp3, ogg) |
| `--tgt-lang` | `eng` | Target language code (36 supported) |
| `--duration-factor` | `1.0` | Speech rate tuning (>1 slower, <1 faster) |
| `--device` | auto-detected | Compute device: cuda, cpu, mps |
| `--min-chunk` | 15 | Min chunk duration in seconds |
| `--max-chunk` | 30 | Max chunk duration (force-split threshold) |
| `--silence-thresh` | -25 | Silence threshold in dBFS |
| `--voice` | first speaker for target lang | Speaker ID for translated output (globally unique). Use `--list-voices` to see available IDs |
| `--list-voices` | — | List all available speaker IDs and exit (loads model, no URL required) |

## How It Works

1. **Download** -- `yt-dlp` extracts audio from the YouTube URL as 16 kHz mono WAV
2. **Silence Detection** -- Computes per-frame RMS energy and identifies silence regions
3. **Chunking** -- Splits the waveform at silence boundaries, respecting min/max chunk size constraints
4. **Translation** -- Each chunk is processed through SeamlessM4T v2 (speech-to-speech translation + vocoder)
5. **Output** -- All translated chunks are concatenated and saved in the requested format

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.2.2, torchaudio 2.2.2 |
| Translation Model | Meta SeamlessM4T v2 (seamless_communication) |
| Audio Framework | fairseq2 0.2.1 |
| Audio Download | yt-dlp |
| Audio Conversion | ffmpeg |

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
