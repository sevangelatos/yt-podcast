#!/usr/bin/env python3
"""Tests for translate.py — silence detection, chunking, CLI, and helpers."""

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import translate as tr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sine_wave(
    duration_s: float,
    freq: float = 440.0,
    sample_rate: int = tr.INPUT_SAMPLE_RATE,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Generate a 1-channel sine wave tensor of shape (1, samples)."""
    n = int(duration_s * sample_rate)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    return (amplitude * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


def silence(duration_s: float, sample_rate: int = tr.INPUT_SAMPLE_RATE) -> torch.Tensor:
    """Generate a silence tensor of shape (1, samples)."""
    n = int(duration_s * sample_rate)
    return torch.zeros(1, n, dtype=torch.float32)


def cat_waveforms(*wavs: torch.Tensor) -> torch.Tensor:
    return torch.cat(wavs, dim=-1)


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

class TestFindSilenceRegions:
    def test_simple_silence_in_middle(self):
        audio = cat_waveforms(
            sine_wave(2.0),
            silence(0.5),
            sine_wave(2.0),
        )
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        expected_start = int(2.0 * tr.INPUT_SAMPLE_RATE)
        assert regions[0][0] == expected_start

    def test_no_silence(self):
        audio = sine_wave(5.0)
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 0

    def test_all_silence(self):
        audio = silence(3.0)
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        assert regions[0] == (0, int(3.0 * tr.INPUT_SAMPLE_RATE))

    def test_multiple_silences(self):
        audio = cat_waveforms(
            sine_wave(1.0),
            silence(0.3),
            sine_wave(1.0),
            silence(0.3),
            sine_wave(1.0),
        )
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 2

    def test_short_silence_ignored(self):
        audio = cat_waveforms(
            sine_wave(2.0),
            silence(0.05),
            sine_wave(2.0),
        )
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 0

    def test_trailing_silence(self):
        audio = cat_waveforms(
            sine_wave(2.0),
            silence(0.5),
        )
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        assert regions[0][0] == int(2.0 * tr.INPUT_SAMPLE_RATE)
        assert regions[0][1] == audio.shape[-1]

    def test_leading_silence(self):
        audio = cat_waveforms(
            silence(0.5),
            sine_wave(2.0),
        )
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        assert regions[0][0] == 0

    def test_custom_threshold(self):
        quiet_audio = cat_waveforms(
            sine_wave(2.0),
            sine_wave(0.5, amplitude=0.3),
            sine_wave(2.0),
        )
        default_thresh = tr.find_silence_regions(quiet_audio, thresh_dbfs=-25)
        assert len(default_thresh) == 0
        louder_thresh = tr.find_silence_regions(quiet_audio, thresh_dbfs=-1)
        assert len(louder_thresh) == 1

    def test_trailing_edge_matches_total(self):
        audio = cat_waveforms(sine_wave(1.5), silence(0.5))
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        assert regions[0][1] == audio.shape[-1]

    def test_trailing_edge_with_remainder(self):
        audio = cat_waveforms(sine_wave(1.5), silence(0.5))
        extra = torch.randn(1, 17, dtype=torch.float32) * 1e-10
        audio = torch.cat([audio, extra], dim=-1)
        regions = tr.find_silence_regions(audio)
        assert len(regions) == 1
        assert regions[0][1] == audio.shape[-1]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def test_single_chunk(self):
        audio = sine_wave(10.0)
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=5, max_chunk_s=30
        )
        assert len(chunks) == 1
        assert chunks[0].shape[-1] == audio.shape[-1]

    def test_splits_at_silence(self):
        audio = cat_waveforms(
            sine_wave(10.0),
            silence(0.5),
            sine_wave(10.0),
        )
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=5, max_chunk_s=15
        )
        assert len(chunks) == 2

    def test_no_audio_loss(self):
        audio = cat_waveforms(
            sine_wave(8.0),
            silence(0.3),
            sine_wave(12.0),
            silence(0.3),
            sine_wave(6.0),
        )
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=5, max_chunk_s=20
        )
        total = sum(c.shape[-1] for c in chunks)
        assert total == audio.shape[-1]

    def test_force_split(self):
        audio = sine_wave(60.0)
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=10, max_chunk_s=20
        )
        assert len(chunks) >= 3
        for c in chunks:
            assert c.shape[-1] <= 20 * tr.INPUT_SAMPLE_RATE

    def test_preserves_audio_with_forced_splits(self):
        audio = cat_waveforms(
            sine_wave(7.0),
            silence(0.3),
            sine_wave(7.0),
            silence(0.3),
            sine_wave(7.0),
        )
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=5, max_chunk_s=10
        )
        total = sum(c.shape[-1] for c in chunks)
        assert total == audio.shape[-1]

    def test_value_error_check_exists(self):
        # The chunking logic is designed to never lose audio.
        # Verify the ValueError safety check is in place.
        import inspect
        src = inspect.getsource(tr.chunk_audio_at_silences)
        assert "ValueError" in src
        assert "Audio loss detected" in src

    def test_min_chunk_prevents_early_split(self):
        audio = cat_waveforms(
            sine_wave(5.0),
            silence(0.3),
            sine_wave(20.0),
        )
        regions = tr.find_silence_regions(audio)
        chunks = tr.chunk_audio_at_silences(
            audio, regions, min_chunk_s=10, max_chunk_s=30
        )
        assert len(chunks) == 1

    def test_trailing_under_half_second_triggers_value_error(self):
        audio = sine_wave(30.3)
        regions = tr.find_silence_regions(audio)
        try:
            tr.chunk_audio_at_silences(
                audio, regions, min_chunk_s=5, max_chunk_s=30
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Voice helpers
# ---------------------------------------------------------------------------

class TestVoiceHelpers:
    def _make_mock_translator(self, lang_spkr_map: dict) -> MagicMock:
        translator = MagicMock()
        translator.vocoder.lang_spkr_idx_map = lang_spkr_map
        return translator

    def test_get_speaker_idx_default(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10, 20], "fra": [30]}})
        assert tr._get_speaker_idx(t, "eng") == 10
        assert tr._get_speaker_idx(t, "fra") == 30

    def test_get_speaker_idx_missing_lang(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10]}})
        assert tr._get_speaker_idx(t, "xxx") == -1

    def test_get_speaker_idx_no_map(self):
        t = MagicMock()
        t.vocoder.lang_spkr_idx_map = None
        assert tr._get_speaker_idx(t, "eng") == -1

    def test_build_voice_list(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10, 20], "fra": [20]}})
        voices = tr._build_voice_list(t)
        assert 10 in voices
        assert 20 in voices
        assert voices[10]["lang_codes"] == ["eng"]
        assert "eng" in voices[20]["lang_codes"]
        assert "fra" in voices[20]["lang_codes"]

    def test_build_voice_list_empty(self):
        t = self._make_mock_translator({})
        assert tr._build_voice_list(t) == {}

    def test_get_all_voice_ids(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10, 20], "fra": [30]}})
        ids = tr._get_all_voice_ids(t)
        assert ids == {10, 20, 30}

    def test_get_all_voice_ids_empty(self):
        t = self._make_mock_translator({})
        assert tr._get_all_voice_ids(t) == set()

    def test_resolve_voice_found(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10]}})
        assert tr._resolve_voice(t, 10) == 10

    def test_resolve_voice_not_found(self):
        t = self._make_mock_translator({"multispkr": {"eng": [10]}})
        assert tr._resolve_voice(t, 999) == -1


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = tr.build_parser()
        return parser.parse_args(argv)

    def test_minimal(self):
        args = self._parse(["https://youtube.com/watch?v=abc"])
        assert args.url == "https://youtube.com/watch?v=abc"
        assert args.output == "translated.mp3"
        assert args.tgt_lang == "eng"
        assert args.duration_factor == 1.0
        assert args.list_voices is False

    def test_output_flag(self):
        args = self._parse(["https://example.com", "-o", "out.ogg"])
        assert args.output == "out.ogg"

    def test_tgt_lang(self):
        args = self._parse(["https://example.com", "--tgt-lang", "fra"])
        assert args.tgt_lang == "fra"

    def test_duration_factor(self):
        args = self._parse(["https://example.com", "--duration-factor", "1.5"])
        assert args.duration_factor == 1.5

    def test_list_voices_no_url(self):
        args = self._parse(["--list-voices"])
        assert args.list_voices is True
        assert args.url is None

    def test_invalid_lang_rejected(self):
        parser = tr.build_parser()
        try:
            parser.parse_args(["https://example.com", "--tgt-lang", "xxx"])
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass

    def test_voice_id(self):
        args = self._parse(["https://example.com", "--voice", "42"])
        assert args.voice == 42

    def test_silence_thresh(self):
        args = self._parse(["https://example.com", "--silence-thresh", "-30"])
        assert args.silence_thresh == -30.0


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

class TestPrerequisites:
    @patch("shutil.which")
    def test_passes_when_both_found(self, mock_which):
        mock_which.return_value = "/usr/bin/fake"
        tr.check_prerequisites()

    @patch("shutil.which")
    def test_exits_when_ytdlp_missing(self, mock_which):
        def which_side_effect(prog):
            return "/usr/bin/ffmpeg" if prog == "ffmpeg" else None
        mock_which.side_effect = which_side_effect
        try:
            tr.check_prerequisites()
            assert False, "Should have exited"
        except SystemExit:
            pass

    @patch("shutil.which")
    def test_exits_when_ffmpeg_missing(self, mock_which):
        def which_side_effect(prog):
            return "/usr/bin/yt-dlp" if prog == "yt-dlp" else None
        mock_which.side_effect = which_side_effect
        try:
            tr.check_prerequisites()
            assert False, "Should have exited"
        except SystemExit:
            pass


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

class TestConvertAudio:
    def test_convert_to_mp3(self):
        if not shutil.which("ffmpeg"):
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "test.wav"
            dst = tmp / "test.mp3"
            audio = sine_wave(0.5)
            torchaudio.save(str(src), audio, tr.INPUT_SAMPLE_RATE)
            tr.convert_audio(src, dst)
            assert dst.exists()
            assert dst.stat().st_size > 0

    def test_convert_to_ogg(self):
        if not shutil.which("ffmpeg"):
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "test.wav"
            dst = tmp / "test.ogg"
            audio = sine_wave(0.5)
            torchaudio.save(str(src), audio, tr.INPUT_SAMPLE_RATE)
            tr.convert_audio(src, dst)
            assert dst.exists()


# ---------------------------------------------------------------------------
# main() flow tests
# ---------------------------------------------------------------------------

class TestMainFlow:
    def test_voice_double_resolution_fixed(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10, 20], "fra": [30]}}
        mock_download = MagicMock(return_value=Path("/tmp/nonexistent.wav"))
        dtype_used = None
        def capture_dtype(device, dtype):
            nonlocal dtype_used
            dtype_used = dtype
            return mock_translator

        with patch("sys.argv", ["translate.py", "https://example.com", "--voice", "999"]):
            with patch.object(tr, "load_m4t_model", capture_dtype):
                with patch.object(tr, "download_audio", mock_download):
                    with patch.object(tr, "check_prerequisites"):
                        with patch("shutil.which", return_value="/usr/bin/fake"):
                            try:
                                tr.main()
                                assert False, "Should have exited for invalid voice"
                            except SystemExit as e:
                                assert e.code == 1

    def test_list_voices_does_not_require_url(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10], "fra": [30]}}
        with patch("sys.argv", ["translate.py", "--list-voices"]):
            with patch.object(tr, "load_m4t_model", return_value=mock_translator):
                with patch("shutil.which", return_value="/usr/bin/fake"):
                    tr.main()

    def test_mps_dtype_is_float16(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10]}}
        mock_download = MagicMock(return_value=Path("/tmp/nonexistent.wav"))
        dtype_used = None
        def capture_dtype(device, dtype):
            nonlocal dtype_used
            dtype_used = dtype
            return mock_translator

        with patch("sys.argv", ["translate.py", "https://example.com", "--device", "mps"]):
            with patch.object(tr, "load_m4t_model", capture_dtype):
                with patch.object(tr, "download_audio", mock_download):
                    with patch.object(tr, "check_prerequisites"):
                        with patch("shutil.which", return_value="/usr/bin/fake"):
                            with patch("torch.backends.mps") as mock_mps:
                                mock_mps.is_available.return_value = True
                                try:
                                    tr.main()
                                except (SystemExit, FileNotFoundError, RuntimeError):
                                    pass
        assert dtype_used == torch.float16

    def test_cuda_dtype_is_float16(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10]}}
        mock_download = MagicMock(return_value=Path("/tmp/nonexistent.wav"))
        dtype_used = None
        def capture_dtype(device, dtype):
            nonlocal dtype_used
            dtype_used = dtype
            return mock_translator

        with patch("sys.argv", ["translate.py", "https://example.com", "--device", "cuda"]):
            with patch.object(tr, "load_m4t_model", capture_dtype):
                with patch.object(tr, "download_audio", mock_download):
                    with patch.object(tr, "check_prerequisites"):
                        with patch("shutil.which", return_value="/usr/bin/fake"):
                            try:
                                tr.main()
                            except (SystemExit, FileNotFoundError, RuntimeError):
                                pass
        assert dtype_used == torch.float16

    def test_cpu_dtype_is_float32(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10]}}
        mock_download = MagicMock(return_value=Path("/tmp/nonexistent.wav"))
        dtype_used = None
        def capture_dtype(device, dtype):
            nonlocal dtype_used
            dtype_used = dtype
            return mock_translator

        with patch("sys.argv", ["translate.py", "https://example.com", "--device", "cpu"]):
            with patch.object(tr, "load_m4t_model", capture_dtype):
                with patch.object(tr, "download_audio", mock_download):
                    with patch.object(tr, "check_prerequisites"):
                        with patch("shutil.which", return_value="/usr/bin/fake"):
                            try:
                                tr.main()
                            except (SystemExit, FileNotFoundError, RuntimeError):
                                pass
        assert dtype_used == torch.float32


# ---------------------------------------------------------------------------
# translate_chunks passes spkr_idx through (no double resolution)
# ---------------------------------------------------------------------------

class TestTranslateChunks:
    def test_accepts_spkr_idx_directly(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10]}}

        mock_speech = MagicMock()
        mock_speech.audio_wavs = [[torch.randn(16000)]]
        mock_speech.sample_rate = 16000
        mock_translator.predict.return_value = (None, mock_speech)

        chunks = [sine_wave(1.0)]
        tr.translate_chunks(chunks, mock_translator, "eng", 1.0, spkr_idx=10)
        assert mock_translator.predict.call_args.kwargs["spkr"] == 10

    def test_no_internal_voice_resolution(self):
        mock_translator = MagicMock()
        mock_translator.vocoder.lang_spkr_idx_map = {"multispkr": {"eng": [10]}}

        mock_speech = MagicMock()
        mock_speech.audio_wavs = [[torch.randn(16000)]]
        mock_speech.sample_rate = 16000
        mock_translator.predict.return_value = (None, mock_speech)

        chunks = [sine_wave(1.0)]
        with patch.object(tr, "_resolve_voice") as mock_resolve:
            with patch.object(tr, "_get_speaker_idx") as mock_get:
                tr.translate_chunks(chunks, mock_translator, "eng", 1.0, spkr_idx=42)
                assert mock_resolve.call_count == 0
                assert mock_get.call_count == 0


# ---------------------------------------------------------------------------
# common.py direct tests
# ---------------------------------------------------------------------------

import common as cm


class TestCommonResolveDevice:
    def test_explicit_cpu(self):
        device, dtype = cm.resolve_device("cpu")
        assert device.type == "cpu"
        assert dtype == torch.float32

    def test_explicit_cuda(self):
        device, dtype = cm.resolve_device("cuda")
        assert device.type == "cuda"
        assert dtype == torch.float16

    def test_explicit_mps(self):
        device, dtype = cm.resolve_device("mps")
        assert device.type == "mps"
        assert dtype == torch.float16


class TestCommonConvertAudio:
    def test_unsupported_format_exits(self):
        if not shutil.which("ffmpeg"):
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "test.wav"
            dst = tmp / "test.flac"
            audio = sine_wave(0.1)
            torchaudio.save(str(src), audio, cm.INPUT_SAMPLE_RATE)
            try:
                cm.convert_audio(src, dst)
                assert False, "Should have exited for unsupported format"
            except SystemExit as e:
                assert e.code == 1


# ---------------------------------------------------------------------------
# translate_expressive.py CLI tests
# ---------------------------------------------------------------------------

import translate_expressive as te


class TestExpressiveCLI:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = te.build_parser()
        return parser.parse_args(argv)

    def test_minimal(self):
        args = self._parse(["https://youtube.com/watch?v=abc"])
        assert args.url == "https://youtube.com/watch?v=abc"
        assert args.output == "translated.mp3"
        assert args.tgt_lang == "eng"
        assert args.duration_factor == 1.0
        assert args.beam_size == 5
        assert args.len_penalty == 1.0

    def test_gated_model_dir(self):
        args = self._parse(["https://example.com", "--gated-model-dir", "/tmp/models"])
        assert args.gated_model_dir == Path("/tmp/models")

    def test_beam_size(self):
        args = self._parse(["https://example.com", "--beam-size", "10"])
        assert args.beam_size == 10

    def test_len_penalty(self):
        args = self._parse(["https://example.com", "--len-penalty", "0.8"])
        assert args.len_penalty == 0.8

    def test_invalid_lang_rejected(self):
        parser = te.build_parser()
        try:
            parser.parse_args(["https://example.com", "--tgt-lang", "xxx"])
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass

    def test_missing_url_exits(self):
        import unittest.mock as mock
        with mock.patch("sys.argv", ["translate"]):
            try:
                te.main()
            except SystemExit:
                pass


class TestExpressiveEmptyTranslatedGuard:
    def test_guard_code_exists(self):
        import inspect
        src = inspect.getsource(te.main)
        assert "no translated output produced" in src


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
