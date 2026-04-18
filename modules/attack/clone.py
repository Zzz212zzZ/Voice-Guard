import os
import sys
import types

import torch
import torch.nn.functional as F

# Stub out MeCab so MeloTTS's japanese.py import succeeds without the real binary.
# We only use English, so the Japanese code path is never called.
if "MeCab" not in sys.modules:
    _mecab = types.ModuleType("MeCab")

    class _Tagger:
        def parse(self, text):
            return ""

    _mecab.Tagger = _Tagger
    sys.modules["MeCab"] = _mecab


class VoiceCloner:
    """Wraps OpenVoice v2 + MeloTTS into a single voice cloning callable.

    Usage:
        cloner = VoiceCloner(openvoice_root="OpenVoice/")
        output_path, sr = cloner.clone("Hello world", "ref.mp3")
    """

    def __init__(self, openvoice_root: str, device: str = None):
        self.openvoice_root = os.path.abspath(openvoice_root)
        self.device = self._resolve_device(device)

        # Paths relative to OpenVoice root
        ckpt_dir = os.path.join(self.openvoice_root, "checkpoints_v2")
        self._converter_config = os.path.join(ckpt_dir, "converter", "config.json")
        self._converter_weights = os.path.join(ckpt_dir, "converter", "checkpoint.pth")
        self._source_se_path = os.path.join(ckpt_dir, "base_speakers", "ses", "en-newest.pth")
        self._output_dir = os.path.join(self.openvoice_root, "outputs_v2")
        self._processed_dir = os.path.join(self.openvoice_root, "processed")
        # Default demo voice for the live presentation
        self.default_reference = os.path.join(self.openvoice_root, "resources", "demo_speaker0.mp3")
        self.sample_rate = 22050

        os.makedirs(self._output_dir, exist_ok=True)

        # Load ToneColorConverter
        from openvoice.api import ToneColorConverter
        self._converter = ToneColorConverter(self._converter_config, device=self.device)
        self._converter.load_ckpt(self._converter_weights)

        # Load source speaker embedding (EN_NEWEST)
        self._source_se = torch.load(self._source_se_path, map_location=self.device)

        # Load MeloTTS
        from melo.api import TTS as MeloTTS
        self._tts = MeloTTS(language="EN_NEWEST", device=self.device)
        self._speaker_id = self._tts.hps.data.spk2id["EN-Newest"]

        print(f"VoiceCloner ready (device={self.device})")

    @staticmethod
    def _resolve_device(device: str = None) -> str:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def clone(self, text: str, reference_audio: str = None, speed: float = 1.0) -> tuple:
        """Clone a voice: generate speech from text using the target voice.

        Args:
            text: English text to speak.
            reference_audio: Path to target speaker audio (3-30s). None = default.
            speed: Speech speed multiplier (0.5-2.0).

        Returns:
            (output_path, sample_rate) tuple.
        """
        if not text or len(text.strip()) < 2:
            raise ValueError("Text must be at least 2 characters.")

        if reference_audio is None:
            reference_audio = self.default_reference

        # 1. Extract target speaker embedding from reference audio
        from openvoice import se_extractor
        target_se, _ = se_extractor.get_se(
            reference_audio, self._converter,
            target_dir=self._processed_dir, vad=True,
        )

        # 2. Generate base TTS audio with MeloTTS
        tmp_path = os.path.join(self._output_dir, "tmp.wav")
        self._tts.tts_to_file(text, self._speaker_id, tmp_path, speed=speed)

        # 3. Convert tone color to match target speaker
        output_path = os.path.join(self._output_dir, "cloned_output.wav")
        self._converter.convert(
            audio_src_path=tmp_path,
            src_se=self._source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell",
        )

        return output_path, self.sample_rate

    def compute_similarity(self, ref_audio_path: str, cloned_audio_path: str) -> float:
        """Cosine similarity between speaker embeddings of two audio files (0-1)."""
        ref_se = self._converter.extract_se([ref_audio_path])
        clone_se = self._converter.extract_se([cloned_audio_path])
        sim = F.cosine_similarity(ref_se.view(1, -1), clone_se.view(1, -1))
        return sim.item()
