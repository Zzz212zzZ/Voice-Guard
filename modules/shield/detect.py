import os
import sys

# Patch omegaconf compatibility for bundled fairseq (expects omegaconf<2.1).
# `is_primitive_type` was removed in omegaconf 2.1; fairseq's checkpoint_utils references it.
import omegaconf._utils
if not hasattr(omegaconf._utils, "is_primitive_type"):
    omegaconf._utils.is_primitive_type = lambda _: True

import librosa
import numpy as np
import torch
from torch import Tensor


class DeepfakeDetector:
    """Wraps W2V-AASIST for single-file deepfake audio detection.

    Usage:
        detector = DeepfakeDetector(ssl_root="SSL_Anti-spoofing/", checkpoints_root="checkpoints/")
        result = detector.detect("audio.wav")
        # result = {"verdict": "FAKE", "confidence": 91.7, "spoof_prob": 0.917, ...}
    """

    MAX_AUDIO_LEN = 64600  # ~4s at 16kHz (model's expected input length)

    def __init__(self, ssl_root: str, checkpoints_root: str, device: str = None):
        self.ssl_root = os.path.abspath(ssl_root)
        self.checkpoints_root = os.path.abspath(checkpoints_root)
        self.device = self._resolve_device(device)

        # Add SSL_Anti-spoofing to path so `from model import Model` works
        if self.ssl_root not in sys.path:
            sys.path.insert(0, self.ssl_root)

        # SSLModel hardcodes `cp_path = 'xlsr2_300m.pt'` (model.py line 24).
        # Temporarily chdir to checkpoints so it finds the file.
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.checkpoints_root)
            from model import Model
            self._model = Model(args=None, device=self.device)
        finally:
            os.chdir(orig_cwd)

        # Load pretrained W2V-AASIST weights
        ckpt_path = os.path.join(self.checkpoints_root, "best_SSL_model_DF.pth")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        # Strip 'module.' prefix from DataParallel-saved checkpoint
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        self._model.load_state_dict(checkpoint)
        self._model.to(self.device)
        self._model.eval()

        print(f"DeepfakeDetector ready (device={self.device})")

    @staticmethod
    def _resolve_device(device: str = None) -> str:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad (by tiling) or truncate audio to MAX_AUDIO_LEN samples."""
        if len(audio) >= self.MAX_AUDIO_LEN:
            return audio[:self.MAX_AUDIO_LEN]
        num_repeats = int(self.MAX_AUDIO_LEN / len(audio)) + 1
        return np.tile(audio, num_repeats)[:self.MAX_AUDIO_LEN]

    def detect(self, audio_path: str) -> dict:
        """Detect whether an audio file is real or fake.

        Args:
            audio_path: Path to audio file.

        Returns:
            Dict with verdict, confidence, spoof_prob, bonafide_prob.
        """
        # Load at 16kHz (model expects 16kHz)
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = self._pad_or_truncate(audio)

        x = Tensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(x)  # shape: (1, 2) — [bonafide, spoof]

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Model output: index 0 = spoof, index 1 = bonafide
        spoof_prob = float(probs[0])
        bonafide_prob = float(probs[1])

        if spoof_prob > 0.7:
            verdict = "FAKE"
        elif spoof_prob > 0.3:
            verdict = "SUSPICIOUS"
        else:
            verdict = "REAL"

        confidence = max(bonafide_prob, spoof_prob) * 100

        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "spoof_prob": round(spoof_prob, 4),
            "bonafide_prob": round(bonafide_prob, 4),
        }

    def detect_with_attention(self, audio_path: str) -> dict:
        """Detect with attention map extraction for explainability.

        Returns the same verdict dict as detect(), plus an 'attention' key
        containing numpy arrays of shape (freq, time) for visualization.
        """
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = self._pad_or_truncate(audio)

        x = Tensor(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, att_maps = self._model(x, return_attention=True)

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        spoof_prob = float(probs[0])
        bonafide_prob = float(probs[1])

        if spoof_prob > 0.7:
            verdict = "FAKE"
        elif spoof_prob > 0.3:
            verdict = "SUSPICIOUS"
        else:
            verdict = "REAL"

        confidence = max(bonafide_prob, spoof_prob) * 100

        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "spoof_prob": round(spoof_prob, 4),
            "bonafide_prob": round(bonafide_prob, 4),
            "attention": {
                "spectral": att_maps["spectral"][0].cpu().numpy(),
                "temporal": att_maps["temporal"][0].cpu().numpy(),
                "combined": att_maps["combined"][0].cpu().numpy(),
            },
        }

    def full_analyze(self, audio_path: str) -> dict:
        """Run W2V-AASIST for verdict + spectral/prosody for explainability.

        Verdict comes solely from W2V-AASIST. Spectral and prosody features
        are provided as raw values for the UI to display — no scoring.
        Includes attention maps for heatmap visualization.
        """
        from modules.shield.spectral import analyze_spectral
        from modules.shield.prosody import analyze_prosody

        w2v = self.detect_with_attention(audio_path)
        spectral = analyze_spectral(audio_path)
        prosody = analyze_prosody(audio_path)

        return {
            "verdict": w2v["verdict"],
            "confidence": w2v["confidence"],
            "spoof_prob": w2v["spoof_prob"],
            "attention": w2v["attention"],
            "spectral": spectral,
            "prosody": prosody,
        }
