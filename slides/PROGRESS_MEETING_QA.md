# VoiceGuard — Progress Meeting Notes & Q&A

## Current Progress Summary

### Attack Tab (Voice Cloning)
- Integrated **OpenVoice v2** for zero-shot voice cloning
- Users upload a short reference clip (5–30 sec), type any text, and generate cloned speech
- The system uses a **three-step pipeline**: (1) extract a speaker embedding from the reference audio, (2) generate base speech from text using MeloTTS, (3) convert the base speech to match the target voice's tone color
- Outputs: cloned audio playback, **similarity score** (cosine similarity of speaker embeddings, typically ~80%), side-by-side waveform and mel-spectrogram comparisons
- "Send to Shield" button passes cloned audio directly to the detection tab

### Shield Tab (Deepfake Detection)
- Integrated **W2V-AASIST** as the sole detection model — it outputs a spoof probability, which is mapped to a verdict: **REAL** (< 30%), **SUSPICIOUS** (30–70%), or **FAKE** (> 70%)
- Added acoustic feature extraction as **explainability aids** (not part of the verdict):
  - Voice quality: Jitter, Shimmer, F0 CV, Energy CV
  - Spectral: Centroid, Bandwidth, Flatness, Rolloff
  - Visualizations: Mel spectrogram, Pitch contour (F0), Energy envelope

### Key Observations
- W2V-AASIST correctly identifies our cloned audio as FAKE with very high confidence (~100%)
- Some **false positives** observed — the model occasionally flags real speech as fake
- Feature visualizations show raw values; we have not yet established reliable "normal ranges" for real vs. fake speech

---

## Q&A — Anticipated Questions

---

### Q1: What is W2V-AASIST and why did you choose it?

**W2V-AASIST** is a neural network that combines two components:

1. **Wav2Vec 2.0 (XLSR)** — a large self-supervised model originally trained on 300,000+ hours of multilingual speech. It converts raw audio into rich acoustic feature representations (1024-dimensional vectors). Think of it as a "universal audio understanding" layer — it has learned general patterns of human speech across many languages.

2. **AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) — a specialized backend that uses **graph attention networks** to model relationships between spectral (frequency) and temporal (time) patterns in audio. It learns to spot artifacts that distinguish real speech from synthesized speech.

**Why we chose it:**
- It is **state-of-the-art** on the ASVspoof benchmarks, which are the standard evaluation datasets for voice anti-spoofing
- It is designed **specifically for audio deepfake detection** — the AASIST component was built to catch spectro-temporal artifacts that voice synthesis methods produce
- Unlike simpler classifiers that look at individual features, the graph attention mechanism captures **relationships across time and frequency simultaneously**, making it harder for synthesizers to fool
- Pretrained weights are publicly available (`best_SSL_model_DF.pth`, trained on the ASVspoof 2019 DF dataset), so we can use it directly without needing our own training data or GPU resources
- It handles diverse attack types (TTS, voice conversion, etc.) in a **single model**, rather than needing separate detectors for each

**Is it specifically for deepfake detection?**
Yes. The wav2vec 2.0 frontend is general-purpose (speech recognition, speaker ID, etc.), but the AASIST backend is purpose-built for anti-spoofing. The combined model was fine-tuned end-to-end on the deepfake detection task.

---

### Q2: How does the detection actually work? What does the model output?

1. Audio is loaded at 16 kHz and padded/truncated to ~4 seconds (64,600 samples)
2. The audio passes through wav2vec 2.0 to get a high-level representation
3. AASIST processes this into a graph structure, applies attention across spectral and temporal nodes, and produces two logits: one for "bonafide" (real) and one for "spoof" (fake)
4. Softmax converts these to probabilities that sum to 1
5. If spoof probability > 0.7 → **FAKE**; between 0.3–0.7 → **SUSPICIOUS**; < 0.3 → **REAL**
6. Confidence = the higher of the two probabilities, shown as a percentage

---

### Q3: Why are the acoustic features (jitter, shimmer, spectral centroid, etc.) not used in the verdict?

We initially tried combining heuristic feature thresholds with the neural model for a weighted ensemble verdict. This caused **false positives on real audio** — the hard-coded thresholds were not calibrated against a proper dataset.

We redesigned the system so that:
- **Only W2V-AASIST determines the verdict** — it was trained on tens of thousands of real and fake audio samples and has learned robust decision boundaries
- The acoustic features are shown as **explainability aids** — they give users (and us) insight into *what* might differ between real and fake audio, without overriding the trained model's judgment

In Phase 3, we plan to run these features across the ASVspoof dataset to understand their distributions for real vs. fake speech, which could inform future improvements.

---

### Q4: Why does the model produce false positives on real speech?

Several likely causes:

1. **Domain mismatch** — The model was trained on ASVspoof 2019 data (specific recording conditions, microphones, and speakers). Our test recordings may have different acoustic characteristics (room noise, microphone quality, compression), which the model hasn't seen
2. **Threshold sensitivity** — Our current threshold (0.7 for FAKE) may need tuning based on our specific use case. The model might be overly sensitive
3. **Short or noisy input** — The model expects ~4 seconds of clean speech. Very short clips are padded by repeating, which can create unnatural patterns that the model flags

**Planned investigation:**
- Benchmark on ASVspoof 2019/5 to measure actual EER (Equal Error Rate)
- Experiment with threshold values
- Test with various recording conditions to map out failure cases

---

### Q5: How does the voice cloning pipeline work?

OpenVoice v2 uses a three-stage approach:

1. **Speaker embedding extraction** — Analyzes the reference audio to capture the speaker's unique voice characteristics (a 256-dimensional vector representing their "tone color")
2. **Text-to-speech** — Uses MeloTTS to generate base speech from the input text (this sounds like a default voice, not the target)
3. **Tone color conversion** — Transforms the base speech to match the target speaker's tone color using the extracted embedding

This means the cloning doesn't directly copy the reference audio — it **transfers the voice identity** onto newly generated speech. The similarity score measures how close the cloned speaker embedding is to the original (cosine similarity).

---

### Q6: What does the similarity score mean? Is 81% good?

The similarity score is the **cosine similarity** between speaker embeddings of the reference and cloned audio:
- **> 90%**: Very close voice match
- **80–90%**: Good match — recognizably similar
- **< 80%**: Partial match, likely degraded by short/noisy reference audio

81% means the cloned voice is **recognizably similar** to the original. It's not a perfect copy, but in a real-world scam scenario, it could be convincing enough over a phone call where audio quality is already degraded.

---

### Q7: Can the Shield detect the Attack's output?

**Yes.** In our testing, when we send cloned audio from the Attack tab to the Shield tab, the model identifies it as **FAKE with ~100% confidence** (spoof probability 0.9997). This is a strong result — it means our detector can reliably catch our own cloner.

However, this is an easy case since both use known methods. Phase 3 will test against:
- Other cloning tools (not just OpenVoice)
- Audio with added noise or compression (simulating phone calls)
- The ASVspoof benchmark datasets with diverse attack methods

---

### Q8: What are the limitations of your current system?

1. **Single detection model** — We rely entirely on W2V-AASIST. If an attack method specifically evades this model, we have no fallback
2. **Fixed input length** — The model processes ~4 seconds of audio. Longer audio is truncated (potentially missing important segments), shorter audio is padded by repeating
3. **No real-time processing** — Both cloning and detection run as batch operations, not streaming
4. **Feature interpretation gap** — We display acoustic features but don't yet know their reliable ranges for real vs. fake speech
5. **English only** — OpenVoice v2 supports multiple languages, but our TTS pipeline currently uses the English speaker preset

---

### Q9: What is the tech stack and what does each component do?

| Component | Technology | Role |
|-----------|-----------|------|
| Web UI | Gradio | Interactive interface with two tabs |
| Voice cloning | OpenVoice v2 + MeloTTS | Generate speech in a target voice |
| Detection model | W2V-AASIST (wav2vec 2.0 + AASIST) | Classify audio as real or fake |
| Feature extraction | librosa | Extract spectral and prosody features |
| Visualization | matplotlib | Waveforms, spectrograms, charts |
| Deep learning | PyTorch | Model inference |
| Cost | **$0** | All open-source, free for academic use |

---

### Q10: What are your next steps?

| Task | Purpose |
|------|---------|
| Benchmark on ASVspoof 2019/5 | Get proper metrics (EER, accuracy) on standard datasets |
| Investigate false positives | Understand domain mismatch and tune thresholds |
| Feature range analysis | Run features across real/fake datasets to establish baselines |
| Robustness testing | Add phone noise and compression to cloned audio, re-test detection |
| Explore model alternatives | Try other detectors or ensemble approaches to reduce false positives |
| Polish UI and demo | Prepare for final presentation |

---

### Q11: How is this different from existing tools?

Most existing tools do **either** cloning or detection in isolation. VoiceGuard combines both in one interface, which allows:
- **Adversarial testing** — clone a voice and immediately test if the detector catches it
- **Educational value** — users see how easy cloning is and how detection works, all in one place
- **Iterative improvement** — we can use our own cloned audio as test cases for the detector, creating a feedback loop

---

### Q12: What would you do differently if starting over?

- Start with ASVspoof evaluation **before** building the UI, to understand the model's baseline performance and failure modes earlier
- Avoid the initial ensemble approach (mixing heuristic features with neural model) — it cost us time debugging false positives before we realized the neural model alone performs better
- Set up automated benchmarking scripts earlier so we can track detection accuracy as we make changes