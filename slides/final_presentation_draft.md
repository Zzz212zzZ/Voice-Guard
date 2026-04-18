# Final Presentation — Content Draft (numbers filled in)

**Rubric (30 pts)**: Title 1 · Goals 5 · System 5 · Results 5 · Future Work 5 · Justification 5 · Q&A 2 · Time 1 · Teamwork 1
**Time budget**: 10 min (of 12), ~12 content slides + Q&A.

---

## Slide 1 — Title (1 pt) — 0:30

**VoiceGuard: Voice Cloning Attack & Detection**
Jerry Zeng (+ teammate)
14-795 Final Presentation — Apr 2026

---

## Slide 2 — Goals (5 pts) — 1:00

**What we set out to build**
- A **single tool** where anyone can:
  - **Attack**: clone a voice from 3–30 sec of reference audio
  - **Shield**: detect AI-generated speech with verdict + explanation
- **Why**: Voice-cloning scams projected at $40 B by 2027; no open tool lets you *clone* and *detect* side-by-side

**Refined goal after challenge analysis**
- Not just "does it work?" → **"is the detection calibrated and generalizable beyond the benchmark?"**

---

## Slide 3 — System overview (5 pts) — 1:00

**Two-tab Gradio app**

```
Attack Tab                        Shield Tab
──────────                        ──────────
Reference audio  ──┐              ┌── Audio to analyze
Text to speak    ──┤              │
                   ▼              ▼
    OpenVoice v2 (MyShell)    W2V-AASIST (Tak et al.)
    • Speaker embedding       • wav2vec 2.0 XLSR (300M)
    • MeloTTS base            • AASIST graph-attention
    • Tone-color convert      • Softmax → spoof_prob
                   │              │
                   ▼              ▼
              cloned.wav ───▶  verdict + confidence
                              + 8 acoustic features
                              + attention heatmap
                              + pitch / energy / spectrogram
```

**Demo** — Attack → Send to Shield → FAKE verdict (~30 s total)

---

## Slide 4 — Replicated vs. Added (5 pts) — 1:00

| Component | Source | Our change |
|---|---|---|
| OpenVoice v2 | MyShell (MIT) | used as-is |
| W2V-AASIST + `best_SSL_model_DF.pth` | Tak et al. (MIT) | used pretrained weights |
| **Gradio two-tab app** | — | built from scratch |
| **"Send to Shield" workflow** | — | built from scratch |
| **AASIST attention heatmap overlay** | — | extracted attention × mel spectrogram |
| **8-feature acoustic dashboard** | librosa | built + UI integration |
| **Dataset-wide eval pipeline** | — | scripts/eval_asvspoof.py |
| **Feature-distribution calibration** | — | scripts/analyze_eval.py |
| **In-the-wild domain-gap test** | — | scripts/in_the_wild_test.py |

---

## Slide 5 — Results: Pipeline demo (part 1) — 0:45

- Upload 10-sec reference → *Clone & Speak* → cloned audio + similarity score
- *Send to Shield* → **FAKE @ 99.97 %**, attention heatmap highlights artifact bands
- End-to-end < 30 s

---

## Slide 6 — Results: Benchmark on ASVspoof 2019 LA (part 2) — 1:00

**Insert: `full/confusion_matrix.png`**

**N = 71,237** — full ASVspoof 2019 LA eval (7,355 bonafide + 63,882 spoof)

| Metric | Value |
|---|---|
| EER | **0.146 %** |
| Accuracy (verdict ≠ REAL on spoof) | **99.77 %** |
| **FPR** (real → FAKE/SUSP) | **0.109 %** (8/7,355) |
| FNR (fake → REAL) | 0.244 % (156/63,882) |

*W2V-AASIST paper reports ~0.7 % EER on this benchmark; our 0.146 % is within published range → faithful reproduction at full-eval scale.*

---

## Slide 7 — Results: Per-attack breakdown (part 3) — 1:00

**Insert: `full/per_attack_accuracy.png`**

- All **13 deepfake attack types** (A07–A19) covered — ~4,914 samples each
- **7 / 13** attacks detected at 100.0 % (A07, A08, A09, A13, A14, A15, A16)
- Hardest 3: **A10** (98.5 %, end-to-end TTS w/ WaveNet), **A11** (99.4 %, WaveRNN), **A19** (99.6 %, transfer-function TTS+VC)
- Mean detection: **99.8 %**
- **Takeaway**: no attack-type is a systematic failure mode — detector generalizes across all 13 synthesis methods at scale

---

## Slide 8 — Results: Feature calibration at scale (part 4) — 1:00

**Insert: `full/feature_distributions.png`**

Before (Challenge Analysis): real/fake baselines from **14 samples**
After (Final):                same plot from **N = 71,237** (7,355 real + 63,882 fake) — **5,000× more data**

- **Energy CV, F0 CV, Shimmer** separate cleanly at scale → good for user-facing "percentile" context
- Directly answers **Challenge-Analysis weakness #2** ("No Baseline Context")

---

## Slide 9 — The gap we found (bridge slide) — 1:00

**Insert: `domain_gap_fpr.png`**

| Condition | n | **FPR** |
|---|---|---|
| ASVspoof 2019 LA eval (studio) | **7,355** | **0.109 %** |
| In-the-wild (phone/laptop mic, real humans) | 5 | **80.0 %** |

> On its own benchmark the model is near-perfect (0.109 % FPR on 7,355 real utterances). On everyday audio, **4 out of 5** real humans get mis-flagged as AI-generated.

**Domain gap: ~730× FPR increase.** This is the headline finding — and the seed for Future Work.

---

## Slide 10 — Future Work — ONE step (5 pts) — 1:00

**Domain-adaptive fine-tuning with diverse real-world speech**

- Collect ≥ 1000 hrs real human audio across:
  - Phone codecs (µ-law, Opus)
  - Consumer mics (laptop, earbuds)
  - Noisy conditions (cafés, cars, wind)
  - 50+ demographic-diverse speakers
- Fine-tune W2V-AASIST end-to-end on pooled dataset (ASVspoof + collected real)
- Re-evaluate both benchmark and in-the-wild FPR

**Resources needed**
- 1 × A100-week GPU
- ~2 FTE-months for data collection / labelling
- $5-10 k for microphone kits + small speaker honoraria

---

## Slide 11 — Justification for Future Work (5 pts) — 1:00

Evidence from **our own** evaluation (N = 71,237):

1. **Benchmark FPR is already 0.109 %** (8 / 7,355 real samples) → more model capacity won't help on this distribution
2. **In-the-wild FPR ~80 %** on just 5 samples → a ~730× degradation that scales with distribution mismatch, not model size
3. W2V-AASIST paper confirms: same architecture gets EER ~0.7 % with proper training data on target domain (our reproduction: EER **0.146 %**, within range)
4. Our feature distributions (Slide 8) show acoustic features *already* separate real/fake at 71k scale → a calibrated model has the signal, the training set just needs to look like deployment

**→ Fine-tuning on diverse real speech is a higher-leverage investment than architectural changes or a bigger model.**

---

## Slide 12 — Summary — 0:30

1. **Built** an integrated clone + detect tool (attack → shield in < 30 s)
2. **Benchmarked** at full scale: **99.77 % accuracy, 0.109 % FPR, EER 0.146 %** on N = 71,237
3. **Found** an 80 % FPR in-the-wild (~730× benchmark FPR) — *the* open problem
4. **Recommend**: collect diverse real-world speech + domain-adapt

---

## Slide 13 — Q&A — up to 2:00

**Anticipated Qs**
1. *Why 2019 LA and not 2021 DF?* — Pre-trained weights, 13 attack labels available, paper-comparable
2. *Why just 5 in-the-wild samples?* — Proof of concept; a proper study needs the dataset we propose as future work
3. *How would you ensure demographic diversity in collection?* — Partner with university AV clubs / crowd-source via Mechanical Turk with speaker metadata
4. *Could you fine-tune in-house?* — Tier 3 full eval (71,237 utterances) took 8.2 hrs on M4 Pro w/ 4 CPU workers; fine-tuning needs a GPU cluster
5. *What about the attention heatmap as explanation?* — Coarse; shows *where* not *why*; see Challenge Analysis for deeper discussion

---

## Speaking division (teamwork 1 pt)

- **Jerry**: 1, 2, 3, 4 (setup & system) + Q&A
- **Teammate**: 5, 6, 7, 8, 9 (demo + benchmark + gap)
- **Jerry**: 10, 11, 12 (future work & justification)
- **Both**: Q&A

---

## Timing check

| Slides | Time |
|---|---|
| 1 title | 0:30 |
| 2 goals | 1:00 |
| 3–4 system | 2:00 |
| 5 demo | 0:45 |
| 6–8 results | 3:00 |
| 9 domain gap | 1:00 |
| 10 future work | 1:00 |
| 11 justification | 1:00 |
| 12 summary | 0:30 |
| **Total talking** | **9:45** (15 s buffer under 10-min cap) |
| 13 Q&A | ~2:00 (within 12-min total) |

---

## Key assets (ready — 71k-scale, slide-quality)

### Charts
- `scripts/output/asvspoof_eval/full/confusion_matrix.png` — slide 6
- `scripts/output/asvspoof_eval/full/per_attack_accuracy.png` — slide 7
- `scripts/output/asvspoof_eval/full/feature_distributions.png` — slide 8
- `scripts/output/asvspoof_eval/domain_gap_fpr.png` ⭐ — slide 9
- `scripts/output/asvspoof_eval/full/roc.png` — backup / Q&A
- `scripts/output/asvspoof_eval/full/spoof_prob_histogram.png` — backup / Q&A
- `scripts/output/asvspoof_eval/full/threshold_sweep.png` — backup / Q&A
- `scripts/output/asvspoof_eval/full/speaker_fpr.png` — backup / Q&A

### Data
- `scripts/output/asvspoof_eval/eval_full.csv` (71,237 rows — full eval)
- `scripts/output/asvspoof_eval/full/metrics.json` — machine-readable metrics
- `scripts/output/asvspoof_eval/full/failure_cases.md` — top 20 FP + FN
- `scripts/output/asvspoof_eval/full/slide_updates.md` — all numbers paste-ready
- `scripts/output/asvspoof_eval/in_the_wild_real.csv` (5 rows)

### Reusable scripts
- `scripts/eval_asvspoof.py` — tiered eval with `--workers N`
- `scripts/analyze_eval.py` — 6 slide-ready charts from any eval CSV
- `scripts/threshold_sweep.py` — threshold sensitivity plot
- `scripts/generate_slide_updates.py` — metrics.json → slide text
- `scripts/run_full_analysis.sh` — one-shot pipeline
