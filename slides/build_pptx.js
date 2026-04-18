// Build VoiceGuard final-presentation pptx with chart inserts.
// Usage:  node build_pptx.js
// Output: "VoiceGuard - Final Presentation.pptx" in this folder.

const path = require("path");
const pptxgen = require("pptxgenjs");

// =======================================================
// Paths
// =======================================================
const CHARTS_ROOT = path.resolve(
  __dirname, "..", "scripts", "output", "asvspoof_eval"
);
const IMG = {
  confusion: path.join(CHARTS_ROOT, "full", "confusion_matrix.png"),
  per_attack: path.join(CHARTS_ROOT, "full", "per_attack_accuracy.png"),
  features: path.join(CHARTS_ROOT, "full", "feature_distributions.png"),
  domain_gap: path.join(CHARTS_ROOT, "domain_gap_fpr.png"),
  spoof_prob: path.join(CHARTS_ROOT, "full", "spoof_prob_histogram.png"),
  threshold_sweep: path.join(CHARTS_ROOT, "full", "threshold_sweep.png"),
};

// =======================================================
// Palette  (Midnight Executive + topic-specific Teal/Red)
// =======================================================
const C = {
  NAVY:    "1E2761",   // primary — dark authority
  NAVY_LT: "2C3E80",
  TEAL:    "2E8B8B",   // REAL / safe (matches chart colors)
  RED:     "E53E3E",   // FAKE / danger (matches chart colors)
  ORANGE:  "DD6B20",   // highlight / headline stats
  BG:      "FFFFFF",
  BG_SOFT: "F8FAFC",   // light content background
  INK:     "0F172A",   // body text
  MUTED:   "64748B",   // captions, secondary text
  LINE:    "E2E8F0",   // subtle dividers
  ACCENT:  "CADCFC",   // light blue ribbon
};

const FONT_HEAD = "Cambria";
const FONT_BODY = "Calibri";
const FONT_MONO = "Consolas";

// =======================================================
// Presentation setup
// =======================================================
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";   // 13.3" x 7.5"
pres.title = "VoiceGuard: Voice Cloning Attack & Detection";
pres.author = "Jerry Zeng";
pres.company = "14-795 Final Presentation";
const W = 13.3, H = 7.5;

// =======================================================
// Reusable helpers
// =======================================================
function addHeader(slide, title, tag) {
  // Left accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: 1.0, fill: { color: C.NAVY }, line: { color: C.NAVY },
  });
  // Title
  slide.addText(title, {
    x: 0.45, y: 0.28, w: W - 3.5, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 30, bold: true, color: C.NAVY,
    valign: "middle", margin: 0,
  });
  // Right-side tag (e.g., slide number / rubric area)
  if (tag) {
    slide.addText(tag, {
      x: W - 3.0, y: 0.35, w: 2.7, h: 0.5,
      fontFace: FONT_BODY, fontSize: 11, color: C.MUTED, italic: true,
      align: "right", valign: "middle", margin: 0,
    });
  }
  // Horizontal divider
  slide.addShape(pres.shapes.LINE, {
    x: 0.45, y: 1.05, w: W - 0.9, h: 0,
    line: { color: C.LINE, width: 1 },
  });
}

function addFooter(slide, n, total) {
  slide.addText(`VoiceGuard · ${n} / ${total}`, {
    x: 0.45, y: H - 0.35, w: W - 0.9, h: 0.25,
    fontFace: FONT_BODY, fontSize: 9, color: C.MUTED,
    align: "right", margin: 0,
  });
}

function statCallout(slide, x, y, w, h, value, label, color) {
  // Background
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.BG_SOFT },
    line: { color: C.LINE, width: 0.75 },
  });
  // Left accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.09, h,
    fill: { color: color || C.NAVY }, line: { color: color || C.NAVY },
  });
  slide.addText(value, {
    x: x + 0.18, y: y + 0.08, w: w - 0.25, h: h * 0.52,
    fontFace: FONT_HEAD, fontSize: 32, bold: true,
    color: color || C.NAVY, valign: "top", margin: 0,
  });
  slide.addText(label, {
    x: x + 0.18, y: y + h * 0.56, w: w - 0.25, h: h * 0.42,
    fontFace: FONT_BODY, fontSize: 11, color: C.MUTED,
    valign: "top", margin: 0,
  });
}

const TOTAL = 13;

// =======================================================
// SLIDE 1 — Title (dark)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.NAVY };

  // Top accent strip
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.15,
    fill: { color: C.RED }, line: { color: C.RED },
  });

  // Course tag
  s.addText("14-795  ·  FINAL PRESENTATION  ·  APR 2026", {
    x: 1.0, y: 1.8, w: W - 2, h: 0.4,
    fontFace: FONT_BODY, fontSize: 14, color: C.ACCENT,
    charSpacing: 4, bold: false, align: "left", margin: 0,
  });

  // Title
  s.addText("VoiceGuard", {
    x: 1.0, y: 2.3, w: W - 2, h: 1.3,
    fontFace: FONT_HEAD, fontSize: 72, bold: true, color: "FFFFFF",
    align: "left", margin: 0,
  });
  s.addText("Voice-Cloning Attack & Detection", {
    x: 1.0, y: 3.6, w: W - 2, h: 0.8,
    fontFace: FONT_HEAD, fontSize: 32, color: C.ACCENT, italic: true,
    align: "left", margin: 0,
  });

  // Byline
  s.addText([
    { text: "Jerry Zeng", options: { bold: true, color: "FFFFFF" } },
    { text: "  +  teammate", options: { color: C.ACCENT } },
  ], {
    x: 1.0, y: 5.6, w: W - 2, h: 0.4,
    fontFace: FONT_BODY, fontSize: 18, margin: 0,
  });

  // Small project tag
  s.addText("A Gradio app that both clones voices and catches them.", {
    x: 1.0, y: 6.1, w: W - 2, h: 0.35,
    fontFace: FONT_BODY, fontSize: 13, color: C.ACCENT, italic: true, margin: 0,
  });
}

// =======================================================
// SLIDE 2 — Goals (5 pts)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Goals", "Rubric: 5 pts");

  // Left column — the "what"
  s.addText("What we set out to build", {
    x: 0.5, y: 1.35, w: 6.5, h: 0.55,
    fontFace: FONT_HEAD, fontSize: 22, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText([
    { text: "A single tool where anyone can:", options: { bold: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Attack — clone a voice from 3–30 s of reference audio", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Shield — detect AI-generated speech with verdict + explanation", options: { bullet: true, breakLine: true, paraSpaceAfter: 18 } },
    { text: "Why now:", options: { bold: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Voice-cloning scams projected at $40 B by 2027", options: { bullet: true, breakLine: true, paraSpaceAfter: 6 } },
    { text: "No open tool pairs clone + detect in one interface", options: { bullet: true } },
  ], {
    x: 0.5, y: 1.95, w: 6.5, h: 4.5,
    fontFace: FONT_BODY, fontSize: 15, color: C.INK,
    valign: "top", margin: 0,
  });

  // Right column — refined goal
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.5, y: 1.35, w: 5.3, h: 4.2,
    fill: { color: C.BG_SOFT }, line: { color: C.NAVY, width: 1.25 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.5, y: 1.35, w: 0.12, h: 4.2,
    fill: { color: C.ORANGE }, line: { color: C.ORANGE },
  });
  s.addText("Refined goal after Challenge Analysis", {
    x: 7.78, y: 1.55, w: 5.0, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 18, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText([
    { text: "Not just \"does it work?\"", options: { italic: true, color: C.MUTED, breakLine: true, paraSpaceAfter: 14 } },
    { text: "→  \"is the detection calibrated and generalizable beyond the benchmark?\"", options: { bold: true, color: C.ORANGE } },
  ], {
    x: 7.78, y: 2.2, w: 4.95, h: 2.4,
    fontFace: FONT_BODY, fontSize: 16, color: C.INK,
    valign: "top", margin: 0,
  });

  s.addText("This framing drives every Results slide →", {
    x: 7.78, y: 4.8, w: 4.95, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: C.MUTED, italic: true,
    valign: "top", margin: 0,
  });

  addFooter(s, 2, TOTAL);
}

// =======================================================
// SLIDE 3 — System overview (5 pts)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "System — Two-Tab Gradio App", "Rubric: 5 pts");

  // Attack tab card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.35, w: 5.9, h: 5.3,
    fill: { color: C.BG_SOFT }, line: { color: C.RED, width: 1.5 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.35, w: 5.9, h: 0.48,
    fill: { color: C.RED }, line: { color: C.RED },
  });
  s.addText("ATTACK  ·  clone a voice", {
    x: 0.65, y: 1.35, w: 5.6, h: 0.48,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: "FFFFFF",
    valign: "middle", charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "OpenVoice v2 (MyShell)", options: { bold: true, color: C.NAVY, fontSize: 16, breakLine: true, paraSpaceAfter: 6 } },
    { text: "Speaker embedding", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "MeloTTS base waveform", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "Tone-color conversion", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Inputs:", options: { bold: true, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "3–30 s reference audio + free text", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Outputs:", options: { bold: true, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Cloned .wav  +  similarity score (0–100 %)", options: { fontSize: 13 } },
  ], {
    x: 0.8, y: 2.0, w: 5.3, h: 4.4,
    fontFace: FONT_BODY, fontSize: 14, color: C.INK,
    valign: "top", margin: 0,
  });

  // Arrow between tabs — big visible chevron
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.5, y: 3.8, w: 0.3, h: 0.5,
    fill: { color: C.NAVY }, line: { color: C.NAVY },
  });
  s.addText("→", {
    x: 6.45, y: 3.8, w: 0.4, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 30, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("Send to Shield", {
    x: 6.3, y: 4.4, w: 0.7, h: 0.3,
    fontFace: FONT_BODY, fontSize: 9, color: C.MUTED, italic: true,
    align: "center", valign: "top", margin: 0,
  });

  // Shield tab card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: 1.35, w: 5.9, h: 5.3,
    fill: { color: C.BG_SOFT }, line: { color: C.TEAL, width: 1.5 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.9, y: 1.35, w: 5.9, h: 0.48,
    fill: { color: C.TEAL }, line: { color: C.TEAL },
  });
  s.addText("SHIELD  ·  detect deepfake", {
    x: 7.05, y: 1.35, w: 5.6, h: 0.48,
    fontFace: FONT_HEAD, fontSize: 16, bold: true, color: "FFFFFF",
    valign: "middle", charSpacing: 2, margin: 0,
  });
  s.addText([
    { text: "W2V-AASIST (Tak et al. 2022)", options: { bold: true, color: C.NAVY, fontSize: 16, breakLine: true, paraSpaceAfter: 6 } },
    { text: "wav2vec 2.0 XLSR (300 M params)", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "AASIST graph-attention backend", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "Softmax → spoof_prob", options: { bullet: true, fontSize: 13, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Outputs:", options: { bold: true, color: C.MUTED, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Verdict: REAL / SUSPICIOUS / FAKE", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "Confidence (0 – 100 %)", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "8-feature acoustic dashboard", options: { fontSize: 13, breakLine: true, paraSpaceAfter: 2 } },
    { text: "Attention heatmap + spectrogram", options: { fontSize: 13 } },
  ], {
    x: 7.2, y: 2.0, w: 5.3, h: 4.4,
    fontFace: FONT_BODY, fontSize: 14, color: C.INK,
    valign: "top", margin: 0,
  });

  s.addText("Live demo end-to-end in  < 30 s", {
    x: 0.5, y: 6.8, w: 12.3, h: 0.35,
    fontFace: FONT_BODY, fontSize: 13, color: C.ORANGE, bold: true, italic: true,
    align: "center", valign: "top", margin: 0,
  });

  addFooter(s, 3, TOTAL);
}

// =======================================================
// SLIDE 4 — Replicated vs Added (5 pts)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Replicated  vs.  Added", "Rubric: 5 pts (System)");

  const headerOpts = {
    bold: true, color: "FFFFFF", fill: { color: C.NAVY },
    align: "left", valign: "middle", fontFace: FONT_BODY, fontSize: 14,
  };
  const cellBase = { fontFace: FONT_BODY, fontSize: 12.5, color: C.INK, valign: "middle" };

  const addedCell = (txt) => ({
    text: txt,
    options: { bold: true, color: C.TEAL, ...cellBase },
  });

  const rows = [
    [{ text: "Component", options: headerOpts },
     { text: "Source", options: headerOpts },
     { text: "Our change", options: headerOpts }],

    [{ text: "OpenVoice v2", options: cellBase },
     { text: "MyShell (MIT)", options: cellBase },
     { text: "used as-is", options: { ...cellBase, color: C.MUTED, italic: true } }],

    [{ text: "W2V-AASIST  +  best_SSL_model_DF.pth", options: cellBase },
     { text: "Tak et al. (MIT)", options: cellBase },
     { text: "used pretrained weights", options: { ...cellBase, color: C.MUTED, italic: true } }],

    [{ text: "Gradio two-tab app", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("built from scratch")],

    [{ text: "\"Send to Shield\" workflow", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("built from scratch")],

    [{ text: "AASIST attention heatmap overlay", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("attention × mel spectrogram")],

    [{ text: "8-feature acoustic dashboard", options: cellBase },
     { text: "librosa", options: cellBase },
     addedCell("dashboard  +  UI integration")],

    [{ text: "Dataset-wide eval pipeline", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("scripts/eval_asvspoof.py (71 k eval)")],

    [{ text: "Feature-distribution calibration", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("scripts/analyze_eval.py")],

    [{ text: "In-the-wild domain-gap test", options: cellBase },
     { text: "—", options: { ...cellBase, color: C.MUTED } },
     addedCell("scripts/in_the_wild_test.py")],
  ];

  s.addTable(rows, {
    x: 0.5, y: 1.4, w: W - 1.0, h: 5.3,
    colW: [4.5, 2.8, 5.0],
    rowH: 0.5,
    border: { type: "solid", pt: 0.75, color: C.LINE },
    fontFace: FONT_BODY, fontSize: 12.5, color: C.INK,
  });

  addFooter(s, 4, TOTAL);
}

// =======================================================
// SLIDE 5 — Live demo (Results, part 1)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Demo  —  Attack  →  Shield", "Rubric: 5 pts (Results)");

  // Step cards
  const steps = [
    { t: "1.  Upload", d: "10-second\nreference clip", color: C.NAVY },
    { t: "2.  Clone", d: "OpenVoice v2\ngenerates cloned .wav", color: C.NAVY },
    { t: "3.  Send to Shield", d: "single-click\nhand-off", color: C.NAVY_LT },
    { t: "4.  Verdict", d: "FAKE\n@  99.97 %", color: C.RED },
  ];
  steps.forEach((step, i) => {
    const x = 0.5 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.5, w: 3.0, h: 2.2,
      fill: { color: C.BG_SOFT }, line: { color: step.color, width: 1.5 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.5, w: 3.0, h: 0.5,
      fill: { color: step.color }, line: { color: step.color },
    });
    s.addText(step.t, {
      x: x + 0.15, y: 1.5, w: 2.7, h: 0.5,
      fontFace: FONT_HEAD, fontSize: 15, bold: true, color: "FFFFFF",
      valign: "middle", margin: 0,
    });
    s.addText(step.d, {
      x: x + 0.15, y: 2.05, w: 2.7, h: 1.55,
      fontFace: FONT_BODY, fontSize: 14, color: C.INK,
      align: "center", valign: "middle", margin: 0,
      bold: i === 3,
    });
  });

  // Bottom demo-result callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.0, w: W - 1.0, h: 2.0,
    fill: { color: C.NAVY }, line: { color: C.NAVY },
  });
  s.addText("LIVE DEMO  ·  end-to-end  <  30 s", {
    x: 0.75, y: 4.15, w: W - 1.5, h: 0.45,
    fontFace: FONT_BODY, fontSize: 14, color: C.ACCENT,
    charSpacing: 3, valign: "top", margin: 0,
  });
  s.addText([
    { text: "Attack:  ", options: { bold: true, color: "FFFFFF" } },
    { text: "Upload → Clone & Speak → cloned audio + similarity score", options: { color: C.ACCENT, breakLine: true, paraSpaceAfter: 12 } },
    { text: "Shield:  ", options: { bold: true, color: "FFFFFF" } },
    { text: "Send to Shield → verdict  FAKE @ 99.97 %  + attention heatmap highlighting artifact bands", options: { color: C.ACCENT } },
  ], {
    x: 0.75, y: 4.7, w: W - 1.5, h: 1.2,
    fontFace: FONT_BODY, fontSize: 14,
    valign: "top", margin: 0,
  });

  s.addText("Demo recording available  ·  fallback if live demo fails", {
    x: 0.5, y: 6.2, w: W - 1.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 11, color: C.MUTED, italic: true,
    align: "center", valign: "top", margin: 0,
  });

  addFooter(s, 5, TOTAL);
}

// =======================================================
// SLIDE 6 — Benchmark results (Results, part 2)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Results  —  ASVspoof 2019 LA  (full eval)", "Rubric: 5 pts (Results)");

  // Confusion matrix on left
  s.addImage({
    path: IMG.confusion,
    x: 0.5, y: 1.4, w: 7.8, h: 5.4,
  });

  // Metric callouts on right
  statCallout(s, 8.6,  1.4, 4.2, 1.05, "0.146 %", "EER  (paper reports ~0.7 %)", C.NAVY);
  statCallout(s, 8.6,  2.6, 4.2, 1.05, "99.77 %", "Overall Accuracy", C.NAVY);
  statCallout(s, 8.6,  3.8, 4.2, 1.05, "0.109 %", "FPR   8 / 7,355 bonafide", C.TEAL);
  statCallout(s, 8.6,  5.0, 4.2, 1.05, "0.244 %", "FNR   156 / 63,882 spoof", C.RED);

  s.addText("N  =  71,237  ·  7,355 bonafide  +  63,882 spoof  ·  8 h 14 min on 4 CPU workers", {
    x: 0.5, y: 6.85, w: W - 1.0, h: 0.35,
    fontFace: FONT_BODY, fontSize: 11, color: C.MUTED, italic: true,
    align: "center", margin: 0,
  });

  addFooter(s, 6, TOTAL);
}

// =======================================================
// SLIDE 7 — Per-attack breakdown (Results, part 3)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Results  —  13 Deepfake Attack Types", "Rubric: 5 pts (Results)");

  // Per-attack chart
  s.addImage({
    path: IMG.per_attack,
    x: 0.4, y: 1.3, w: 8.6, h: 4.3,
  });

  // Right takeaway box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.2, y: 1.4, w: 3.7, h: 5.3,
    fill: { color: C.BG_SOFT }, line: { color: C.NAVY, width: 1.25 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.2, y: 1.4, w: 0.09, h: 5.3,
    fill: { color: C.ORANGE }, line: { color: C.ORANGE },
  });
  s.addText("Takeaways", {
    x: 9.4, y: 1.55, w: 3.4, h: 0.45,
    fontFace: FONT_HEAD, fontSize: 17, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText([
    { text: "7 / 13  attacks detected at 100.0 %", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Hardest: A10 (98.5 %) — end-to-end TTS w/ WaveNet vocoder", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Runner-up: A11 (99.4 %) WaveRNN, A19 (99.6 %) transfer TTS+VC", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Mean detection: 99.8 % — no attack is a systemic blind spot", options: { bullet: true, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Each bar ≈ 4,914 utterances", options: { bullet: true, color: C.MUTED, fontSize: 11, italic: true } },
  ], {
    x: 9.4, y: 2.1, w: 3.4, h: 4.5,
    fontFace: FONT_BODY, fontSize: 12.5, color: C.INK,
    valign: "top", margin: 0,
  });

  // Bottom caption
  s.addText("Detection generalizes across TTS, VC, and neural-vocoder synthesis methods at scale.", {
    x: 0.5, y: 5.75, w: 8.5, h: 0.35,
    fontFace: FONT_BODY, fontSize: 12, color: C.MUTED, italic: true, margin: 0,
  });

  addFooter(s, 7, TOTAL);
}

// =======================================================
// SLIDE 8 — Feature calibration at scale (Results, part 4)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Results  —  Feature Calibration at Scale", "Rubric: 5 pts (Results)");

  // Feature distribution image — wide so landscape layout
  s.addImage({
    path: IMG.features,
    x: 0.4, y: 1.25, w: 9.5, h: 4.8,
  });

  // Right info column — "before / after"
  statCallout(s, 10.1, 1.3, 2.9, 1.2, "14", "samples before\n(Challenge Analysis)", C.MUTED);
  statCallout(s, 10.1, 2.7, 2.9, 1.2, "71,237", "samples now\nNearly 5,000× more", C.ORANGE);

  s.addShape(pres.shapes.RECTANGLE, {
    x: 10.1, y: 4.2, w: 2.9, h: 1.85,
    fill: { color: C.BG_SOFT }, line: { color: C.NAVY, width: 1.0 },
  });
  s.addText("Answers Challenge-Analysis weakness #2", {
    x: 10.22, y: 4.3, w: 2.7, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 12, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText("Shimmer, F0 CV, Energy CV separate cleanly — ready for UI percentile context.", {
    x: 10.22, y: 5.0, w: 2.7, h: 1.0,
    fontFace: FONT_BODY, fontSize: 11.5, color: C.INK, italic: true,
    valign: "top", margin: 0,
  });

  s.addText("Violin plots: black line = median; colored region = kernel density.", {
    x: 0.5, y: 6.15, w: 9.3, h: 0.35,
    fontFace: FONT_BODY, fontSize: 10.5, color: C.MUTED, italic: true, margin: 0,
  });

  addFooter(s, 8, TOTAL);
}

// =======================================================
// SLIDE 9 — THE GAP WE FOUND (headline)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "The Gap We Found", "⭐ headline finding");

  // Left — big callouts showing FPR delta
  statCallout(s, 0.5, 1.4, 5.0, 1.6, "0.109 %", "FPR on ASVspoof 2019 LA\nstudio-quality, 7,355 real utterances", C.TEAL);
  statCallout(s, 0.5, 3.2, 5.0, 1.6, "80.0 %", "FPR on in-the-wild real humans\nphone / laptop mic, n = 5", C.RED);

  // The 730x stat — big & loud
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.0, w: 5.0, h: 1.7,
    fill: { color: C.NAVY }, line: { color: C.NAVY },
  });
  s.addText("~ 730×", {
    x: 0.6, y: 5.1, w: 4.8, h: 1.0,
    fontFace: FONT_HEAD, fontSize: 56, bold: true, color: C.ORANGE,
    align: "center", valign: "middle", margin: 0,
  });
  s.addText("FPR increase from benchmark → wild", {
    x: 0.6, y: 6.0, w: 4.8, h: 0.6,
    fontFace: FONT_BODY, fontSize: 14, color: "FFFFFF", italic: true,
    align: "center", valign: "middle", margin: 0,
  });

  // Right — image (domain gap bar chart)
  s.addImage({
    path: IMG.domain_gap,
    x: 5.9, y: 1.4, w: 7.1, h: 4.8,
  });

  // Sub-caption below chart
  s.addText("On its own benchmark, the model is near-perfect.  On everyday audio, 4 / 5 real humans are mis-flagged as AI.", {
    x: 5.9, y: 6.3, w: 7.1, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, color: C.INK, italic: true,
    align: "center", valign: "middle", margin: 0,
  });

  addFooter(s, 9, TOTAL);
}

// =======================================================
// SLIDE 10 — Future Work (5 pts)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Future Work  —  one step", "Rubric: 5 pts");

  // Big banner with the ONE step
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.4, w: W - 1.0, h: 1.3,
    fill: { color: C.NAVY }, line: { color: C.NAVY },
  });
  s.addText("Domain-adaptive fine-tuning  on diverse real-world speech", {
    x: 0.65, y: 1.45, w: W - 1.3, h: 1.2,
    fontFace: FONT_HEAD, fontSize: 26, bold: true, color: "FFFFFF",
    align: "center", valign: "middle", margin: 0,
  });

  // Two columns: Data + Compute
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.0, w: 6.2, h: 3.8,
    fill: { color: C.BG_SOFT }, line: { color: C.TEAL, width: 1.25 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.0, w: 0.1, h: 3.8,
    fill: { color: C.TEAL }, line: { color: C.TEAL },
  });
  s.addText("Data  —  collect ≥ 1,000 hrs real human audio", {
    x: 0.75, y: 3.15, w: 5.8, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 17, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText([
    { text: "Phone codecs  (µ-law, Opus, GSM)", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Consumer mics  (laptop, earbuds, headset)", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Noisy conditions  (café, car, wind)", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "50+ demographically diverse speakers", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Pool with ASVspoof, fine-tune end-to-end", options: { bullet: true } },
  ], {
    x: 0.9, y: 3.7, w: 5.6, h: 3.0,
    fontFace: FONT_BODY, fontSize: 13.5, color: C.INK,
    valign: "top", margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.1, y: 3.0, w: 5.7, h: 3.8,
    fill: { color: C.BG_SOFT }, line: { color: C.ORANGE, width: 1.25 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.1, y: 3.0, w: 0.1, h: 3.8,
    fill: { color: C.ORANGE }, line: { color: C.ORANGE },
  });
  s.addText("Resources — think big", {
    x: 7.35, y: 3.15, w: 5.3, h: 0.5,
    fontFace: FONT_HEAD, fontSize: 17, bold: true, color: C.NAVY,
    valign: "top", margin: 0,
  });
  s.addText([
    { text: "1 × A100-week  GPU for fine-tuning", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "~ 2 FTE-months  for data collection + labelling", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "$5–10 k  microphones + speaker honoraria", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Re-evaluate on both benchmark and new in-the-wild set", options: { bullet: true } },
  ], {
    x: 7.5, y: 3.7, w: 5.1, h: 3.0,
    fontFace: FONT_BODY, fontSize: 13.5, color: C.INK,
    valign: "top", margin: 0,
  });

  addFooter(s, 10, TOTAL);
}

// =======================================================
// SLIDE 11 — Justification for Future Work (5 pts)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Justification  —  evidence from our eval", "Rubric: 5 pts");

  // Four evidence cards
  const cards = [
    {
      n: "1", t: "Benchmark FPR is already near-zero",
      d: "0.109 %  (8 / 7,355 real).  A bigger / deeper model\nwon't help when you're already at the floor.",
      color: C.TEAL,
    },
    {
      n: "2", t: "In-the-wild FPR is catastrophic",
      d: "80 %  (4 / 5 real humans).  A ~730× degradation that\nscales with distribution mismatch, not model size.",
      color: C.RED,
    },
    {
      n: "3", t: "The architecture is fine — data isn't",
      d: "W2V-AASIST paper: EER ~0.7 % when train ≈ test.\nOur reproduction:  EER 0.146 %  — within range.",
      color: C.NAVY,
    },
    {
      n: "4", t: "Features already separate real / fake",
      d: "At 71 k scale, shimmer / F0-CV / energy-CV\ndistributions split cleanly → signal exists, domain doesn't.",
      color: C.ORANGE,
    },
  ];
  cards.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.5 + col * 6.4;
    const y = 1.4 + row * 2.65;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 6.2, h: 2.45,
      fill: { color: C.BG_SOFT }, line: { color: C.LINE, width: 0.75 },
    });
    // Number tab
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.85, h: 2.45,
      fill: { color: c.color }, line: { color: c.color },
    });
    s.addText(c.n, {
      x, y, w: 0.85, h: 2.45,
      fontFace: FONT_HEAD, fontSize: 44, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(c.t, {
      x: x + 1.0, y: y + 0.25, w: 5.1, h: 0.6,
      fontFace: FONT_HEAD, fontSize: 16, bold: true, color: C.NAVY,
      valign: "top", margin: 0,
    });
    s.addText(c.d, {
      x: x + 1.0, y: y + 0.95, w: 5.1, h: 1.4,
      fontFace: FONT_BODY, fontSize: 13, color: C.INK,
      valign: "top", margin: 0,
    });
  });

  s.addText("→  Fine-tuning on diverse real speech is a higher-leverage investment than architectural changes or a bigger model.", {
    x: 0.5, y: 6.75, w: W - 1.0, h: 0.4,
    fontFace: FONT_BODY, fontSize: 13.5, color: C.ORANGE, bold: true, italic: true,
    align: "center", margin: 0,
  });

  addFooter(s, 11, TOTAL);
}

// =======================================================
// SLIDE 12 — Summary
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.BG };
  addHeader(s, "Summary", "");

  const items = [
    { n: "Built", d: "an integrated clone + detect tool  —  attack → shield in  < 30 s", color: C.NAVY },
    { n: "Benchmarked", d: "at full scale:  99.77 % accuracy,  0.109 % FPR,  EER  0.146 %   (N = 71,237)", color: C.TEAL },
    { n: "Found", d: "80 % FPR on in-the-wild real humans  —  ~730× benchmark FPR", color: C.RED },
    { n: "Recommend", d: "collect diverse real-world speech  +  domain-adapt fine-tune", color: C.ORANGE },
  ];
  items.forEach((it, i) => {
    const y = 1.6 + i * 1.28;
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: 0.8, y, w: 1.0, h: 1.0,
      fill: { color: it.color }, line: { color: it.color },
    });
    s.addText(`${i+1}`, {
      x: 0.8, y, w: 1.0, h: 1.0,
      fontFace: FONT_HEAD, fontSize: 36, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    // Bold label
    s.addText(it.n, {
      x: 2.0, y: y + 0.1, w: 2.3, h: 0.45,
      fontFace: FONT_HEAD, fontSize: 22, bold: true, color: it.color,
      valign: "top", margin: 0,
    });
    // Description
    s.addText(it.d, {
      x: 2.0, y: y + 0.55, w: W - 2.5, h: 0.5,
      fontFace: FONT_BODY, fontSize: 15, color: C.INK,
      valign: "top", margin: 0,
    });
  });

  addFooter(s, 12, TOTAL);
}

// =======================================================
// SLIDE 13 — Q&A (dark)
// =======================================================
{
  const s = pres.addSlide();
  s.background = { color: C.NAVY };

  // Top accent
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.15,
    fill: { color: C.RED }, line: { color: C.RED },
  });

  s.addText("Q&A", {
    x: 1.0, y: 2.4, w: W - 2, h: 1.8,
    fontFace: FONT_HEAD, fontSize: 120, bold: true, color: "FFFFFF",
    align: "left", valign: "middle", margin: 0,
  });
  s.addText("Thanks for listening.", {
    x: 1.0, y: 4.3, w: W - 2, h: 0.7,
    fontFace: FONT_HEAD, fontSize: 26, color: C.ACCENT, italic: true,
    align: "left", margin: 0,
  });

  // Small anticipated-Q hint at bottom
  s.addText([
    { text: "Anticipated:  ", options: { color: C.ACCENT, italic: true } },
    { text: "why 2019 LA? · 5 in-the-wild samples? · fine-tune in-house? · heatmap as explanation?", options: { color: "FFFFFF", fontSize: 12 } },
  ], {
    x: 1.0, y: 6.4, w: W - 2, h: 0.5,
    fontFace: FONT_BODY, fontSize: 12, margin: 0,
  });
}

// =======================================================
// Write file
// =======================================================
const outPath = path.join(__dirname, "VoiceGuard - Final Presentation.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved:", outPath);
});
