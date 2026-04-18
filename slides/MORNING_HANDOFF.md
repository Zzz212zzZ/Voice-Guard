# 早安 ☀️ — Final Presentation 弹药盘点 (2026-04-18 05:01)

## TL;DR

**Tier 3 完成**. 8h 14min 跑完 71,237 utterances. 所有 71k 规模的 slide-ready 图表都生成了. Slide draft 里所有 "N=2,800" 相关数字已自动替换成 Tier 3 数字.

你只要做三件事：
1. 打开 Google Slides 模板
2. 按 [final_presentation_draft.md](final_presentation_draft.md) 一页一页贴内容
3. 把 `scripts/output/asvspoof_eval/full/*.png` 里的图 drag 进对应 slide

---

## 核心数字 (记这个讲台上能秒答)

| 维度 | 数值 |
|---|---|
| **Benchmark EER** | **0.146 %** (paper reports ~0.7%，我们的 reproduction 更好) |
| Benchmark Accuracy | 99.77 % |
| **Benchmark FPR (studio)** | **0.109 %** (8 / 7,355 real utt) |
| **In-the-wild FPR** | **80.0 %** (4 / 5 real humans 误判为 FAKE) |
| **Domain Gap** | ~730× 倍放大 |
| 最弱攻击 | A10 (WaveNet TTS) — 98.5 % |
| 最强攻击 | 7 种 tied at 100.0 % |

---

## 8 张 slide-ready 图（所有都在 `scripts/output/asvspoof_eval/full/`）

| 文件 | 用在哪 slide | 说什么 |
|---|---|---|
| `confusion_matrix.png` | Slide 6 | 2×3 混淆矩阵，7347+3+5 / 156+36+63690 |
| `per_attack_accuracy.png` | Slide 7 | 13 种 attack 的检测率，突出 A10 |
| `feature_distributions.png` | Slide 8 | 8 特征 violin plot，N=71k vs 14 |
| `../domain_gap_fpr.png` ⭐ | Slide 9 | **headline 图**，0.1% vs 80% |
| `roc.png` | Q&A backup | EER 点高亮 |
| `spoof_prob_histogram.png` | Q&A backup | 模型 bimodal confidence |
| `threshold_sweep.png` | Q&A backup | 显示阈值在很大范围内等效 |
| `speaker_fpr.png` | Q&A backup | 按 speaker_id 的 FPR breakdown |

---

## 推荐的工作流程

1. **打开 slide 模板** (https://docs.google.com/presentation/d/1-dKgbBW8mtcXbl_cRTfTAktN1V4PFkcR/)
2. **按模板顺序填 12 张内容 slide**，每张的内容在 [final_presentation_draft.md](final_presentation_draft.md) 里已经写好
3. **插图**: Finder 打开 `scripts/output/asvspoof_eval/full/`，drag png 到 slide
4. **预演时间**: 目标 10:00，slide draft 里第 197-210 行的 timing table 是按 9:45 设计的（留 15s buffer）

---

## 如果有时间再做的优化（优先级从高到低）

1. ✏️ **Teammate 名字补上** (title slide + teamwork split 部分用 `+ teammate` 占位)
2. 🎥 **demo 录制** — Attack → Send to Shield → FAKE verdict (<30s) 录好插到 Slide 5，比现场跑稳
3. 🔊 **在 Q&A 可能会被问**: "A10 为什么最弱?" → A10 是 end-to-end TTS with WaveNet vocoder, WaveNet 产生的 artifacts 比其他 vocoder 更接近自然 speech
4. 🎯 **demo 的 reference audio 选短且清晰** — OpenVoice v2 越好的 reference, similarity score 越高, demo 效果越震撼
5. 🏁 **最后一次 rehearsal** — 录下来自己听，修正语速 / 口头禅

---

## 后台跑完的其他资产

- **metrics.json** — machine-readable 所有数字，可以直接 `jq` 查
- **slide_updates.md** — 如果你想自己另写 slide，这是所有 paste-ready 文字
- **failure_cases.md** — top 20 false positives + top 20 false negatives (utt_id, spoof_prob, features)

---

## 脚本备忘（将来想 reproduce 或扩展）

```bash
# 重跑 analysis (改了 CSV 或脚本之后)
bash scripts/run_full_analysis.sh

# 单独跑一个 tier
python scripts/analyze_eval.py --tier full

# 如果想删 log + tmp (你之前说过可以)
rm scripts/output/asvspoof_eval/tier3.log
```

---

## 今天的时间建议

- **上午**: 填 Google Slides (~1-2 hr)，包括贴图 + 校对数字
- **中午前**: 第一次走一遍 timing，看会不会超 10 min
- **下午**: Demo 录制 + 走第二次
- **晚上**: 最后 rehearsal + 把 teammate 部分 align 清楚

Final presentation 今天就能交给我的话我可以帮你再 review 一遍.

GL 🎯
