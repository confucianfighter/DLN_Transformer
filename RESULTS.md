# Experiment Results

## Hymba-Like Shakespeare Repeat Task

Task setup:
- Data: `data/tiny_shakespeare.txt`, sentence-fragment repeat.
- Input sequence length: 256.
- Source fragment: positions `0:64`.
- Blank bridge: positions `64:191`.
- Marker/recall region: final 65 positions.
- Target: reproduce the original 64-character fragment in the final 64 positions.
- Metrics: recall accuracy over final 64 characters, exact accuracy over full 64-character recall.

Architecture baseline:
- Hymba-like model with parallel Mamba + causal RoPE attention.
- 8 layers, GQA-style heads, 16 meta tokens, local attention window 128, full attention layers `1,4,8`.
- Character tokenization, one token per character.

| Run | Dim | Schedule | Interspersion | Epochs | Best Recall | Best Exact | Notes |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `runs_hymba_text_repeat_dense_d32_l8_e40` | 32 | dense | none | stopped at 21 | 97.86% | 25.20% | Dense control was stopped early at epoch 21. |
| `runs_hymba_text_repeat_mr11244211_d32_l8_e40` | 32 | `1,1,2,4,4,2,1,1` | none | 40 | 97.88% | 27.90% | Best at epoch 38; final dipped. Active ratio reported as 0.730. |
| `runs_hymba_text_repeat_mr11244211_d32_l8_selfmix_e60` | 32 | `1,1,2,4,4,2,1,1` | full sequence | 41 | 97.88% carried | 27.90% carried | Failed immediately; epoch 41 clean recall collapsed to 18.65%. Stopped. |
| `runs_hymba_text_repeat_mr11244211_d32_l8_blankmix_e60` | 32 | `1,1,2,4,4,2,1,1` | blank bridge only | 60 | 99.36% | 69.50% | Continued from compressed epoch 40. Best/final at epoch 60. |
| `runs_hymba_text_repeat_mr11244211_d64_l8_e40` | 64 | `1,1,2,4,4,2,1,1` | none | stopped at 8 | 93.22% | 0.90% | Initial no-dropout probe; stopped to restart with robustness settings. |
| `runs_hymba_text_repeat_mr11244211_d64_l8_robust_e40` | 64 | `1,1,2,4,4,2,1,1` | none | stopped at 7 | 18.65% | 0.00% | Dropout 0.05 + weight decay 1e-4 + label smoothing 0.01 stalled at floor. |
| `runs_hymba_text_repeat_mr11244211_d64_l8_lightrobust_e40` | 64 | `1,1,2,4,4,2,1,1` | none | stopped at 34 checkpoint | 99.78% | 87.90% | Dropout 0.03 + weight decay 1e-5, no label smoothing. Epoch 35 had started but checkpoint was epoch 34. |
| `runs_hymba_text_repeat_mr11244211_d64_l8_blankmix_e54` | 64 | `1,1,2,4,4,2,1,1` | blank bridge only | stopped at 39 checkpoint | 99.81% | 88.80% | Continued from d64 lightrobust checkpoint. Slow improvement only, not a fast climb. |
| `runs_hymba_text_repeat_dense_d64_l8_lightrobust_e40` | 64 | dense | none | stopped at 21 checkpoint | 98.63% | 43.50% | Matched light robustness. Much slower and plateaued far below compressed exact. |
| `runs_hymba_text_repeat_mr12488421_d64_l8_lightrobust_e40` | 64 | `1,2,4,8,8,4,2,1` | none | stopped at 24 checkpoint | 98.10% | 30.20% | Active ratio 0.498. Learned cleanly but exact plateaued low. |
| `runs_hymba_text_repeat_mr12488421_d64_l8_blankmix_e34` | 64 | `1,2,4,8,8,4,2,1` | blank bridge only | 34 | 98.10% | 30.90% | Blank-mix did not unlock the 8x bottleneck. |
| `runs_hymba_text_repeat_mr1124444211_d64_l10_lightrobust_e40` | 64 | `1,1,2,4,4,4,4,2,1,1` | none | stopped at 18 checkpoint | 99.26% | 65.60% | 10-layer 4x-depth control. Positive but below 8-layer 4x best. |
| `runs_hymba_text_repeat_mr1124884211_d64_l10_surgery_train5to10_e54` | 64 | `1,1,2,4,8,8,4,2,1,1` | none | stopped at 41 checkpoint | 99.70% | 83.30% | Initialized from 8-layer 4x model, inserted 8x middle layers, froze layers 1-4 and trained 5-10. |
| `runs_hymba_text_repeat_mr1124884211_d64_l10_surgery_fullfinetune_e54` | 64 | `1,1,2,4,8,8,4,2,1,1` | none | 54 | 99.84% | 90.70% | Full fine-tune after staged 8x insertion. Best result so far. |

Current lighter robustness recipe for 64-dim runs:
- Dropout: `0.03`.
- Weight decay: `1e-5`.
- Label smoothing: `0.0`.
- Gradient clipping: `1.0`.
- Clean no-interspersion baseline first, then blank-bridge interspersion continuation.

Interspersion recipe that worked:
- Resume from the no-interspersion compressed checkpoint.
- Each epoch, run a frozen prediction pass over the original training examples.
- Replace only blank-bridge input tokens, positions `64:191`.
- Interval: every roughly 8 positions.
- Span: 1-2 sampled generated tokens.
- Insert probability: `0.0`, so replacement only; no sequence shifting.
- Top-k sampling: `5`, temperature `0.8`.
- Generated/self tokens are context only: their targets are set to `-100` and ignored by cross-entropy.
- Clean evaluation stays on the original uncorrupted repeat task.

Negative interspersion finding:
- Corrupting the full sequence is invalid for this repeat task because it can destroy source memory tokens at positions `0:64`.
- That version collapsed to the blank/majority floor after one epoch.

64-dim wall-clock comparison:
- Compressed no-interspersion averaged `57.5s/epoch`.
- Dense control averaged roughly `151s/epoch`.
- Compressed reached `40%` exact at epoch 12 in `686.6s`.
- Dense reached `40.2%` exact at epoch 13 in `1964.8s`.
- Compressed reached `80%` exact at epoch 19 in `1089.2s`; dense had not reached `50%` by epoch 21.

Qualitative "beer scale" for repeat samples:
- `0-1 beers`: tiny one-character slips; still reads essentially exact.
- `2-3 beers`: phonetic drift or proper-name wobble, but clearly intelligible.
- `4-6 beers`: syntactically plausible Shakespeare that is noticeably warped.
- `7+ beers`: slurry, blanks, or low-information mush.
- `12 Pack Shakespeare`: total collapse; recognizably trying to be Shakespeare, but no longer reliably speaking English.
- Useful shorthand during run triage:
  `cap480` recovered to roughly `0-1 beers`, early `cap320` was more like `4-7 beers`, and early `mem512` from `cap480` looked closer to `2-4 beers`.
