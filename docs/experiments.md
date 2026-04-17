# Experiments

## 2026-04-17: First Bottleneck Comparison

### Goal

Test whether the first `4 -> 1 -> 4` compression bottleneck is viable, and whether its gain survives a roughly matched parameter comparison against a no-bottleneck baseline.

### Shared setup

- Dataset: `data/tiny_shakespeare.txt`
- Tokenizer: lowercase 7-position positional character tokenizer
- Vocab size: `192`
- Embedding/model width: `256`
- Positional scheme: sequential RoPE + continuous newline-count RoPE + line-position features in reserved free dims
- Training budget: `2000` steps
- Batch size: `16`
- Sequence length: `256`
- Device: `cuda` on GTX 980 Ti

### Runs

#### Baseline A: Small no-bottleneck baseline

Command:

```powershell
python train.py --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --n-pre-layers 2 --n-bottleneck-layers 0 --n-post-layers 2 --compression-factor 1 --device cuda --out data\baseline_2k.pt
```

Result:

- Parameters: `3,524,864`
- Final validation loss: `1.5857`
- Final perplexity: `4.88`

#### Bottleneck B: First compression model

Command:

```powershell
python train.py --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --n-pre-layers 2 --n-bottleneck-layers 2 --n-post-layers 2 --compression-factor 4 --device cuda --out data\bottleneck_2k.pt
```

Result:

- Parameters: `5,755,136`
- Final validation loss: `0.4034`
- Final perplexity: `1.50`

#### Baseline C: Parameter-matched no-bottleneck baseline

Command:

```powershell
python train.py --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --n-pre-layers 3 --n-bottleneck-layers 0 --n-post-layers 3 --mlp-hidden-dim 896 --compression-factor 1 --device cuda --out data\baseline_matched_2k.pt
```

Result:

- Parameters: `5,819,648`
- Final validation loss: `1.5549`
- Final perplexity: `4.73`

#### Ablation D: Compress/expand only

Command:

```powershell
python train.py --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --n-pre-layers 2 --n-bottleneck-layers 0 --n-post-layers 2 --compression-factor 4 --device cuda --out data\compress_expand_only_2k.pt
```

Result:

- Parameters: `4,050,176`
- Final validation loss: `0.4099`
- Final perplexity: `1.51`

#### Multirate E: Shared-weight `4 -> 2 -> 1 -> 2 -> 4`

Command:

```powershell
python train.py --architecture multirate --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --fine-down-cycles 4 --mid-down-cycles 2 --core-cycles 1 --mid-up-cycles 2 --fine-up-cycles 4 --mlp-hidden-dim 768 --device cuda --out data\multirate_2k.pt
```

Result:

- Parameters: `3,197,696`
- Final validation loss: `0.4147`
- Final perplexity: `1.51`

### Interpretation

The first bottleneck model did not collapse and substantially outperformed both no-bottleneck baselines, including the roughly parameter-matched baseline.

However, the compress/expand-only ablation nearly matched the full bottleneck model:

- Full bottleneck: `val_loss=0.4034`
- Compress/expand only: `val_loss=0.4099`

That means the main gain currently appears to come from the compression/expansion path itself, not from having transformer blocks operating at compressed resolution.

At the same time, the shared-weight multirate architecture achieved essentially the same validation quality:

- Shared multirate: `val_loss=0.4147`

This is important because it is much closer to the intended architecture than the earlier unshared bottleneck prototype, and it does so with fewer parameters than both the compress/expand-only and full bottleneck models.

### Why the result might still be misleading

These numbers are encouraging, but they are not yet enough to conclude that compression itself is the full cause.

Potential confounds:

- The compress/expand path may act as a strong extra mixing mechanism, independent of the bottleneck semantics.
- The anchor-position choice on the compressed path might accidentally create an easier inductive bias on this dataset.
- The architecture may be benefiting from an implementation shortcut that behaves more like a learned residual block than a genuine information bottleneck.
- Validation loss alone does not prove generation quality or structural fidelity.

### Next checks

1. Add generation and qualitative inspection from checkpoints.
2. Test a weaker bottleneck such as `compression_factor=2`.
3. Compare against other non-bottleneck extra-mixing controls.
4. Inspect failure/success specifically around punctuation, line boundaries, and word starts.
