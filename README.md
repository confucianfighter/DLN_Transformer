# Positional Character Tokenizer

This project currently contains a tokenizer prototype for a character-level transformer.

Alphabetic input is normalized to lowercase before positional tokenization.

Each alphabetic character receives one of seven position-specific labels:

- `1`
- `2`
- `3`
- `middle`
- `-3`
- `-2`
- `-1`

Resolution happens from the edges inward using this order:

`0, -1, 1, -2, 2, -3, 3, ...`

Labels are assigned in this order:

`1, -1, 2, -2, 3, -3, middle, middle, ...`

The assigned labels are then written back in the word's original reading order.

Example:

- assignment order for `Thing`: `T1, g-1, h2, n-2, i3`
- emitted token sequence: `T1, h2, i3, n-2, g-1`

For words longer than 7 characters, the extra center characters all receive `middle`.

Example:

- `Position` becomes `P1, o2, s3, imiddle, tmiddle, i-3, o-2, n-1`

## Files

- `data/tiny_shakespeare.txt`: downloaded training text
- `data/tiny_shakespeare_sample_tokens.txt`: readable excerpt plus tokenized output
- `data/transition_counts.npy`: next-token count matrix with whitespace skipped after final word chars
- `data/transition_probs.npy`: normalized transition probabilities
- `data/embedding_init.npy`: `vocab_size x 256` initializer with probabilities in the first `vocab_size` dims
- `data/transition_report.txt`: readable top-continuation report
- `tokenizer.py`: tokenizer implementation and vocab builder
- `model.py`: compact Mistral-style decoder with Spatial 2D RoPE and a first compression bottleneck
- `train.py`: baseline trainer for tiny Shakespeare
- `test_tokenizer.py`: small verification script

## Usage

```powershell
python tokenizer.py --inspect-word Thing
python test_tokenizer.py
python train.py --steps 200 --seq-len 256 --batch-size 16
```

The tokenizer script also writes a readable sampled excerpt to
`data/tiny_shakespeare_sample_tokens.txt` by default.

It also writes transition and initializer artifacts using this rule:

- internal word characters use the immediate next token
- final word characters skip whitespace and target the next non-whitespace token

The baseline model uses Spatial 2D RoPE:

- standard sequence RoPE over token index within the current training window
- newline-count RoPE, where the position only increments when a `"\n"` token appears and does not reset

Character position within the current line is carried in a separate learned side channel instead of a reset-style RoPE.
It is injected into a reserved slice of the free dimensions beyond the first `vocab_size` logit-aligned dimensions.

The current model also includes a first moment-of-truth bottleneck:

- full-resolution pre-blocks
- one non-overlapping compression stage
- bottleneck blocks at compressed resolution
- learned expansion back to full resolution
- full-token loss at the original sequence length

Set `--compression-factor 1 --n-bottleneck-layers 0` for a true no-bottleneck baseline.
