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
- `teacher_student.py`: teacher-cache distillation utilities for the interleaved student cascade
- `test_tokenizer.py`: small verification script
- `RESULTS.md`: compact experiment ledger, including the repeat-task qualitative "beer scale"
- `docs/experiments.md`: longer-form commands, metrics, and replication notes

## Usage

```powershell
python tokenizer.py --inspect-word Thing
python test_tokenizer.py
python train.py --steps 200 --seq-len 256 --batch-size 16
python teacher_student.py smoke
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

## Cascade Notes

The interleaved student stack is trained as a cascade, not as independent free-running models.

- Teacher cache: `python teacher_student.py cache_teacher --checkpoint data/multirate_2k.pt --out data/teacher_cache_multirate.npz --device cuda`
- Student 1 training: `python teacher_student.py train_student --cache data/teacher_cache_multirate.npz --out data/interleaved_student_1k.pt --steps 1000 --device cuda`
- Student 2 cache: `python teacher_student.py cache_student --checkpoint data/interleaved_student_1k.pt --base-cache data/teacher_cache_multirate.npz --out data/teacher_cache_g2.npz --device cuda`
- Student 2 training: `python teacher_student.py train_student --cache data/teacher_cache_g2.npz --out data/interleaved_student_g2_1k.pt --steps 1000 --device cuda`
- Student 3 cache: `python teacher_student.py cache_student --checkpoint data/interleaved_student_g2_1k.pt --base-cache data/teacher_cache_g2.npz --out data/teacher_cache_g3.npz --device cuda`
- Student 3 training: `python teacher_student.py train_student --cache data/teacher_cache_g3.npz --out data/interleaved_student_g3_1k.pt --steps 1000 --device cuda`

Inference contract for the student stages:

- `committed_ids` are always the accepted discrete token history
- `prediction_logits` are the upstream stage logits aligned to those committed tokens
- the correct rollout is `teacher -> student1 -> student2 -> ...`
- running `student2` alone gives misleadingly bad output quality even when cache validation loss is low

Current saved checkpoints and cache losses:

- Teacher `data/multirate_2k.pt`: `val_loss=0.4147` on the base language-model task
- Student 2 `data/interleaved_student_g2_1k.pt`: `val_loss=0.046843`, `ppl=1.047957` on `data/teacher_cache_g2.npz`
- Student 3 `data/interleaved_student_g3_1k.pt`: `val_loss=0.052078`, `ppl=1.053458` on `data/teacher_cache_g3.npz`

Interpretation:

- cache loss keeps dropping sharply across stages
- full-cascade generation is noticeably better than standalone student generation
- adding stage 3 did not obviously clean up the text; it stayed Shakespeare-like in structure but still noisy and misspelled
- this suggests the main bottleneck is rollout/generation fidelity, not just cache fitting

Rollout-aware student training is now supported in `teacher_student.py`:

- `--rollout-steps N`: unroll a generated suffix of length `N` inside each student batch
- `--rollout-weight W`: blend teacher-forced loss with generated-suffix supervised loss
- `--soft-rollout-weight S`: blend teacher-forced loss with a differentiable self-fed rollout that reuses soft committed token distributions
- `--soft-rollout-temperature T`: temperature used when building those soft committed token distributions
- `--generation-noise-rate P`: corrupt a generation-marked suffix covering about fraction `P` of each training window with plausible wrong committed tokens during teacher-forced training
- `--generation-noise-weight A`: blend clean teacher-forced loss with noisy generation-marked teacher-forced loss using weight `A`
- `--congruence-eval-steps K`: periodically measure how closely self-fed rollout predictions stay aligned with clean teacher-forced predictions
- `--congruence-rollout-steps R`: rollout depth used for that congruence check

The rollout path marks generated suffix positions with a separate phase embedding so the student can distinguish intake/prompt tokens from self-generated continuation tokens.
The soft-rollout path keeps the same interleaved architecture but replaces hard committed-token feedback inside the rollout branch with soft committed distributions projected through the token embedding table.
The generation-noise path extends that idea by teaching the student that generation-marked context can already be corrupted before the next prediction is made, using contiguous noisy suffixes rather than isolated random flips.
The clean branch stays in the objective as an anchor, so training can trade off fit and robustness instead of replacing one with the other.

When congruence evaluation is enabled, `train_student` also reports:

- `cong_kl`: divergence between clean teacher-forced next-token distributions and self-fed rollout distributions
- `cong_argmax`: argmax agreement between those two modes
- `cong_gold`: accuracy of the self-fed rollout argmax against the gold next token

To train all student stages with the same noise and congruence settings, use `train_cascade` instead of manually running `cache_teacher`, `train_student`, and `cache_student` with hand-copied flags:

```powershell
python teacher_student.py train_cascade --teacher-checkpoint data/multirate_2k.pt --out-dir data/cascade_noise_run --stages 3 --steps 1000 --device cuda --rollout-steps 16 --rollout-weight 0.4 --generation-noise-rate 0.15 --generation-noise-weight 0.25 --congruence-eval-steps 4 --congruence-rollout-steps 16
```

To try the smallest continuous-generation change first, prefer a stage-1 or short cascade run with soft rollout enabled before changing the architecture more aggressively:

```powershell
python teacher_student.py train_student --cache data/teacher_cache_g2.npz --out data/soft_rollout_trial.pt --steps 1000 --device cuda --rollout-steps 8 --soft-rollout-weight 0.4 --soft-rollout-temperature 1.0
```

## Trajectory: Frozen Reservoir Specialists

Working name: continuously learning combinatoric transformer.

The current long-copy and noisy-recall experiments suggest a useful role for small specialist models even when exact recall is imperfect: use them as fixed reservoirs rather than as live stream generators.

The reservoir hypothesis:

- Train or precompute a domain specialist over a fixed corpus such as Shakespeare.
- Freeze its emitted token stream, hidden states, or projected key/value states.
- Give a small downstream predictor cross-attention to that fixed reservoir.
- Let the predictor learn soft combinatoric pointers into the reservoir rather than memorizing the whole style distribution in its own weights.

In this view the reservoir acts like a glossary or sample-text basis, not literal retrieval chunks. The downstream model can attend to rhythm, syntax, phrase shapes, punctuation cadence, names, and lexical patterns as virtually free external structure. Its trainable capacity can then focus on gating, composition, and generalization.

The mechanism is snapshot-memory combinatorics: preserve useful frozen snapshots of prior model behavior, then train the active model to form soft attention pointers into those snapshots and recombine what it finds. The snapshot is fixed, but the pointer pattern is dynamic.

Candidate staged system:

- Fixed Shakespeare reservoir: frozen states from a small reciter/anchor model or from encoded Shakespeare chunks.
- Live input encoder: encodes the current prompt or arbitrary input.
- Decoder/predictor: causal generator with cross-attention to both the fixed reservoir and live input.
- Gating/comparison path: learns when to lean on the reservoir, when to follow the input, and how to merge them.

Training should prevent reservoir dependence:

- Use normal causal next-token loss on the decoder stream.
- Randomly drop or mask reservoir attention so the decoder remains a competent language model.
- Train with varying encoder holdout lengths: the decoder sees teacher-forced generated-region tokens that are not present in the encoder memory.
- Include no-anchor batches and full-anchor batches to preserve both standalone generation and reservoir use.

Signal degradation from a tiny or imperfect reservoir should be partially mitigated by giving the active model its own learned embeddings for the live input. The current stage is not forced to decode only from inherited context; it receives fresh token structure directly while using the frozen context as an additional reference field.

Near-term validation:

- Compare a small decoder with and without frozen-reservoir cross-attention at the same trainable parameter count.
- Track validation loss, no-anchor loss, degraded-anchor loss, and qualitative generation.
- Treat success as several coherent Shakespeare-like sentences from arbitrary/noisy input on constrained hardware, not exact quotation.

If this works, the fixed reservoir behaves like externalized parameters: stable structure stored outside the optimizer that a small model can compose through attention.

### Reservoir Ladder

A stronger version is to turn each useful specialist into the frozen context for the next one.

Stage A is trained until it reaches substantial compression or recall over a domain. It does not need exact reproduction; it only needs enough stable structure to produce useful samples, hidden states, or key/value states. Once A is good enough, freeze it and select reservoir samples where it behaves well.

Stage B is then trained with its own learned embeddings and causal objective, but with cross-attention into A's frozen reservoir context. B should not spend its trainable capacity relearning everything A already organized. It should learn how to use A as an external reference field while still developing its own predictive structure.

The process can repeat:

```text
train A -> freeze A reservoir
train B with cross-attention to frozen A -> freeze B reservoir
train C with cross-attention to frozen B -> freeze C reservoir
...
```

The inference goal is not a large live cascade where every stage must run on every token. The goal is one small active stage running on the shoulders of frozen mini-giants: precomputed reservoirs that store structure from earlier training phases. Prior stages become static context, not recurring compute. They act as fixed inherited context for the current stage.

Each frozen ancestor is a snapshot memory. Its value comes from the active stage's learned pointer combinatorics: attention can select, blend, suppress, and recombine inherited structures instead of copying a single retrieved chunk.

This is different from ordinary distillation or a standard stacked pipeline:

- the earlier model is not merely compressed into the next model's weights
- the earlier model does not have to generate live tokens during inference
- its useful behavior is preserved as frozen attention material
- the active model learns composition, gating, and prediction against that fixed material

If successful, each phase adds durable structure without requiring all previous trainable models to remain active. The system accumulates grammar, style, phrase shapes, and domain habits as reusable context, while the current small model learns how to combine that context with the live input.

Ideally the progression compounds combinatorically: each stage does not merely add another fixed cache, but learns new compositions over inherited structures. Later stages should be able to recombine earlier grammar, style, phrase, and domain patterns into a larger effective behavior space than any single stage stores directly.

A more ambitious hypothesis is that the inherited context becomes a dynamic reference system. Earlier reservoirs may start as concrete text-like structure, but later stages could distill higher-level regularities through their own combinatorics. The active model would then learn not only to attend to stored phrases or rhythms, but to point into increasingly abstract inherited structure and combine it with the current input.

This needs direct testing rather than assumption. Useful signs would include improved generation under reservoir dropout, better transfer from fixed reservoir contexts to new prompts, and qualitative behavior that reflects higher-level style or semantic control rather than only local phrase copying.

Conceptually, the frozen inherited context can be treated as a symbolic resonator. Training shapes the resonant cavity: the stored context does not run as an active generator, but its fixed structure can be selectively excited by the current model through attention. The active stage learns which inherited structures to excite, suppress, and combine for the current prediction.

The resonant-cavity metaphor matters because the ancestor can be a tiny babbling expert or memory node. It does not have to promise perfect recall or perfect generation. It only has to be consistent enough that later stages can learn its regularities, point into them, and use them as stable inherited structure.

Because each tiny expert is cheap, later nodes could attend across multiple babbling specialists rather than a single ancestor. A joining node can interleave contexts from a left ancestor, a right ancestor, and its own stream. The hope is that each specialist's obsession is consistent enough for the joining node to find useful invariants across their patterns, even when none of the streams is perfectly correct.

One candidate context schedule:

```text
left ancestor context
right ancestor context
self/current context
left ancestor context
right ancestor context
self/current context
...
```

This keeps the load small while exposing the active stage to multiple stable symbolic resonators. The useful signal would come from learned invariants across the interleaved memories, not from trusting any one ancestor as an oracle.

### Engramatic Processing

There is a related live-resonator path distinct from the frozen-cache version. After training a useful resonator, create a version that accepts three interleaved runtime inputs, such as left ancestor output, right ancestor output, and self or reality:

```text
left output
right output
self/reality
left output
right output
self/reality
...
```

In this mode the resonator is not only fixed inherited context. It runs as a small active principle-node. Its purpose is not perfect reproduction; it should resonate according to its learned principles, biases, obsessions, or even consistent lack of principle. A joining node can then observe the interleaved outputs of multiple live resonators and learn useful invariants, disagreements, and combinations.

This is closer to internal dialogue or competing subagents: multiple small voices produce biased interpretations at runtime, and a downstream node learns how to use their agreement, conflict, and repetition. In this terminology, an engram is the durable memory trace or specialist imprint, while engramatic processing is the runtime use of those traces as active symbolic resonators.

### Interleaven Context Networking

The live version can be described as interleaven context networking: multiple resonator streams are woven together into a structured runtime context. A comparator or joining node receives alternating slices from the participating streams and learns the relationships among them.

The simplest comparator input pattern is:

```text
left resonator
right resonator
reality/current stream
left resonator
right resonator
reality/current stream
...
```

For metacognitive behavior, the comparator should learn not only what each stream says, but how each stream relates to reality and to the other streams. It can learn that one resonator is stylistically useful but off-task, another is syntactically useful but factually weak, or both are confidently wrong. Consistent wrongness can still be useful if it has a stable shape that the comparator can recognize, suppress, invert, or route around.

The core test is whether a joining node can learn stream-level control: use, ignore, blend, or redirect biased resonators based on the current reality stream. If it can, interleaven context networking becomes a candidate mechanism for reflective or metacognitive behavior in larger systems.

## Vision Multirate Control

The most recent positive control uses the canonical scratch ViT MNIST repo cloned under:

`external_baselines/PyTorch-Scratch-Vision-Transformer-ViT`

Baseline tiny ViT:

- `64`-dim embeddings
- `6` transformer layers
- `4` attention heads
- `49` patch tokens plus `CLS`
- `98.98%` MNIST test accuracy after `20` epochs

Raw multirate schedule:

- patch-token rates: `1, 1/2, 1/4, 1/4, 1/2, 1`
- CLI schedule: `--multirate_schedule 1,2,4,4,2,1`
- patch context: `49 -> 25 -> 13 -> 13 -> 25 -> 49`
- `CLS` token remains uncompressed
- `98.82%` final MNIST test accuracy, best `98.91%`

This is currently the cleanest positive evidence that the learned compress/expand path can preserve task-relevant information under a hard middle-context clamp.

Additional attention-cap stress test:

- same `1,2,4,4,2,1` state schedule
- every attention layer limited to `13` patch keys/values plus `CLS`
- trained checkpoint dropped to `78.59%` without adaptation
- five capped fine-tune epochs restored test accuracy to `98.59%`

Inference timing on this tiny model is mixed:

- batch `128`: baseline `18.65 ms`, multirate `17.62 ms`, capped multirate `17.19 ms`
- batch `1`: baseline `9.46 ms`, multirate `10.93 ms`, capped multirate `11.25 ms`
- conclusion: accuracy survives compression, but this small unoptimized implementation only shows a throughput win at larger batch size

Parameter-free transition timing isolates token-count work better:

- `--rate_transition_mode mean` uses average pooling down and repeat up
- `--rate_transition_mode stride` uses strided token selection down and repeat up
- batch `128`: baseline `17.43 ms`, mean multirate `15.32 ms`, stride multirate `15.14 ms`
- batch `1`: baseline `8.84 ms`, mean multirate `10.01 ms`, stride multirate `9.84 ms`

Eight-layer 8x middle schedule:

- schedule: `--n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1`
- patch context: `49 -> 25 -> 13 -> 7 -> 7 -> 13 -> 25 -> 49`
- deepest layers run on `7` patch tokens plus `CLS`
- final MNIST test accuracy: `98.91%`

Closest-to-baseline bypass control:

- command mode: `--rate_transition_mode bypass`
- no learned transition projections, transition LayerNorms, adapters, or pooling modules
- skipped patch tokens bypass compressed middle layers and are reinserted on the way back up
- same trainable parameter count as an equal-depth full-context ViT
- 8-layer `1,2,4,8,8,4,2,1` run: `98.48%` MNIST test accuracy
- batch-128 timing against same-depth 8-layer baseline: `13.55 ms` vs `16.75 ms`

Parameter-free mean transition control:

- command mode: `--rate_transition_mode mean`
- down-rate transitions average sibling patch tokens instead of discarding one
- up-rate transitions repeat compressed patch tokens
- no learned transition parameters; same `277,002` params as the 8-layer baseline
- 8-layer `1,2,4,8,8,4,2,1` run: `98.74%` MNIST test accuracy
- batch-128 timing: `13.09 ms`

Direct 8-layer full-context control:

- command: `--n_layers 8`
- parameters: `277,002`
- final MNIST test accuracy: `99.05%`
- batch-128 timing: `16.75 ms`
- interpretation: the multirate variants trade a small accuracy drop for lower batched latency; the 8-layer baseline itself does not degrade

Width-reduced mean multirate control:

- command: `--embed_dim 32 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean`
- parameters: `71,946`
- final MNIST test accuracy: `97.92%`
- direct 32-dim full-context baseline: `97.38%`
- interpretation: at constrained width, mean multirate beat same-parameter full context by `0.54%`
- 20 extra low-LR epochs improved the 32-dim mean model to `98.36%` final, `98.39%` best
- 32-dim batch-128 timing: full baseline `13.75 ms`, mean multirate `14.94 ms`

Additional learned-multirate training:

- starting point: 8-layer learned `1,2,4,8,8,4,2,1` at `98.91%`
- 20 extra low-LR epochs from checkpoint reached `99.13%`
- this slightly beats the 8-layer full-context baseline run at `99.05%`
- targeted middle-layer fine-tuning remains the cleaner isolation test

Convolutional rate-transition control:

- command: `--rate_transition_mode conv`
- downsample uses depthwise/pointwise `Conv1d`; upsample uses depthwise transposed conv plus pointwise `Conv1d`
- parameters: `304,458`
- final MNIST test accuracy: `99.01%`, best `99.05%`
- batch-128 timing: `20.41 ms`
- interpretation: viable for accuracy, but slower than full context in this unfused implementation

CIFAR-10 larger-image control:

- setup: `32x32x3`, patch size `4`, `64` patch tokens
- 32-dim full-context baseline: `57.36%` final, `57.46%` best, `16.52 ms` batch-128
- 32-dim mean multirate: `54.65%` final, `54.82%` best, `13.53 ms` batch-128
- interpretation: larger token grid restores throughput advantage, but mean transitions lose accuracy on the harder task

CIFAR-10 learned-transition compressive run:

- command: `--image_size 32 --n_channels 3 --embed_dim 64 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1`
- parameters: `330,506`
- 20-epoch checkpoint accuracy: `68.00%`
- 44-epoch continued checkpoint accuracy: `71.82%`
- 44-epoch checkpoint loss: `0.7958`
- batch-128 timing: `11.68 ms`, about `10,960 images/sec`
- Mamba 64-dim patch classifier control: `55.50%` best after 20 epochs
- next check: resume the 64-dim learned multirate checkpoint beyond 44 epochs to test whether the steady late climb continues

See `docs/experiments.md` for commands, results, and prior-art notes.
