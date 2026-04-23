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

## 2026-04-17: Interleaved Student Cascade

### Goal

Test whether repeated distillation through the interleaved teacher-student lattice improves generation quality, or whether the system is mostly overfitting the teacher-forced cache objective.

### Inference contract

The student checkpoints are not intended to run alone during generation.

Each student stage consumes:

- `committed_ids`: the accepted discrete token history
- `prediction_logits`: the upstream stage logits aligned to those same committed tokens

The intended rollout is:

`teacher -> student1 -> student2 -> student3`

Running `student2` by itself produced severe repetition and did not match training-time semantics.

### Existing second-stage check

Saved checkpoint:

- `data/interleaved_student_g2_1k.pt`

Teacher-forced validation against `data/teacher_cache_g2.npz`:

- `val_loss=0.046843`
- `ppl=1.047957`

This was far better than the observed free-running text quality, which already suggested a train/inference mismatch or rollout-fragility issue.

### Third-stage run

Commands:

```powershell
python teacher_student.py cache_student --checkpoint data/interleaved_student_g2_1k.pt --base-cache data/teacher_cache_g2.npz --embedding-init data/embedding_init.npy --out data/teacher_cache_g3.npz --device cuda
python teacher_student.py train_student --cache data/teacher_cache_g3.npz --vocab data/positional_char_vocab.json --embedding-init data/embedding_init.npy --out data/interleaved_student_g3_1k.pt --batch-size 16 --seq-len 128 --steps 1000 --eval-interval 100 --eval-steps 20 --lr 3e-4 --weight-decay 0.1 --train-split 0.9 --seed 7 --device cuda
```

Training result:

- Parameters: `5,477,632`
- Final logged validation loss: `0.0395`
- Final logged perplexity: `1.04`

Post-training validation pass on the saved checkpoint:

- `val_loss=0.052078`
- `ppl=1.053458`

### Qualitative comparison

Prompt: `king richard`

`teacher -> student1 -> student2`

> king richard toketers of your grave lettle pounts
> powelts, teglight fur god for betice.
>
> gloucester:
> how owr some done, that some sitter rodgue souns;
> woults good seiss podour'gway down.

`teacher -> student1 -> student2 -> student3`

> king richard toders, time, too lowers:
> too pownginans, to come some soudice foudly,
> for lodces thethers, becouns, treats,
> to gentlent all; that down the todaces;

Prompt: `to be, or not to be`

`teacher -> student1 -> student2`

> to be, or not to be son are sorracted detice,
> the hon tre powerame hon at the god before.
>
> gloucester:
> you good for love here; the powetion,

`teacher -> student1 -> student2 -> student3`

> to be, or not to be fordon bee too most;
> too wot too moft to mole lovers, that some too most
> toe loves too look your great lofe come.
>
> lord, lord:

Prompt: `what says the queen?`

`teacher -> student1 -> student2`

> what says the queen?
>
> guchione:
> some old good son, it good poison; the powetharous grouns of young ogeares ot goodon,

`teacher -> student1 -> student2 -> student3`

> what says the queen?
>
> first poldhtors:
> what serven toketely; the mowding conding courts,
> too powerds erchens come cousin,

### Interpretation

Stage 3 did not collapse, and it preserved some dramatic/shakespeare-like formatting cues such as speaker labels and line breaks.

However, it did not clearly clean up spelling or semantic coherence. It mostly changed the flavor of the nonsense while keeping the same general level of corruption.

So the current picture is:

- cache-fitting loss keeps improving with added stages
- free-running generation improves substantially when the full cascade is respected
- adding a third stage does not yet produce a corresponding jump in text quality

This suggests that the limiting problem is likely in the generation contract itself, or in exposure-bias/rollout instability across the token-history plus logit-history channels, rather than simply in stage depth.

### Next checks

1. Build a reusable cascade generation script instead of ad hoc inline sampling.
2. Compare `teacher`, `teacher->student1`, `teacher->student1->student2`, and `teacher->student1->student2->student3` with fixed seeds.
3. Measure rollout-aware metrics such as repetition rate, unique token ratio, and token entropy under self-generated continuations.
4. If quality still does not improve, treat a joining model as the next architectural branch instead of blindly adding more distillation stages.

## 2026-04-17: Rollout-Aware Student Training

### Motivation

The student cache losses became extremely low while free-running cascade text remained noticeably corrupted.

That suggests a train/inference mismatch:

- training uses clean committed token history plus clean upstream logits
- generation uses self-sampled committed tokens plus drifting upstream logits

To start addressing that, `teacher_student.py` now supports a mixed objective that tells the model whether it is still consuming intake/prompt tokens or has entered generated rollout.

### Implementation

New student features:

- a phase embedding in `InterleavedTeacherStudent`
- `phase=0` for intake/teacher-forced prefix tokens
- `phase=1` for generated rollout suffix tokens
- `rollout_student_loss(...)` for truncated multi-step supervised training on a generated suffix

New CLI flags on `train_student`:

```powershell
--rollout-steps N
--rollout-weight W
```

The effective training loss is:

`(1 - W) * teacher_forced_loss + W * rollout_loss`

where `rollout_loss` uses the model's own argmax committed tokens over the suffix while still supervising against the true continuation.

### Sanity check

A one-step CUDA sanity run completed successfully:

```powershell
python teacher_student.py train_student --cache data/teacher_cache_g2.npz --vocab data/positional_char_vocab.json --embedding-init data/embedding_init.npy --out data/rollout_sanity.pt --batch-size 4 --seq-len 32 --steps 1 --eval-interval 1 --eval-steps 1 --lr 3e-4 --weight-decay 0.1 --train-split 0.9 --seed 7 --device cuda --rollout-steps 8 --rollout-weight 0.5
```

Observed output:

- `train_loss=5.3200`
- `train_tf=5.3437`
- `train_rollout=5.2963`
- `val_loss=4.8585`

This does not prove the fix works, but it confirms the rollout-aware objective is wired correctly enough to train on real cache data.

## 2026-04-19: Fresh Multirate Strict-Cap Fine-Tune

### Goal

Test whether the fresh progressive multirate model can be trained to operate under a strict per-stage context cap, instead of merely tolerating that cap at inference time.

The target architecture was the 11-stage fresh multirate schedule:

`1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/16, 1/8, 1/4, 1/2, 1`

### Implementation note

The strict cap is now implemented with actual stage-local chunked attention rather than a full dense attention matrix plus mask.

Under `stage_context_cap=16`, each query block is evaluated against only the keys/values needed for its own last-16-token causal window.

That means these results now reflect both:

- a valid information-flow test
- a real reduction in attention-matrix size inside capped runs

### Pre-fine-tune cap check

Saved checkpoint:

- `fresh_multirate/fresh_multirate_11stage_incremental_2k.pt`

Uncapped validation:

- `val_loss=1.4715`
- `ppl=4.36`

Strict cap evaluation with `stage_context_cap=16`:

- raw-token-equivalent stage windows: `16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16`
- capped `val_loss=1.5133`
- capped `ppl=4.54`

So the untuned model degraded only modestly under the cap.

### End-to-end strict-cap fine-tune

Command:

```powershell
python fresh_multirate\train.py --init-from-checkpoint fresh_multirate\fresh_multirate_11stage_incremental_2k.pt --out fresh_multirate\fresh_multirate_11stage_cap16_e2e_1k.pt --steps 1000 --eval-interval 100 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 1e-4 --weight-decay 0.1 --stage-context-cap 16 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1 --stage6-layers 1 --stage7-layers 1 --stage8-layers 1 --stage9-layers 1 --stage10-layers 1 --stage11-layers 1
```

Training highlights:

- step `1`: `val_loss=1.5219`
- step `200`: `val_loss=1.4748`
- step `600`: `val_loss=1.4777`
- step `1000`: `val_loss=1.4648`

Saved checkpoint:

- `fresh_multirate/fresh_multirate_11stage_cap16_e2e_1k.pt`

Post-training evaluation:

- capped `val_loss=1.4633`
- capped `ppl=4.32`

For reference, evaluating that same fine-tuned checkpoint without the cap gave:

- full-context `val_loss=1.4816`
- full-context `ppl=4.40`

### Interpretation

The strict-cap fine-tune worked.

A short end-to-end adaptation run recovered the capped model from `1.5133` to `1.4633`, which is slightly better than the original uncapped checkpoint (`1.4715`).

That is the strongest evidence so far that the fresh multirate branch can reorganize itself around a genuinely shorter effective context budget without collapsing.

The capped result is now backed by actual local attention computation rather than dense masked attention, so this is no longer just a conceptual compression win.

### Current status for next session

- best fresh strict-cap checkpoint: `fresh_multirate/fresh_multirate_11stage_cap16_e2e_1k.pt`
- best measured capped validation loss so far: `1.4633`
- the cap value currently tested end to end is `16`
- next clean experiment is to repeat the same recipe with `stage_context_cap=8` and `stage_context_cap=4`
- the local-cap path now uses actual stage-local chunked attention

## 2026-04-19: Fresh Multirate Strict-Cap Fine-Tune at 8

### Goal

Push the same 11-stage fresh multirate schedule to a stricter local memory budget:

`stage_context_cap=8`

This corresponds to raw-token-equivalent stage windows:

`8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8`

### Pre-fine-tune cap check

Starting checkpoint:

- `fresh_multirate/fresh_multirate_11stage_incremental_2k.pt`

Untuned capped evaluation:

- `val_loss=1.5448`
- `ppl=4.69`

This is worse than the `cap=16` starting point, but still far from a collapse.

### End-to-end strict-cap fine-tune

Command:

```powershell
python fresh_multirate\train.py --init-from-checkpoint fresh_multirate\fresh_multirate_11stage_incremental_2k.pt --out fresh_multirate\fresh_multirate_11stage_cap8_e2e_1k.pt --steps 1000 --eval-interval 100 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 1e-4 --weight-decay 0.1 --stage-context-cap 8 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1 --stage6-layers 1 --stage7-layers 1 --stage8-layers 1 --stage9-layers 1 --stage10-layers 1 --stage11-layers 1
```

Training highlights:

- step `1`: `val_loss=1.5510`
- step `200`: `val_loss=1.4857`
- step `600`: `val_loss=1.4800`
- step `1000`: `val_loss=1.4613`

Saved checkpoint:

- `fresh_multirate/fresh_multirate_11stage_cap8_e2e_1k.pt`

Post-training evaluation:

- capped `val_loss=1.4599`
- capped `ppl=4.31`

For reference, evaluating that same checkpoint without the cap gave:

- full-context `val_loss=1.5129`
- full-context `ppl=4.54`

### Interpretation

The `cap=8` fine-tune also worked.

Even with only `8` local states per stage, the model recovered from `1.5448` to `1.4599`, which is slightly better than both:

- the original uncapped 11-stage checkpoint (`1.4715`)
- the `cap=16` specialized checkpoint (`1.4633`)

The fact that the `cap=8` checkpoint gets worse when evaluated at full context (`1.5129`) is useful: it suggests the model is not merely robust to the cap, but is actively specializing into that tighter memory regime.

### Updated status for next session

- best fresh strict-cap checkpoint so far: `fresh_multirate/fresh_multirate_11stage_cap8_e2e_1k.pt`
- best measured capped validation loss so far: `1.4599`
- strongest tested strict cap so far: `8`
- next clean experiment is `stage_context_cap=4`

## 2026-04-19: Progressive Capped Growth at 8

### Goal

Test whether the fresh multirate hierarchy trains more efficiently when it is grown progressively under the same strict local cap from the beginning, instead of specializing an already-built 11-stage model afterward.

The common cap for the whole ladder was:

`stage_context_cap=8`

### Training ladder

#### 5-stage base

Command:

```powershell
python fresh_multirate\train.py --out fresh_multirate\fresh_multirate_cap8_5stage_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 3e-4 --weight-decay 0.1 --stage-context-cap 8 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1
```

Result:

- checkpoint: `fresh_multirate/fresh_multirate_cap8_5stage_2k.pt`
- capped `val_loss=1.5005`
- capped `ppl=4.48`

#### 7-stage incremental

Command:

```powershell
python fresh_multirate\train.py --init-from-checkpoint fresh_multirate\fresh_multirate_cap8_5stage_2k.pt --freeze-existing --out fresh_multirate\fresh_multirate_cap8_7stage_incremental_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 3e-4 --weight-decay 0.1 --stage-context-cap 8 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1 --stage6-layers 1 --stage7-layers 1
```

Result:

- checkpoint: `fresh_multirate/fresh_multirate_cap8_7stage_incremental_2k.pt`
- capped `val_loss=1.4819`
- capped `ppl=4.40`

#### 9-stage incremental

Command:

```powershell
python fresh_multirate\train.py --init-from-checkpoint fresh_multirate\fresh_multirate_cap8_7stage_incremental_2k.pt --freeze-existing --out fresh_multirate\fresh_multirate_cap8_9stage_incremental_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 3e-4 --weight-decay 0.1 --stage-context-cap 8 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1 --stage6-layers 1 --stage7-layers 1 --stage8-layers 1 --stage9-layers 1
```

Result:

- checkpoint: `fresh_multirate/fresh_multirate_cap8_9stage_incremental_2k.pt`
- capped `val_loss=1.4797`
- capped `ppl=4.39`

#### 11-stage incremental

Command:

```powershell
python fresh_multirate\train.py --init-from-checkpoint fresh_multirate\fresh_multirate_cap8_9stage_incremental_2k.pt --freeze-existing --out fresh_multirate\fresh_multirate_cap8_11stage_incremental_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --seq-len 256 --lr 3e-4 --weight-decay 0.1 --stage-context-cap 8 --device cuda --stage1-layers 1 --stage2-layers 1 --stage3-layers 1 --stage4-layers 1 --stage5-layers 1 --stage6-layers 1 --stage7-layers 1 --stage8-layers 1 --stage9-layers 1 --stage10-layers 1 --stage11-layers 1
```

Result:

- checkpoint: `fresh_multirate/fresh_multirate_cap8_11stage_incremental_2k.pt`
- capped `val_loss=1.4802`
- capped `ppl=4.39`

### Interpretation

Progressive capped growth is viable.

It produced a clean, monotonic improvement from 5-stage to 7-stage and then largely plateaued:

- 5-stage: `1.5005`
- 7-stage: `1.4819`
- 9-stage: `1.4797`
- 11-stage: `1.4802`

So the progressive ladder under a hard `cap=8` does work, but this first pass did **not** beat the direct end-to-end `11-stage cap=8` specialization checkpoint:

- direct `11-stage cap=8` fine-tune: `1.4599`
- progressive `11-stage cap=8`: `1.4802`

That suggests one of two things:

- the progressive scheme is sound, but the frozen incremental recipe is too restrictive
- or the already-built 11-stage model is a better starting point for reorganizing under the cap than the narrow ladder is for discovering the structure from scratch

### Current status for next session

- best strict-cap checkpoint overall is still `fresh_multirate/fresh_multirate_11stage_cap8_e2e_1k.pt`
- best progressive strict-cap checkpoint is `fresh_multirate/fresh_multirate_cap8_9stage_incremental_2k.pt` or `...11stage...`, both at about `1.48`
- next useful experiment is to repeat the progressive ladder with either:
- partial rather than full freezing during each expansion
- or a short end-to-end unfreeze after each incremental step

## 2026-04-19: Session Recap

### What we established

- The fresh multirate branch can operate under very hard local memory caps without collapsing.
- A strict local cap is no longer only a conceptual ablation. The cap path now uses actual stage-local chunked attention rather than a dense full-length attention matrix plus mask.
- Direct end-to-end specialization under the cap currently works better than the first fully frozen progressive ladder.

### Most important quantitative findings

- Original 11-stage fresh multirate checkpoint:
- full-context `val_loss=1.4715`
- `cap=16` untuned `val_loss=1.5133`
- `cap=8` untuned `val_loss=1.5448`

- Direct end-to-end cap-specialized checkpoints:
- `fresh_multirate/fresh_multirate_11stage_cap16_e2e_1k.pt`: capped `val_loss=1.4633`
- `fresh_multirate/fresh_multirate_11stage_cap8_e2e_1k.pt`: capped `val_loss=1.4599`

- Progressive `cap=8` ladder:
- 5-stage: `1.5005`
- 7-stage: `1.4819`
- 9-stage: `1.4797`
- 11-stage: `1.4802`

### Salient qualitative excerpts

Direct `11-stage cap=16`:

> king richard ii:
> but this i fear; having my brother speech.
>
> king richard iii:
> somehting thou shalt know, young less a mother:

Direct `11-stage cap=8`:

> king richard ii:
> but this is not yet and my brother henry,
> the hand-make prince of him, and my help
> a patrician commonwealth as the tribunes all.

Progressive `11-stage cap=8`:

> king richard ii:
> god the heavens in thy farthest wise.
>
> hermione:
> thou shalt never battle and fear the heir from

### Steps we took

1. Built strict-cap evaluation and generation support for the fresh multirate branch.
2. Replaced dense masked attention with actual local chunked attention so capped runs materially reduce the attention working set.
3. Fine-tuned the saved 11-stage model end to end at `cap=16`.
4. Repeated the same end-to-end specialization at `cap=8`.
5. Ran a full progressive `cap=8` ladder: `5 -> 7 -> 9 -> 11`.
6. Compared quantitative and qualitative results between direct specialization and progressive growth.

### Current interpretation

- The cap itself is real and trainable. The model can reorganize around `cap=8` and still produce better capped loss than the original full-context checkpoint.
- The direct `11-stage cap=8` specialization appears to be actively adapted to the tighter regime, because its full-context score gets worse (`1.5129`) while its capped score gets better (`1.4599`).
- Progressive capped growth is viable, but the fully frozen expansion recipe probably leaves too much of the outer stack fixed. It improves the ladder quickly at first, then plateaus near `1.48`.

### Why this matters

- On limited hardware such as a GTX 980 Ti, local caps plus deep multirate structure are now a plausible route to much more memory-efficient inference and potentially cheaper training.
- The fact that `cap=8` still works qualitatively and quantitatively makes much deeper hierarchies more credible than they looked at the start of the session.
- This is what motivated the next idea: a very deep narrow repeater trained for long-range reconstruction rather than open-ended language modeling.

### Planned next steps

1. Build a deep repeater/reconstruction branch at low width, starting with `d_model=64`.
2. Test replay/copy behavior before attempting any frozen-memory lexicon system.
3. Prefer a staged depth ladder such as `11 -> 21 -> 32` rather than jumping directly to the deepest version.
4. Keep the first repeater task synthetic and exact:
   input long text, then force exact replay after a delimiter or in a decode phase.

### Reasoning for next steps

- A repeater task is a cleaner compression test than language modeling because it removes semantic ambiguity and scores exact information preservation directly.
- `d_model=64` makes very deep experiments realistic on the current hardware.
- If a deep narrow repeater can reliably preserve long held-out strings, then a frozen-memory predictor with cross-attention becomes much more justified.
- Training the repeater on one data split and the downstream predictor on a different split is the cleanest way to force chunking and lexical reuse rather than simple continuation memorization.

## 2026-04-19: Repeater Scaffold

### Goal

Prepare a dedicated deep low-width repeater branch for exact replay experiments without disturbing the direct LM branch.

### Implementation

Added new files under `fresh_multirate/`:

- `repeater_model.py`: generic odd-depth multirate repeater with configurable `stage_count`
- `repeater_train.py`: trains on `source + separator + source`
- `repeater_generate.py`: greedy replay from a trained repeater checkpoint

The repeater objective uses:

- a single autoregressive stream
- a replay-only loss mask
- loss applied only on the second `source` span after the separator

This keeps the task focused on preserved recall rather than generic prefix modeling.

### Smoke check

Command:

```powershell
python fresh_multirate\repeater_train.py --out fresh_multirate\repeater_smoke.pt --steps 1 --eval-interval 1 --eval-steps 1 --batch-size 2 --source-len 32 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --device cpu
```

Observed output:

- parameters: `611,776`
- step `1` train loss: `5.1940`
- step `1` val loss: `5.5408`

Greedy replay from that 1-step smoke checkpoint produced nonsense, which is expected at this stage, but it confirmed the decode path runs end to end.

### Ready next runs

The intended first real ladder is:

1. `11-stage`, `d_model=64`
2. `21-stage`, `d_model=64`
3. `32-stage` only if the lower-depth repeaters are healthy

The intent is to test exact replay before attempting any frozen-memory lexicon predictor.

## 2026-04-19: First Real Repeater Run

### Goal

Test whether the new low-width repeater branch can learn exact replay at a meaningful chunk length before pushing to deeper hierarchies.

### Run

Command:

```powershell
python fresh_multirate\repeater_train.py --out fresh_multirate\repeater_11stage_d64_cap8_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 128 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --device cuda
```

Result:

- checkpoint: `fresh_multirate/repeater_11stage_d64_cap8_2k.pt`
- parameters: `611,776`
- final validation loss: `1.6948`
- final perplexity: `5.45`

### Qualitative replay check

Held-out prompt:

> king richard was slain in the field by treacherous friends and left without a crown.

Replay:

> to menting and son.
>
> ferthing the cariting.
>
> frience our coursing thee the conding to him cantin

Held-out prompt:

> to be, or not to be, that is the question whether tis nobler in the mind to suffer

Replay:

> to the second of man of more and and the couse them in a with them, then the comes and contines

### Interpretation

This run learned structure, but it did **not** learn faithful replay.

The model is behaving like a small language model conditioned on the source rather than a true repeater. It produces locally plausible Shakespeare-like continuations instead of reconstructing the exact source string.

That means the current objective is still too easy to satisfy with generic predictive statistics.

### Most likely next adjustment

Before pushing depth, make the replay task more diagnostic. The cleanest next changes are:

1. shorten the initial source length to something like `32` or `64` and verify exact replay first
2. add a stronger separator or gap so the model cannot smoothly continue the source as ordinary language modeling
3. add exact replay metrics such as token accuracy, exact match rate, and longest correct prefix

### Why this matters

The result is still useful: it prevents us from misreading falling replay loss as successful memory compression.

At the moment, the repeater branch is learning stylistic continuation, not exact stored recall. That has to be fixed before the `21-stage` and `32-stage` runs are worth interpreting.

## 2026-04-19: Repeater Conv Branch Scaffold

### Goal

Add a cheap local inductive bias to the repeater blocks while keeping the attention path intact.

### Implementation

The repeater block now supports an optional parallel causal convolution branch:

- attention branch on normalized hidden states
- depthwise causal 1D convolution plus pointwise projection on normalized hidden states
- both branches are added residually before the MLP

New repeater training flags:

- `--use-conv-branch`
- `--conv-kernel-size`

### Smoke check

Command:

```powershell
python fresh_multirate\repeater_train.py --out fresh_multirate\repeater_conv_smoke.pt --steps 1 --eval-interval 1 --eval-steps 1 --batch-size 2 --source-len 32 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --use-conv-branch --conv-kernel-size 3 --device cpu
```

Observed output:

- parameters: `659,648`
- step `1` train loss: `5.4147`
- step `1` val loss: `5.4148`

### Current status

The `attention + conv` repeater path is live and ready for a real comparison against the current attention-only repeater baseline.

## 2026-04-19: First Real Repeater Run With Parallel Conv

### Goal

Test whether adding a cheap local convolution branch in parallel with attention improves replay behavior on the existing natural-text copy task.

### Run

Command:

```powershell
python fresh_multirate\repeater_train.py --out fresh_multirate\repeater_11stage_d64_cap8_conv_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 128 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --use-conv-branch --conv-kernel-size 3 --device cuda
```

Result:

- checkpoint: `fresh_multirate/repeater_11stage_d64_cap8_conv_2k.pt`
- parameters: `659,648`
- final validation loss: `1.6844`
- final perplexity: `5.39`

For reference, the attention-only version at the same settings finished at:

- `val_loss=1.6948`
- `ppl=5.45`

### Qualitative replay check

Held-out prompt:

> king richard was slain in the field by treacherous friends and left without a crown.

Replay:

> hears my lord.
>
> second the canging of contenter consence. i shall have the hath heaven marreles

Held-out prompt:

> to be, or not to be, that is the question whether tis nobler in the mind to suffer

Replay:

> the thought the come, and shall the come, and that is is and is the you, and shander therefore

### Interpretation

The parallel conv branch helps a little on the loss, but it does **not** solve the core problem.

The model is still behaving like a conditioned language model rather than a faithful repeater. It continues to exploit linguistic regularities instead of reconstructing the source string.

That means the main bottleneck is still the task design, not just the local mixing operator.

### Current conclusion

- `attention + conv` is a plausible local mixer and may still be useful later
- but the current natural-text replay objective remains too easy to satisfy with continuation behavior
- the next meaningful repeater step should be a more diagnostic task, not more architecture scaling on the same task

## 2026-04-19: First Synthetic Copy Run

### Goal

Remove the natural-language continuation shortcut entirely and test whether the same `11-stage d_model=64 cap=8` repeater can preserve literal sequence identity on a synthetic copy task.

### Run

Command:

```powershell
python fresh_multirate\repeater_train.py --task synthetic-copy --out fresh_multirate\repeater_11stage_d64_cap8_synth_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 128 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --device cuda
```

Result:

- checkpoint: `fresh_multirate/repeater_11stage_d64_cap8_synth_2k.pt`
- parameters: `612,480`
- final validation loss: `5.2430`
- final perplexity: `189.23`
- final token accuracy: `0.0054`
- final exact match rate: `0.0000`
- final mean correct prefix length: `0.01`

### Interpretation

This run did not just fail to copy well. It failed almost completely.

The metrics stayed essentially at chance for the whole run, which is a very different failure mode from the earlier natural-text replay runs.

That result is informative:

- the natural-text repeater was exploiting language structure rather than storing exact sequences
- once that shortcut is removed, the current architecture/objective combination is not yet learning usable recall

### What this means

This strongly suggests that we do **not** yet have a functioning exact-memory system in the current repeater branch.

The next useful adjustments are likely to be:

1. shorten the source length sharply, for example `16`, `32`, or `64`
2. test a shallower control before pushing depth again
3. consider a task variant like delayed copy or reverse copy only after exact short copy works

### Current conclusion

The synthetic objective was the right scientific move. It removed the ambiguity immediately.

At the moment:

- natural-text replay success was mostly stylistic conditioning
- synthetic exact replay is not yet working

So the architecture still needs a simpler memory benchmark before deeper or more ambitious frozen-memory ideas are worth pursuing.

## 2026-04-19: Plain-Stack Synthetic Copy Control

### Goal

Test whether the synthetic-copy failure is specific to the multirate bottlenecked hierarchy, or whether a plain non-compressive stack at the same depth and width also fails.

### Run

Command:

```powershell
python fresh_multirate\repeater_train.py --task synthetic-copy --plain-stack --out fresh_multirate\repeater_11stage_d64_plain_cap8_synth_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 128 --stage-count 11 --d-model 64 --n-heads 4 --mlp-hidden-dim 192 --stage-context-cap 8 --device cuda
```

Result:

- checkpoint: `fresh_multirate/repeater_11stage_d64_plain_cap8_synth_2k.pt`
- parameters: `612,480`
- final validation loss: `5.2430`
- final perplexity: `189.24`
- final token accuracy: `0.0057`
- final exact match rate: `0.0000`
- final mean correct prefix length: `0.01`

### Interpretation

The plain non-compressive control failed almost identically to the multirate model.

That is an important negative result:

- the current synthetic-copy failure is **not** primarily caused by the multirate bottleneck
- at `11 stages`, `d_model=64`, and `source_len=128`, the broader deep-narrow architecture/objective combination is the problem

### Current conclusion

The next most informative control is now width, not bottlenecking.

The clean follow-up is:

1. plain `11-stage`, wider model, same synthetic-copy task
2. if that works, then compare wider multirate against wider plain

## 2026-04-19: Plain-Stack Synthetic Copy at 256 Width

### Goal

Test whether the synthetic-copy failure at `d_model=64` is primarily a width issue.

### Run

Command:

```powershell
python fresh_multirate\repeater_train.py --task synthetic-copy --plain-stack --out fresh_multirate\repeater_11stage_d256_plain_cap8_synth_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 128 --stage-count 11 --d-model 256 --n-heads 8 --mlp-hidden-dim 768 --stage-context-cap 8 --device cuda
```

Result:

- checkpoint: `fresh_multirate/repeater_11stage_d256_plain_cap8_synth_2k.pt`
- parameters: `9,478,656`
- final validation loss: `5.2442`
- final perplexity: `189.46`
- final token accuracy: `0.0051`
- final exact match rate: `0.0000`
- final mean correct prefix length: `0.01`

### Interpretation

Increasing width from `64` to `256` did not materially change the result.

So for this exact synthetic-copy setup:

- the failure is not mainly the multirate bottleneck
- the failure is not mainly narrow width

The more likely issue is the current training formulation itself, or a deeper architectural mismatch between this autoregressive replay setup and true exact-copy learning.

### Pedagogical takeaway

This is an important negative result:

- natural-text replay looked encouraging only because the model could exploit linguistic continuation
- once exact synthetic recall is demanded, both narrow and wide plain stacks fail in essentially the same way

So the next changes should target the task formulation before scaling width or depth further.

## 2026-04-20: Control Ladder Scaffold

### Goal

Build a proper control ladder for exact-copy experiments instead of relying only on transformer variants.

### Implementation

Added a tiny seq2seq LSTM baseline:

- `fresh_multirate/lstm_copy_train.py`

This baseline uses:

- the same tokenizer
- the same `synthetic-copy` or `text-copy` task split
- exact replay metrics:
- token accuracy
- exact match rate
- mean correct prefix length

### Smoke check

Command:

```powershell
python fresh_multirate\lstm_copy_train.py --task synthetic-copy --out fresh_multirate\lstm_copy_synth_smoke.pt --steps 1 --eval-interval 1 --eval-steps 1 --batch-size 2 --source-len 16 --d-model 64 --hidden-size 128 --device cpu
```

Observed output:

- parameters: `235,712`
- step `1` val loss: `5.2657`
- token accuracy: `0.0000`
- exact match: `0.0000`

This only verifies the path is live; it is not a substantive result yet.

### Next clean ladder

1. LSTM synthetic copy at short lengths such as `16`, `32`, maybe `64`
2. Plain transformer synthetic copy at the same lengths
3. Compressed / multirate transformer synthetic copy at the same lengths

### Why this matters

This ladder should tell us whether:

- the task itself is learnable in our pipeline
- plain transformers can solve it at modest lengths on our hardware
- compression fails only after a known-good plain control exists

## 2026-04-20: First Real LSTM Synthetic Copy Control

### Goal

Test whether exact synthetic copy is learnable at all in the current tokenizer / data pipeline, using the simplest control in the ladder: a tiny seq2seq LSTM.

### Run

Command:

```powershell
python fresh_multirate\lstm_copy_train.py --task synthetic-copy --out fresh_multirate\lstm_copy_synth_len16_2k.pt --steps 2000 --eval-interval 200 --eval-steps 20 --batch-size 16 --source-len 16 --d-model 64 --hidden-size 128 --num-layers 1 --device cuda
```

Result:

- checkpoint: `fresh_multirate/lstm_copy_synth_len16_2k.pt`
- parameters: `235,712`
- final validation loss: `5.1989`
- final perplexity: `181.07`
- final token accuracy: `0.0105`
- final exact match rate: `0.0000`
- final mean correct prefix length: `0.02`

### Interpretation

This control also failed badly, even at the short source length of `16`.

That is a strong diagnostic result:

- the current failure is not specific to the transformer
- it is not specific to bottlenecking
- it is not even specific to long synthetic copy lengths

So the remaining suspect is the task formulation itself, or some interaction between the tokenizer/vocabulary and the synthetic-copy objective.

### Current conclusion

We should pause architecture comparisons and re-examine the exact copy task setup before drawing any more conclusions from model failures.

## 2026-04-20: Scratch ViT MNIST Multirate Control

### Goal

Switch from the failing exact-copy control ladder to a known-good tiny vision-transformer benchmark, then test whether the same multirate compression idea survives on a task that should train reliably on the local machine.

The external control was cloned from:

- `s-chh/PyTorch-Scratch-Vision-Transformer-ViT`

This is the familiar small scratch ViT setup:

- dataset: `MNIST`
- image size: `28 x 28`
- patch size: `4`
- patch tokens: `49`
- total sequence length: `50` including `CLS`
- embedding width: `64`
- layers: `6`
- attention heads: `4`
- MLP multiplier: `2`

### Baseline run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs --model_path ./checkpoints --data_path ./data
```

Result:

- parameters: `210,058`
- final train accuracy: `98.74%`
- final test accuracy: `98.98%`
- final test loss: `0.0336`
- duration: `21:10`

This establishes that the upstream tiny ViT benchmark is working correctly in this environment.

### Middle half-rate compression

First compression schedule:

`1, 1, 1/2, 1/2, 1, 1`

Implementation:

- keep the `CLS` token uncompressed
- compress only patch tokens
- run the middle two encoder layers on half-rate patch context
- expand back before the final two encoder layers

For MNIST this clamps the sequence as:

- full: `49` patch tokens + `CLS` = `50`
- middle: `ceil(49 / 2) = 25` patch tokens + `CLS` = `26`

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate --model_path ./checkpoints_multirate --data_path ./data --use_multirate
```

Result:

- parameters: `226,890`
- final train accuracy: `98.19%`
- final test accuracy: `98.54%`
- final test loss: `0.0428`
- duration: `20:37`

Interpretation:

- The half-rate middle bottleneck only cost `0.44%` absolute test accuracy against the baseline.
- The result is strong enough to justify testing deeper rate schedules instead of stopping at the first bottleneck.

### Raw step-down / step-up multirate schedule

Second compression schedule:

`1, 1/2, 1/4, 1/4, 1/2, 1`

Implementation:

- generalized the scratch ViT to accept `--multirate_schedule`
- schedule entries are integer rate denominators
- `1,2,4,4,2,1` corresponds to `1, 1/2, 1/4, 1/4, 1/2, 1`
- `CLS` remains full-rate at every stage
- patch token counts are target-count based so expansion restores the intended resolution

For MNIST this clamps patch context as:

`49 -> 25 -> 13 -> 13 -> 25 -> 49`

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_124421 --model_path ./checkpoints_multirate_124421 --data_path ./data --multirate_schedule 1,2,4,4,2,1
```

Result:

- parameters: `243,722`
- final train accuracy: `98.64%`
- final test accuracy: `98.82%`
- best test accuracy: `98.91%`
- final test loss: `0.0373`
- duration: `21:19`

### Comparison

| Model | Patch-token schedule | Middle sequence length | Parameters | Final test accuracy | Best test accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline scratch ViT | `49,49,49,49,49,49` | `50` including `CLS` | `210,058` | `98.98%` | `98.98%` |
| Half-rate middle | `49,49,25,25,49,49` | `26` including `CLS` | `226,890` | `98.54%` | `98.54%` |
| Step-down / step-up | `49,25,13,13,25,49` | `14` including `CLS` | `243,722` | `98.82%` | `98.91%` |

### Interpretation

This is the strongest positive control so far for the multirate compression idea.

The raw `1, 1/2, 1/4, 1/4, 1/2, 1` model trained from scratch and essentially matched the baseline tiny ViT within normal MNIST-scale noise, while forcing the middle transformer blocks to operate on only `13` patch tokens plus `CLS`.

This is a much cleaner signal than the previous synthetic-copy experiments because the baseline task is known to train correctly and the external repo reports near-perfect MNIST behavior under similar settings.

Important caveat:

- MNIST classification is forgiving. A classifier can discard information that a reconstruction or OCR model would need.
- The result proves the compression path can preserve task-relevant information for a simple visual classification benchmark, not that it preserves all input information.

### Prior-art positioning

A first pass suggests this sits near several established ViT efficiency lines:

- Token Merging / ToMe: merges similar tokens inside existing ViTs for throughput gains, often without retraining. Official repo: `facebookresearch/ToMe`; paper: `Token Merging: Your ViT But Faster`.
- Token Pooling: downsampling tokens by minimizing reconstruction error, motivated by redundancy after attention smoothing.
- PatchMerger / Learning to Merge Tokens in Vision Transformers: learned token merging between intermediate ViT layers.
- Learnable Token Merging / LTM-Transformer: replaces transformer blocks with learnable token-merging variants and frames part of the motivation through an information-bottleneck lens.
- Hierarchical vision transformers such as Swin, PVT, and MViT: reduce spatial/token resolution across stages and often increase channel capacity as resolution falls.

The current experiment is therefore not claiming token reduction in ViTs is new.

What appears distinctive in this local prototype is the very small, explicit, symmetric rate schedule applied inside a plain scratch ViT:

- deterministic adjacent patch compression
- learned compress/expand projections
- `CLS` preserved at full rate
- schedule returns to the original token resolution before classification
- no similarity search, pruning policy, or large hierarchical backbone

The closest conceptual relatives are PatchMerger / learnable token merging and multiscale ViTs, but our current setup is a deliberately minimal control: it tests whether a plain tiny ViT can tolerate a `1 -> 1/2 -> 1/4 -> 1/4 -> 1/2 -> 1` rate path with almost no architectural machinery.

### Next checks

1. Repeat the raw schedule with at least one more seed to estimate run-to-run noise.
2. Run `FashionMNIST`, because MNIST may be too forgiving.
3. Add throughput/FLOP accounting to verify the expected savings in the middle layers.
4. Try the progressive regime only if harder datasets fail from scratch.
5. Try a reconstruction or OCR-adjacent task after classification controls are stable.

## 2026-04-20: Scratch ViT Attention Context Cap Recovery

### Goal

Test a stricter constraint than merely changing token rate.

Keep the successful `1, 1/2, 1/4, 1/4, 1/2, 1` multirate state schedule, but cap every attention layer's key/value context to the shortest patch-token count used anywhere in that schedule.

For MNIST:

- state schedule: `49 -> 25 -> 13 -> 13 -> 25 -> 49` patch tokens
- attention key/value cap: `13` patch tokens plus `CLS`
- query/output tokens still follow the full multirate schedule
- `CLS` is always included in the attention context

Important distinction:

- This is not the discarded `4,4,4,4,4,4` experiment.
- The model still uses the same `1,2,4,4,2,1` state schedule and starts from the trained `1,2,4,4,2,1` checkpoint.

### Implementation

Added:

- `--attention_context_patch_cap N`: limits attention keys/values to `CLS + N` patch tokens
- `--eval_only`: evaluates a loaded checkpoint without training
- `--init_model_path`: initializes from one checkpoint directory while saving fine-tuned weights elsewhere

The first implementation uses a literal prefix truncation of patch-token context. That is a deliberately harsh and spatially biased cap, because full-rate layers only attend to the first `13` patch tokens rather than an adaptive, pooled, or windowed context.

### Capped evaluation before adaptation

Starting checkpoint:

- `external_baselines/PyTorch-Scratch-Vision-Transformer-ViT/checkpoints_multirate_124421/mnist/ViT_model.pt`

Command:

```powershell
python main.py --dataset mnist --batch_size 128 --n_workers 0 --model_path ./checkpoints_multirate_124421 --data_path ./data --multirate_schedule 1,2,4,4,2,1 --attention_context_patch_cap 13 --load_model True --eval_only
```

Result:

- uncapped checkpoint reference: `98.82%` final test accuracy
- capped without adaptation: `78.59%` test accuracy
- capped test loss: `0.7565`

Interpretation:

- The trained model does not tolerate the hard attention-context cap out of the box.
- This is expected because the full-rate layers were trained with full attention context.

### Five-epoch capped fine-tune

Command:

```powershell
python main.py --dataset mnist --epochs 5 --warmup_epochs 1 --batch_size 128 --n_workers 0 --lr 0.0001 --output_path ./runs_multirate_124421_cap13_ft5 --model_path ./checkpoints_multirate_124421_cap13_ft5 --init_model_path ./checkpoints_multirate_124421/mnist --data_path ./data --multirate_schedule 1,2,4,4,2,1 --attention_context_patch_cap 13
```

Result by epoch:

- epoch `1`: `98.31%`
- epoch `2`: `98.28%`
- epoch `3`: `98.51%`
- epoch `4`: `98.52%`
- epoch `5`: `98.59%`

Final result:

- train accuracy: `98.06%`
- test accuracy: `98.59%`
- test loss: `0.0471`
- duration: `5:48`

### Interpretation

This is a strong recovery result.

The hard attention cap initially dropped the trained model from `98.82%` to `78.59%`, but only five low-LR fine-tune epochs restored it to `98.59%`.

That means the model can adapt to the stricter rule where every attention layer sees at most `13` patch keys/values plus `CLS`, even when the hidden-state schedule expands back to `25` and `49` patch-token states.

This does not yet prove the cap is optimal or unbiased:

- the current cap uses the first `13` patch tokens as keys/values
- that is likely worse than pooled context, evenly spaced context, learned context selection, or local windows
- MNIST remains forgiving

But as a stress test, it is encouraging: most of the lost accuracy came back very quickly without restarting from scratch.

### Next checks

1. Replace prefix truncation with a less spatially biased context selector.
2. Compare prefix cap, evenly spaced cap, pooled cap, and learned cap.
3. Repeat on FashionMNIST.
4. Add actual attention FLOP accounting for capped key/value length.
5. Try training from scratch with the cap enabled after the fine-tune path is stable.

## 2026-04-20: Scratch ViT Inference Timing

### Goal

Measure whether the tiny MNIST multirate models show an inference-time benefit, not just accuracy retention.

### Implementation

Added:

- `external_baselines/PyTorch-Scratch-Vision-Transformer-ViT/benchmark_inference.py`

The benchmark:

- builds the same scratch ViT architecture
- optionally loads a checkpoint
- supports `--multirate_schedule`
- supports `--attention_context_patch_cap`
- runs CUDA warmup iterations
- synchronizes before and after timed loops
- reports mean latency and images/sec

### Batch throughput test

Shared settings:

- device: `cuda`
- batch size: `128`
- warmup steps: `50`
- timed steps: `300`
- input: random `1 x 28 x 28` images

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --checkpoint ./checkpoints/mnist
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --checkpoint ./checkpoints_multirate_124421/mnist --multirate_schedule 1,2,4,4,2,1
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --checkpoint ./checkpoints_multirate_124421_cap13_ft5/mnist --multirate_schedule 1,2,4,4,2,1 --attention_context_patch_cap 13
```

Results:

| Model | Accuracy reference | Mean latency | Images/sec | Relative latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `98.98%` | `18.6480 ms` | `6864.01` | `1.00x` |
| Multirate `1,2,4,4,2,1` | `98.82%` | `17.6191 ms` | `7264.85` | `0.94x` |
| Multirate + cap-13 fine-tune | `98.59%` | `17.1860 ms` | `7447.90` | `0.92x` |

Interpretation:

- Batched throughput improves modestly.
- The capped model is fastest in this batch setting.
- The gain is only about `8%` latency reduction because the model is extremely small and the added compress/expand modules are not free.

### Batch-1 latency test

Shared settings:

- device: `cuda`
- batch size: `1`
- warmup steps: `100`
- timed steps: `1000`

Commands:

```powershell
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --checkpoint ./checkpoints/mnist
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --checkpoint ./checkpoints_multirate_124421/mnist --multirate_schedule 1,2,4,4,2,1
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --checkpoint ./checkpoints_multirate_124421_cap13_ft5/mnist --multirate_schedule 1,2,4,4,2,1 --attention_context_patch_cap 13
```

Results:

| Model | Mean latency | Images/sec | Relative latency |
| --- | ---: | ---: | ---: |
| Baseline | `9.4631 ms` | `105.67` | `1.00x` |
| Multirate `1,2,4,4,2,1` | `10.9303 ms` | `91.49` | `1.16x` |
| Multirate + cap-13 fine-tune | `11.2521 ms` | `88.87` | `1.19x` |

Interpretation:

- Batch-1 latency gets worse.
- At this tiny model size, the extra Python/module overhead and compress/expand projections dominate any attention savings.
- The cap also adds concatenation/slicing overhead in the current naive implementation.

### Current conclusion

The accuracy result is much stronger than the wall-clock result on this tiny MNIST model.

This does not invalidate the compression result. It means the current implementation is a research-control implementation, not an optimized inference kernel.

To show speed convincingly, the next timing work should use:

1. larger sequence lengths or harder image sizes where attention cost dominates
2. a cleaner FLOP/attention-op estimate alongside wall-clock time
3. fused or lower-overhead transition modules
4. avoiding dynamic module overhead and runtime context concatenation
5. possibly `torch.compile` after correctness is stable

## 2026-04-20: Parameter-Free Multirate Timing Control

### Goal

Remove learned compress/expand projections from the timing comparison so the control changes mostly the number of patch-token states processed by each transformer layer.

This answers a narrower question:

- if we only change how many tokens each layer runs over, do we see the expected speed direction?

### Implementation

Added `--rate_transition_mode`:

- `learned`: current learned `Linear + LayerNorm` compress/expand path
- `mean`: average adjacent token groups on downsample, repeat tokens on upsample
- `stride`: take every `N`th token on downsample, repeat tokens on upsample

The `mean` and `stride` modes add no trainable parameters.

They still preserve `CLS` and use the same rate schedule machinery.

### Batch throughput timing

Shared settings:

- random inputs
- no checkpoint loading
- batch size: `128`
- warmup steps: `50`
- timed steps: `300`

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --multirate_schedule 1,2,4,4,2,1 --rate_transition_mode mean
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --multirate_schedule 1,2,4,4,2,1 --rate_transition_mode stride
```

Results:

| Model | Parameters | Mean latency | Images/sec | Relative latency |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `210,058` | `17.4344 ms` | `7341.80` | `1.00x` |
| Mean multirate | `210,058` | `15.3189 ms` | `8355.68` | `0.88x` |
| Stride multirate | `210,058` | `15.1370 ms` | `8456.08` | `0.87x` |

### Batch-1 timing

Shared settings:

- random inputs
- no checkpoint loading
- batch size: `1`
- warmup steps: `100`
- timed steps: `1000`

Results:

| Model | Mean latency | Images/sec | Relative latency |
| --- | ---: | ---: | ---: |
| Baseline | `8.8423 ms` | `113.09` | `1.00x` |
| Mean multirate | `10.0114 ms` | `99.89` | `1.13x` |
| Stride multirate | `9.8396 ms` | `101.63` | `1.11x` |

### Interpretation

This isolates the speed issue more cleanly.

With parameter-free transitions, the batched inference result moves in the expected direction:

- `12-13%` lower latency at batch `128`
- same parameter count as baseline
- same layer count
- only fewer token states through intermediate layers

Batch `1` is still slower because the model is very small and the implementation still pays extra Python/module/slicing overhead for the rate transitions.

This suggests the multirate path can produce real throughput gains when enough work is batched, but the current scratch implementation is not optimized enough to reduce single-sample latency.

Important caveat:

- These parameter-free modes have not yet been trained for accuracy.
- The next accuracy check should train `mean` mode first, because it is less lossy than pure stride while preserving the no-extra-parameter timing control.

## 2026-04-20: Eight-Layer 8x Middle Multirate ViT

### Goal

Add two middle layers at `1/8` patch-token rate and compare accuracy against the earlier tiny ViT controls.

The target schedule is:

`1, 1/2, 1/4, 1/8, 1/8, 1/4, 1/2, 1`

CLI denominator form:

`1,2,4,8,8,4,2,1`

For MNIST patch tokens, this gives:

`49 -> 25 -> 13 -> 7 -> 7 -> 13 -> 25 -> 49`

So the deepest two transformer layers operate on only `7` patch tokens plus `CLS`.

### Smoke check

Command:

```powershell
python main.py --dataset mnist --epochs 1 --warmup_epochs 1 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_smoke --model_path ./checkpoints_multirate_12488421_smoke --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1
```

Result:

- parameters: `327,498`
- epoch-1 train accuracy: `82.86%`
- epoch-1 test accuracy: `88.81%`
- epoch-1 test loss: `0.4544`

This confirmed the deeper schedule trains cleanly.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421 --model_path ./checkpoints_multirate_12488421 --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1
```

Result:

- parameters: `327,498`
- final train accuracy: `98.83%`
- final test accuracy: `98.91%`
- best test accuracy: `98.91%`
- final test loss: `0.0329`
- duration: `21:47`

### Comparison

| Model | Layers | Patch-token schedule | Minimum patch tokens | Parameters | Final test accuracy | Best test accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Baseline scratch ViT | `6` | `49,49,49,49,49,49` | `49` | `210,058` | `98.98%` | `98.98%` |
| Step-down / step-up | `6` | `49,25,13,13,25,49` | `13` | `243,722` | `98.82%` | `98.91%` |
| 8x middle | `8` | `49,25,13,7,7,13,25,49` | `7` | `327,498` | `98.91%` | `98.91%` |

### Interpretation

The `1/8` middle schedule works.

Adding two deeper, heavily compressed middle layers did not hurt MNIST accuracy. The model ended at `98.91%`, matching the best observed accuracy from the 6-layer `1/4` middle schedule and staying within `0.07%` absolute of the 6-layer full-context baseline.

This is encouraging because it suggests the model can tolerate an even more aggressive central representation:

- `7` patch tokens plus `CLS` in the deepest layers
- learned return path back to full `49` patch-token resolution
- no progressive growth needed for this MNIST control

Important caveat:

- The model has more parameters and more layers than the 6-layer baseline.
- The better comparison for efficiency is a same-depth 8-layer full-context baseline plus timing/FLOP estimates.

### Next checks

1. Run an 8-layer full-context baseline.
2. Benchmark inference timing for 8-layer baseline vs 8-layer `1,2,4,8,8,4,2,1`.
3. Try the same schedule on FashionMNIST.
4. If FashionMNIST degrades, try progressive growth from `1,2,4,4,2,1` into `1,2,4,8,8,4,2,1`.

## 2026-04-21: Baseline-Like Bypass Multirate ViT

### Goal

Remove all learned transition circuitry and test the cleanest version of the execution-saving idea:

- same ordinary ViT encoder layers
- same layer count as an 8-layer baseline
- no learned compress projections
- no learned expand projections
- no transition LayerNorms
- no adapters
- fewer token states execute through compressed middle layers

### Implementation

Added:

```powershell
--rate_transition_mode bypass
```

In this mode the model does not instantiate transition submodules. The only extra runtime state is a stack of skipped patch-token tensors used to restore the sequence during the up-rate half of the schedule.

Bypass transition semantics:

- downsample: keep every other patch token active and store skipped tokens on a stack
- upsample: reinsert the stored skipped tokens in their original interleaved positions
- `CLS` is never skipped or compressed
- only active tokens execute through the next transformer layer

For the 8-layer schedule:

`1, 1/2, 1/4, 1/8, 1/8, 1/4, 1/2, 1`

the active patch-token count is:

`49 -> 25 -> 13 -> 7 -> 7 -> 13 -> 25 -> 49`

The skipped tokens are carried around the compressed stack and restored on the way up.

### Timing control

Compared against an 8-layer full-context baseline with the same parameter count.

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --n_layers 8
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode bypass --n_layers 8
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode bypass --n_layers 8
```

Results:

| Model | Batch | Parameters | Mean latency | Images/sec |
| --- | ---: | ---: | ---: | ---: |
| 8-layer baseline | `128` | `277,002` | `16.7458 ms` | `7643.73` |
| 8-layer bypass multirate | `128` | `277,002` | `13.5508 ms` | `9445.94` |
| 8-layer bypass multirate | `1` | `277,002` | `11.8356 ms` | `84.49` |

Batch throughput improved by about `19%` against the same-depth baseline.

### Smoke check

Command:

```powershell
python main.py --dataset mnist --epochs 1 --warmup_epochs 1 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_bypass_smoke --model_path ./checkpoints_multirate_12488421_bypass_smoke --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode bypass
```

Result:

- parameters: `277,002`
- epoch-1 train accuracy: `48.36%`
- epoch-1 test accuracy: `53.48%`
- epoch-1 test loss: `1.2013`

This was much slower early learning than the learned-transition 8-layer model.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_bypass --model_path ./checkpoints_multirate_12488421_bypass --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode bypass
```

Result:

- parameters: `277,002`
- final train accuracy: `97.89%`
- final test accuracy: `98.48%`
- best test accuracy: `98.48%`
- final test loss: `0.0515`
- duration: `21:35`

### Comparison

| Model | Layers | Transition mode | Parameters | Final test accuracy | Batch-128 latency |
| --- | ---: | --- | ---: | ---: | ---: |
| 6-layer baseline | `6` | none | `210,058` | `98.98%` | `17.43 ms` in earlier run |
| 8-layer baseline | `8` | none | `277,002` | not trained yet | `16.75 ms` |
| 8-layer learned `1,2,4,8,8,4,2,1` | `8` | learned projections | `327,498` | `98.91%` | not remeasured in this table |
| 8-layer bypass `1,2,4,8,8,4,2,1` | `8` | skip/reinsert only | `277,002` | `98.48%` | `13.55 ms` |

### Interpretation

This is the cleanest control so far for the user's hypothesis:

- The transformer layers are ordinary ViT layers.
- There are no transition parameters.
- There are no transition LayerNorms.
- The only structural change is that some tokens skip some middle layers and are reinserted later.

The result is encouraging but weaker than the learned-transition model:

- bypass mode is much faster in batched inference
- bypass mode preserves high MNIST accuracy
- bypass mode is still `0.50%` absolute below the 6-layer baseline and `0.43%` below the learned 8x-middle model

This suggests the learned transitions are helpful for optimization and/or information mixing, but they are not required for the basic multirate idea to work.

### Next checks

1. Train an 8-layer full-context baseline for direct accuracy comparison.
2. Try bypass mode on FashionMNIST.
3. Try a hybrid: bypass down/up plus a tiny learned gate or residual scale, not a full projection.
4. Try progressive growth from the trained learned model into bypass mode only if bypass stalls on harder datasets.

## 2026-04-21: Parameter-Free Mean 8x Middle ViT

### Goal

Test whether keeping sibling information during down-rate transitions is better than bypass's hard skip/reinsert rule, while still avoiding learned transition circuitry.

This uses the same 8-layer schedule:

`1, 1/2, 1/4, 1/8, 1/8, 1/4, 1/2, 1`

CLI denominator form:

`1,2,4,8,8,4,2,1`

### Implementation

Command mode:

```powershell
--rate_transition_mode mean
```

Transition semantics:

- downsample: average adjacent patch-token groups
- upsample: repeat compressed patch tokens
- `CLS` remains unchanged
- no learned transition projections
- no transition LayerNorms
- same trainable parameter count as an equal-depth full-context ViT

### Smoke check

Command:

```powershell
python main.py --dataset mnist --epochs 1 --warmup_epochs 1 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_mean_smoke --model_path ./checkpoints_multirate_12488421_mean_smoke --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Result:

- parameters: `277,002`
- epoch-1 train accuracy: `55.79%`
- epoch-1 test accuracy: `56.98%`
- epoch-1 test loss: `1.0847`

For comparison, bypass mode reached `53.48%` test accuracy after one epoch.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_mean --model_path ./checkpoints_multirate_12488421_mean --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Result:

- parameters: `277,002`
- final train accuracy: `98.48%`
- final test accuracy: `98.74%`
- best test accuracy: `98.74%`
- final test loss: `0.0413`
- duration: `21:37`

### Timing

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Results:

| Model | Batch | Parameters | Mean latency | Images/sec |
| --- | ---: | ---: | ---: | ---: |
| 8-layer mean multirate | `128` | `277,002` | `13.0947 ms` | `9774.97` |
| 8-layer mean multirate | `1` | `277,002` | `11.9905 ms` | `83.40` |

### Updated comparison

| Model | Layers | Transition mode | Parameters | Final test accuracy | Batch-128 latency |
| --- | ---: | --- | ---: | ---: | ---: |
| 8-layer baseline | `8` | none | `277,002` | not trained yet | `16.75 ms` |
| 8-layer bypass `1,2,4,8,8,4,2,1` | `8` | skip/reinsert only | `277,002` | `98.48%` | `13.55 ms` |
| 8-layer mean `1,2,4,8,8,4,2,1` | `8` | average/repeat | `277,002` | `98.74%` | `13.09 ms` |
| 8-layer learned `1,2,4,8,8,4,2,1` | `8` | learned projections | `327,498` | `98.91%` | not remeasured |

### Interpretation

Mean transitions are a better no-extra-parameter control than bypass on MNIST.

Compared with bypass, mean mode:

- improves final test accuracy from `98.48%` to `98.74%`
- keeps the same `277,002` trainable parameters
- slightly improves batch-128 latency in this measurement
- still trails the learned-transition model by `0.17%` absolute accuracy

This supports the suspicion that discarding every other active token is unnecessarily harsh. Simply carrying the average of sibling patch states through the compressed middle layers recovers about half of the gap between hard bypass and learned transitions.

## 2026-04-21: Eight-Layer Full-Context Baseline

### Goal

Train the direct 8-layer non-compressive control for the 8-layer multirate experiments.

This checks whether the 8-layer baseline degrades under the same training recipe, which would make the multirate results easier to overinterpret.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_baseline_8layer --model_path ./checkpoints_baseline_8layer --data_path ./data --n_layers 8
```

Result:

- parameters: `277,002`
- final train accuracy: `99.00%`
- final test accuracy: `99.05%`
- best test accuracy: `99.05%`
- final test loss: `0.0304`
- duration: `25:05`

### Updated comparison

| Model | Layers | Transition mode | Parameters | Final test accuracy | Batch-128 latency |
| --- | ---: | --- | ---: | ---: | ---: |
| 6-layer baseline | `6` | none | `210,058` | `98.98%` | `18.65 ms` checkpoint run |
| 8-layer baseline | `8` | none | `277,002` | `99.05%` | `16.75 ms` no-checkpoint timing |
| 8-layer bypass `1,2,4,8,8,4,2,1` | `8` | skip/reinsert only | `277,002` | `98.48%` | `13.55 ms` |
| 8-layer mean `1,2,4,8,8,4,2,1` | `8` | average/repeat | `277,002` | `98.74%` | `13.09 ms` |
| 8-layer learned `1,2,4,8,8,4,2,1` | `8` | learned projections | `327,498` | `98.91%` | not remeasured |

### Interpretation

The 8-layer full-context baseline did not degrade. It slightly outperformed the 6-layer baseline and all current 8-layer multirate variants on MNIST accuracy.

That makes the current 8-layer multirate result a speed/parameter-efficiency tradeoff, not an accuracy win:

- mean multirate is `0.31%` absolute below the 8-layer full-context baseline
- mean multirate is about `22%` faster than the same-depth full-context timing control at batch `128`
- both models have the same trainable parameter count, but the multirate model executes fewer token states in the middle layers

The next natural test is width reduction: check whether the multirate schedule can preserve more accuracy than a full-context baseline when both are reduced from `64` to `32` embedding dimensions.

## 2026-04-21: Width-Reduced Mean Multirate ViT

### Goal

Test whether the 8-layer mean multirate schedule still performs well when reducing channel width from `64` to `32`.

This checks whether the multirate sparse-activation structure can preserve accuracy under a much smaller embedding dimension.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_mean_d32 --model_path ./checkpoints_multirate_12488421_mean_d32 --data_path ./data --n_layers 8 --embed_dim 32 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Result:

- parameters: `71,946`
- final train accuracy: `97.05%`
- final test accuracy: `97.92%`
- best test accuracy: `97.92%`
- final test loss: `0.0732`
- duration: `23:46`

### Comparison

| Model | Width | Layers | Transition mode | Parameters | Final test accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| 8-layer baseline | `64` | `8` | none | `277,002` | `99.05%` |
| 8-layer mean multirate | `64` | `8` | average/repeat | `277,002` | `98.74%` |
| 8-layer mean multirate | `32` | `8` | average/repeat | `71,946` | `97.92%` |

### Interpretation

The 32-dim mean multirate model stayed strong despite cutting trainable parameters by about `74%` versus the 64-dim 8-layer baseline.

This does not yet prove the multirate schedule is better than a 32-dim full-context baseline, because that direct control has not been trained yet. But it is a useful result: the reduced-width multirate model remains close to the original 6-layer 64-dim baseline while using far fewer parameters.

The next required control is:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_baseline_8layer_d32 --model_path ./checkpoints_baseline_8layer_d32 --data_path ./data --n_layers 8 --embed_dim 32 --n_attention_heads 4
```

## 2026-04-21: Width-Reduced Full-Context Baseline

### Goal

Train the direct 32-dim full-context baseline for the width-reduction comparison.

This checks whether the 32-dim mean multirate result is actually helped by the multirate schedule, or whether a narrow full-context ViT performs similarly.

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_baseline_8layer_d32 --model_path ./checkpoints_baseline_8layer_d32 --data_path ./data --n_layers 8 --embed_dim 32 --n_attention_heads 4
```

Result:

- parameters: `71,946`
- final train accuracy: `96.52%`
- final test accuracy: `97.38%`
- best test accuracy: `97.38%`
- final test loss: `0.0844`
- duration: `23:35`

### Updated width comparison

| Model | Width | Layers | Transition mode | Parameters | Final test accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| 8-layer baseline | `32` | `8` | none | `71,946` | `97.38%` |
| 8-layer mean multirate | `32` | `8` | average/repeat | `71,946` | `97.92%` |
| 8-layer baseline | `64` | `8` | none | `277,002` | `99.05%` |
| 8-layer mean multirate | `64` | `8` | average/repeat | `277,002` | `98.74%` |

### Interpretation

The 32-dim mean multirate model outperformed the 32-dim full-context baseline by `0.54%` absolute with the same parameter count.

This is currently the cleanest support for the channel-width hypothesis:

- at width `64`, full-context wins on accuracy
- at width `32`, mean multirate wins on accuracy
- both 32-dim models have identical trainable parameter counts

This suggests the multirate schedule may be acting as a useful sparse/hierarchical activation structure when channel capacity is constrained.

## 2026-04-21: Convolutional Rate-Transition ViT

### Goal

Test a "convolutional attention" variant while preserving explicit multirate compression.

The idea is not to remove the rate schedule. Instead, replace simple mean or linear projection transitions with local convolutional pattern matching:

- down-rate transitions use local convolution over patch tokens
- attention still runs at the compressed scale
- up-rate transitions use local transposed convolution
- the same `1,2,4,8,8,4,2,1` token-rate schedule is preserved

### Implementation

Added:

```powershell
--rate_transition_mode conv
```

Transition modules:

- `ConvPatchCompressor`: depthwise `Conv1d(kernel=3, stride=2, padding=1)` plus pointwise `Conv1d(kernel=1)` and `LayerNorm`
- `ConvPatchExpander`: depthwise `ConvTranspose1d(kernel=4, stride=2, padding=1)` plus pointwise `Conv1d(kernel=1)` and `LayerNorm`
- `CLS` remains unchanged

This is a learned local transition, but it is smaller than the learned flatten/projection transition:

- learned projection transition model: `327,498` params
- conv transition model: `304,458` params
- mean/bypass/full 8-layer baseline: `277,002` params

### Smoke check

Command:

```powershell
python main.py --dataset mnist --epochs 1 --warmup_epochs 1 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_conv_smoke --model_path ./checkpoints_multirate_12488421_conv_smoke --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode conv
```

Result:

- parameters: `304,458`
- epoch-1 train accuracy: `33.63%`
- epoch-1 test accuracy: `35.89%`
- epoch-1 test loss: `1.6020`

### Full run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_multirate_12488421_conv --model_path ./checkpoints_multirate_12488421_conv --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode conv
```

Result:

- parameters: `304,458`
- final train accuracy: `99.07%`
- final test accuracy: `99.01%`
- best test accuracy: `99.05%`
- final test loss: `0.0298`
- duration: `26:55`

### Timing

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode conv
python benchmark_inference.py --batch_size 1 --steps 1000 --warmup 100 --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode conv
```

Results:

| Model | Batch | Parameters | Mean latency | Images/sec |
| --- | ---: | ---: | ---: | ---: |
| 8-layer conv multirate | `128` | `304,458` | `20.4050 ms` | `6272.96` |
| 8-layer conv multirate | `1` | `304,458` | `19.0193 ms` | `52.58` |

### Updated comparison

| Model | Transition mode | Parameters | Final test accuracy | Best test accuracy | Batch-128 latency |
| --- | --- | ---: | ---: | ---: | ---: |
| 8-layer baseline | none | `277,002` | `99.05%` | `99.05%` | `16.75 ms` |
| 8-layer mean multirate | average/repeat | `277,002` | `98.74%` | `98.74%` | `13.09 ms` |
| 8-layer learned multirate | learned projections | `327,498` | `98.91%` | `98.91%` | not remeasured |
| 8-layer learned multirate + extra training | learned projections | `327,498` | `99.13%` | `99.13%` | not remeasured |
| 8-layer conv multirate | local conv/deconv | `304,458` | `99.01%` | `99.05%` | `20.41 ms` |

### Interpretation

The convolutional transition idea works for accuracy but not for speed in this naive implementation.

It reached the same best test accuracy as the 8-layer full-context baseline (`99.05%`) with fewer parameters than the learned projection transition model. That supports the idea that local convolutional transition mixing is a viable way to build compressed multiscale representations.

However, the current depthwise plus pointwise convolution modules are slow enough to erase the execution savings:

- mean multirate: `13.09 ms`
- full 8-layer baseline: `16.75 ms`
- conv multirate: `20.41 ms`

So this variant is useful as an architectural accuracy probe, not yet as an efficient inference path.

The next practical version should either:

- use mean down/up plus a very small learned local residual, or
- move convolution into a fused transition kernel, or
- test this on a harder dataset where the extra local mixing may be worth the added compute.

## 2026-04-21: Additional Epochs for 32-Dim Mean Multirate ViT

### Goal

Test whether the width-reduced mean multirate model benefits from extra low-LR training, following the same exposure-compensation idea that helped the learned 64-dim multirate model.

### Starting checkpoint

Starting model:

- width: `32`
- layers: `8`
- transition mode: `mean`
- schedule: `1,2,4,8,8,4,2,1`
- checkpoint: `external_baselines/PyTorch-Scratch-Vision-Transformer-ViT/checkpoints_multirate_12488421_mean_d32/mnist/ViT_model.pt`
- original final test accuracy: `97.92%`

### Fine-tune run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 1 --batch_size 128 --n_workers 0 --lr 0.0001 --output_path ./runs_multirate_12488421_mean_d32_ft20 --model_path ./checkpoints_multirate_12488421_mean_d32_ft20 --init_model_path ./checkpoints_multirate_12488421_mean_d32/mnist --data_path ./data --n_layers 8 --embed_dim 32 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Result:

- parameters: `71,946`
- final train accuracy: `97.77%`
- final test accuracy: `98.36%`
- best test accuracy: `98.39%`
- final test loss: `0.0575`
- duration: `22:58`

### Timing

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --n_layers 8 --embed_dim 32 --n_attention_heads 4
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --n_layers 8 --embed_dim 32 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Results:

| Model | Batch | Parameters | Mean latency | Images/sec |
| --- | ---: | ---: | ---: | ---: |
| 32-dim 8-layer baseline | `128` | `71,946` | `13.7513 ms` | `9308.22` |
| 32-dim mean multirate | `128` | `71,946` | `14.9389 ms` | `8568.21` |

### Updated width comparison

| Model | Width | Layers | Transition mode | Parameters | Final test accuracy | Best test accuracy | Batch-128 latency |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8-layer baseline | `32` | `8` | none | `71,946` | `97.38%` | `97.38%` | `13.75 ms` |
| 8-layer mean multirate | `32` | `8` | average/repeat | `71,946` | `97.92%` | `97.92%` | `14.94 ms` |
| 8-layer mean multirate + extra training | `32` | `8` | average/repeat | `71,946` | `98.36%` | `98.39%` | `14.94 ms` |

### Interpretation

The extra training helped the 32-dim mean multirate model substantially:

- `97.92% -> 98.36%` final accuracy
- `98.39%` best accuracy
- `+1.01%` absolute over the same-parameter 32-dim full-context baseline

This is currently the strongest same-parameter accuracy result for the multirate idea.

The throughput result is less favorable at width `32`. Unlike the 64-dim comparison, the 32-dim mean multirate model is slower than the full-context baseline in this implementation. At this small width, the mean/repeat transition overhead dominates the saved attention work.

So the result splits cleanly:

- accuracy: strong win under constrained channel width
- latency: not a win at width `32` in the current unfused implementation

## 2026-04-21: CIFAR-10 Larger-Image 32-Dim Control

### Goal

Move the 32-dim comparison from MNIST to a larger and harder image setting.

CIFAR-10 uses:

- image size: `32 x 32`
- input channels: `3`
- patch size: `4`
- patch tokens: `64`

The 8-layer multirate schedule becomes:

`64 -> 32 -> 16 -> 8 -> 8 -> 16 -> 32 -> 64`

### Baseline run

Command:

```powershell
python main.py --dataset cifar10 --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_baseline_8layer_d32 --model_path ./checkpoints_cifar10_baseline_8layer_d32 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 32 --n_attention_heads 4
```

Result:

- parameters: `73,450`
- final train accuracy: `52.90%`
- final test accuracy: `57.36%`
- best test accuracy: `57.46%`
- final test loss: `1.1852`
- duration: `37:10`

### Mean multirate run

Command:

```powershell
python main.py --dataset cifar10 --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_mean_8layer_d32 --model_path ./checkpoints_cifar10_mean_8layer_d32 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 32 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Result:

- parameters: `73,450`
- final train accuracy: `50.20%`
- final test accuracy: `54.65%`
- best test accuracy: `54.82%`
- final test loss: `1.2405`
- duration: `36:42`

### Timing

Commands:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 32 --n_attention_heads 4
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 32 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1 --rate_transition_mode mean
```

Results:

| Model | Batch | Parameters | Mean latency | Images/sec |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 32-dim baseline | `128` | `73,450` | `16.5157 ms` | `7750.21` |
| CIFAR-10 32-dim mean multirate | `128` | `73,450` | `13.5304 ms` | `9460.21` |

### Interpretation

CIFAR-10 reverses the MNIST accuracy result under this exact 20-epoch recipe.

The full-context 32-dim baseline beat mean multirate:

- baseline best: `57.46%`
- mean multirate best: `54.82%`
- gap: `-2.64%` absolute for mean multirate

However, larger images do restore the expected throughput advantage:

- baseline batch-128 latency: `16.52 ms`
- mean multirate batch-128 latency: `13.53 ms`
- latency reduction: about `18%`

This is a useful split result:

- speed edge appears on the larger token grid
- accuracy suffers with the parameter-free mean transition on the harder task

The likely next CIFAR-10 checks are:

1. fine-tune mean multirate for 20 extra low-LR epochs
2. try learned transitions on CIFAR-10
3. try a less aggressive schedule such as `1,2,4,4,2,1` or an 8-layer `1,1,2,4,4,2,1,1`
4. raise width to `64` to see whether the accuracy gap is a capacity issue

## 2026-04-21: CIFAR-10 64-Dim Learned Multirate ViT

### Goal

Test whether learned transitions recover the CIFAR-10 accuracy lost by the 32-dim parameter-free mean transition model.

### Run

Command:

```powershell
python main.py --dataset cifar10 --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64 --model_path ./checkpoints_cifar10_learned_8layer_d64 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Result:

- parameters: `330,506`
- final train accuracy: `62.50%`
- final test accuracy: `67.49%`
- best test accuracy: `67.49%`
- final test loss: `0.9073`
- duration: `32:10`

Checkpoint verification:

```powershell
python main.py --dataset cifar10 --eval_only --load_model True --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64 --model_path ./checkpoints_cifar10_learned_8layer_d64 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Verified result:

- test accuracy: `67.49%`
- test loss: `0.9073`

Timing command:

```powershell
python benchmark_inference.py --batch_size 128 --steps 300 --warmup 50 --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Timing result:

- batch: `128`
- mean latency: `11.6790 ms`
- throughput: `10959.84 images/sec`

### Interpretation

The learned 64-dim compressive model is the strongest CIFAR-10 result so far, but it still needs the matched 64-dim full-context baseline before we can claim an advantage.

Current CIFAR-10 comparison:

| Model | Width | Transition | Parameters | Final test accuracy | Batch-128 latency |
| --- | ---: | --- | ---: | ---: | ---: |
| Full context | `32` | none | `73,450` | `57.36%` | `16.5157 ms` |
| Multirate | `32` | mean | `73,450` | `54.65%` | `13.5304 ms` |
| Multirate | `64` | learned | `330,506` | `67.49%` | `11.6790 ms` |

The 64-dim learned model is faster than both recorded 32-dim CIFAR-10 runs in this benchmark harness, which is likely a GPU utilization/kernel-shape effect rather than a pure FLOP story. The missing control is:

```powershell
python main.py --dataset cifar10 --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_baseline_8layer_d64 --model_path ./checkpoints_cifar10_baseline_8layer_d64 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4
```

## 2026-04-21: Additional Epochs for Learned 8x Middle ViT

### Goal

Test whether the best learned compressive model benefits from extra training after the initial 20-epoch run.

The motivation is that compressed middle layers process fewer patch-token states per batch. They receive the same number of optimizer steps as full-resolution layers, but less token-level gradient exposure. Additional fine-tuning may partly compensate.

### Starting checkpoint

Starting model:

- 8-layer learned multirate schedule: `1,2,4,8,8,4,2,1`
- checkpoint: `external_baselines/PyTorch-Scratch-Vision-Transformer-ViT/checkpoints_multirate_12488421/mnist/ViT_model.pt`
- original final test accuracy: `98.91%`
- original final test loss: `0.0329`

### Fine-tune run

Command:

```powershell
python main.py --dataset mnist --epochs 20 --warmup_epochs 1 --batch_size 128 --n_workers 0 --lr 0.0001 --output_path ./runs_multirate_12488421_ft20 --model_path ./checkpoints_multirate_12488421_ft20 --init_model_path ./checkpoints_multirate_12488421/mnist --data_path ./data --n_layers 8 --multirate_schedule 1,2,4,8,8,4,2,1
```

Result:

- parameters: `327,498`
- final train accuracy: `99.11%`
- final test accuracy: `99.13%`
- best test accuracy: `99.13%`
- final test loss: `0.0275`
- duration: `24:38`

### Comparison

| Model | Layers | Transition mode | Parameters | Training recipe | Final test accuracy |
| --- | ---: | --- | ---: | --- | ---: |
| 8-layer baseline | `8` | none | `277,002` | 20 epochs | `99.05%` |
| 8-layer learned multirate | `8` | learned projections | `327,498` | 20 epochs | `98.91%` |
| 8-layer learned multirate | `8` | learned projections | `327,498` | 20 epochs + 20 low-LR fine-tune | `99.13%` |

### Interpretation

The extra training helped.

The learned 8x-middle model moved from `98.91%` to `99.13%`, crossing the same-depth full-context baseline in this run.

This supports the hypothesis that compressed stages may need additional effective training exposure. The current fine-tune updated all weights, so it does not isolate whether the gain came specifically from middle compressed layers. The cleaner next test is a targeted fine-tune where the full-resolution layers are frozen and only compressed middle blocks plus transition modules are updated.

## 2026-04-21: CIFAR-10 64-Dim Learned Multirate Long Run

### Goal

Compare a small learned-compressive ViT against a 64-dim Mamba patch classifier on CIFAR-10, then test whether the compressive model keeps improving with additional epochs.

### Models

Compressive ViT:

- schedule: `1,2,4,8,8,4,2,1`
- image: `32x32x3`
- patch size: `4`
- layers: `8`
- width: `64`
- attention heads: `4`
- parameters: `330,506`

Mamba control:

- patch size: `4`
- layers: `6`
- width: `64`
- state dim: `16`
- parameters: `200,362`

### Key Commands

Initial 7-epoch multirate check:

```powershell
python main.py --dataset cifar10 --epochs 7 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64_e7 --model_path ./checkpoints_cifar10_learned_8layer_d64_e7 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Continuation to 20 epochs from epoch 7:

```powershell
python main.py --dataset cifar10 --epochs 20 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64_e20_from_e7 --model_path ./checkpoints_cifar10_learned_8layer_d64_e20_from_e7 --init_model_path ./checkpoints_cifar10_learned_8layer_d64_e7/cifar10 --start_epoch 7 --best_acc 0.5237 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Full continuation from the 24-epoch selective-tune checkpoint to 34 epochs:

```powershell
python main.py --dataset cifar10 --epochs 34 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64_e34_full_from_e24 --model_path ./checkpoints_cifar10_learned_8layer_d64_e34_full_from_e24 --init_model_path ./checkpoints_cifar10_learned_8layer_d64_e24_outer25_ft/cifar10 --start_epoch 24 --best_acc 0.6861 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

Full continuation from epoch 34 to 44:

```powershell
python main.py --dataset cifar10 --epochs 44 --warmup_epochs 5 --batch_size 128 --n_workers 0 --output_path ./runs_cifar10_learned_8layer_d64_e44_full_from_e34 --model_path ./checkpoints_cifar10_learned_8layer_d64_e44_full_from_e34 --init_model_path ./checkpoints_cifar10_learned_8layer_d64_e34_full_from_e24/cifar10 --start_epoch 34 --best_acc 0.7029 --data_path ./data --image_size 32 --n_channels 3 --n_layers 8 --embed_dim 64 --n_attention_heads 4 --multirate_schedule 1,2,4,8,8,4,2,1
```

### Results

Mamba baseline/control:

- epoch 17: `55.41%`
- epoch 19: `55.50%`
- epoch 20: `55.32%`
- best: `55.50%`

Learned multirate 64-dim:

- epoch 7: `52.37%`
- epoch 20: `68.00%`, loss `0.9006`
- selective middle/adjacent fine-tune best: `68.61%`
- epoch 30: `69.24%`, loss `0.8690`
- epoch 33: `70.29%`, loss `0.8427`
- epoch 36: `70.67%`, loss `0.8319`
- epoch 40: `70.95%`, loss `0.8175`
- epoch 42: `71.62%`, loss `0.8031`
- epoch 43: `71.78%`, loss `0.7972`
- epoch 44: `71.82%`, loss `0.7958`

### Interpretation

The 64-dim learned-compressive model decisively separated from the Mamba patch classifier control. Mamba appeared to plateau near `55%`, while the learned multirate ViT continued climbing through epoch 44.

The late-epoch gains are notable because the middle two blocks operate at `1/8` patch-token resolution. This is consistent with the working hypothesis that compressed middle stages may need more update exposure before they fully exploit their reduced-rate representation.

### Follow-Up Note

Resume the 64-dim learned multirate checkpoint beyond epoch 44 later. The curve was still improving at the end of the run, so it is worth checking whether it keeps climbing, saturates, or shows a grokking-like delayed jump.

Next scale-up: train the same learned schedule at width `128` for `40` epochs and compare against the known 256-dim, 200-epoch benchmark.

## 2026-04-23: Shakespeare Repeat at 256-512 Characters with a 64-Dim Hymba

### Goal

Test whether a very small `d=64` multirate Hymba-like model can:

- solve a character-level delayed repeat task at `256` characters
- adapt to tighter attention context caps without immediate collapse
- transfer a recovered capped checkpoint to a harder `512`-character repeat task

This is the most paper-ready repeat setup currently in the repo because it has:

- explicit run scripts
- fixed data generation arguments
- stored sample dumps for qualitative inspection
- a staged adaptation story with both successful and failed branches

### Task and data

Data source:

- `data/tiny_shakespeare.txt`

Tokenizer:

- pure character-level repeat vocabulary built from normalized Shakespeare text

Repeat template:

- source span of length `mem_len`
- blank bridge of length `128`, contributing `127` input tokens after causal shift
- recall side of length `mem_len + 1`

So the effective input lengths for the main settings are:

- `mem_len=256`: `256 + 127 + 257 = 640`
- `mem_len=512`: `512 + 127 + 513 = 1152`

The Hymba stack also prepends `16` meta tokens, so the first stage sees:

- `656` tokens for the `256` task
- `1168` tokens for the `512` task

To avoid the sentence-span bottleneck, repeat data was generated with contiguous non-overlapping chunk spans inside the train/test split:

- `--span_mode chunk`
- `--chunk_span_len 2048`

This preserves train/test separation while allowing `mem_len >= 512`.

### Shared architecture

Model family:

- Hymba-like hybrid with parallel Mamba + causal RoPE attention

Shared core settings:

- layers: `12`
- width: `64`
- heads: `4`
- KV heads: `2`
- Mamba state dim: `16`
- meta tokens: `16`
- local attention window: `128`
- full-attention layers: `1,4,9,12`
- multirate schedule: `1,1,2,4,8,16,16,8,4,2,1,1`
- dropout: `0.03`
- optimizer: `RMSprop`
- learning rate: `1.5e-4` for the `256` cap-480 run, `1e-4` for the `512` transfer run
- weight decay: `1e-5`
- repeat noise probability: `0.02`
- batch size: `4` for `mem_len=256`, `1` for `mem_len=512`

Parameter count reported by the runner:

- `1,000,066`

### Implementation changes required for these runs

Code paths:

- `external_baselines/tcn_text_repeat_runner.py`
- `external_baselines/hymba_text_repeat_runner.py`
- `external_baselines/hymba_shakespeare_lm_runner.py`
- `sample_hymba_non_exact.py`

Main changes:

1. Added chunk-based repeat spans with `--span_mode chunk --chunk_span_len 2048`.
2. Added `--attention_context_cap` support to the Hymba attention path.
3. Replaced the first masked-cap prototype with a real sliced/local K/V attention path for capped local attention.
4. Kept full sequence hidden states intact while limiting attention reach on capped layers.

### Reproduction scripts

Primary run scripts:

- `run_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_from192_e80.ps1`
- `run_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap480_from192_e80.ps1`
- `run_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap320_from_cap480_e80.ps1`
- `run_hymba_text_repeat_mr11248161684211_d64_l12_mem512_chunk2048_cap480_from_cap480_e80.ps1`

These scripts encode the exact task and architecture arguments used in the results below.

### Key quantitative results

#### Baseline chunked `mem_len=256`

Run:

- `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_from192_e80`

Initialization:

- resumed from the earlier `mem192` best-recall checkpoint

Early signal:

- epoch 1: `85.29%` recall, `0.00%` exact, `273.6s/epoch`

Interpretation:

- the chunked dataset is stricter than the earlier sentence-span setup, but still learnable

#### Hard cap failures

`cap=80` on the `256` task:

- resumed and scratch versions both collapsed early
- scratch run epoch 1: `18.26%` recall, `0.00%` exact, `374.4s/epoch`

Interpretation:

- a direct `1/8` cap is too aggressive in this formulation

#### Successful staged cap at `480`

Run:

- `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap480_from192_e80`

Initialization:

- resumed from `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem192_from128_e60\best_recall_checkpoint.pt`

First logged epochs:

- epoch 1: `85.10%` recall, `0.00%` exact, `273.4s`
- epoch 2: `86.29%` recall, `0.00%` exact, `268.6s`

Best recovered metrics from later checkpoints:

- best recall: `99.9984%`
- best exact: `99.6%`

Interpretation:

- staged adaptation at `cap=480` fully recovered the `256` repeat task
- this is the main positive result: a `d=64` model recovered nearly perfect delayed recall under a context cap below the full active sequence length

#### Harder staged cap at `320`

Run:

- `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap320_from_cap480_e80`

Initialization:

- resumed from `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap480_from192_e80\best_recall_checkpoint.pt`

First nine logged epochs:

- epoch 1: `52.54%` recall, `0.00%` exact, `350.1s`
- epoch 2: `60.42%` recall, `0.00%` exact, `351.0s`
- epoch 3: `63.21%` recall, `0.00%` exact, `320.1s`
- epoch 4: `65.52%` recall, `0.00%` exact, `301.9s`
- epoch 5: `67.18%` recall, `0.00%` exact, `300.2s`
- epoch 6: `67.87%` recall, `0.00%` exact, `303.5s`
- epoch 7: `68.70%` recall, `0.00%` exact, `300.9s`
- epoch 8: `70.18%` recall, `0.00%` exact, `303.3s`
- epoch 9: `71.02%` recall, `0.00%` exact, `303.1s`

Interpretation:

- `480 -> 320` is a much harder step than `uncapped/192 -> 480`
- staged compression still gives gradual recovery, but not the near-immediate survivability seen at `480`

### Transfer to `mem_len=512`

Run:

- `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256to512_chunk2048_cap480_from_cap480_e80`

Initialization:

- resumed from `runs_hymba_text_repeat_mr11248161684211_d64_l12_mem256_chunk2048_cap480_from192_e80\best_recall_checkpoint.pt`

Task change:

- source repeat length increased from `256` to `512`
- attention cap remained at `480`

This is a sharper test than the `256` capped run because the cap is now shorter than the source span itself.

Current status:

- two epochs completed
- this run was launched in foreground/transcript mode, so the terminal transcript does not currently include the epoch metrics lines that `train_stdout.log` captures in other runs
- sample dumps for epochs 1 and 2 do exist

What is already clear qualitatively:

- the run did **not** collapse immediately
- epoch-1 samples were poor but recognizably structured
- epoch-2 samples improved somewhat and stayed far above the `cap80` / blank-collapse failure mode

This run is therefore promising, but not yet ready for a paper table because the current transcript path did not preserve the numeric epoch summaries.

### Qualitative interpretation

The qualitative pattern across these runs is consistent:

- easy or recovered settings preserve exact character identity
- middling settings preserve syntax, speaker/type information, and local orthography while drifting on names and rare substrings
- failure settings partially blank or slurry the second half of the recall span

We used a lightweight shorthand for reporting sample quality in `RESULTS.md`:

- `0-1 beers`: tiny one-character slips
- `2-3 beers`: phonetic/name drift, but intelligible
- `4-6 beers`: warped but still recognizably Shakespearean
- `7+ beers`: slurry, blanks, or low-information mush
- `12 Pack Shakespeare`: total collapse

Rough mapping:

- recovered `cap480` at `256`: `0-1 beers`
- early `cap320`: `4-7 beers`
- early `512` from recovered `cap480`: between `2-4 beers` and `12 Pack Shakespeare` depending on epoch; it showed noticeable qualitative improvement after the first bad samples

### Main takeaways

1. A `12`-layer `d=64` Hymba-like model can solve a delayed `256`-character Shakespeare repeat task nearly perfectly.
2. A direct harsh cap (`80`) fails, even from scratch.
3. Staged context compression works at `cap=480`, recovering to `99.9984%` recall and `99.6%` exact.
4. Further compression to `320` is possible in principle but adapts much more slowly.
5. Transfer from the recovered `256` capped checkpoint to a `512` repeat task is not an immediate collapse, which supports the hypothesis that the hierarchy is doing real compression/retrieval work rather than only brute-force local lookup.

### Reporting caveats

For formal reporting, use:

- the `cap480` and `cap320` runs for hard quantitative claims
- the `512` run only as a qualitative/ongoing result unless the numeric per-epoch summaries are recovered from a better logging path

If this line of work becomes a paper section, the next clean replication step is:

1. rerun the `512` transfer with terminal-visible output **and** a persistent `train_stdout.log`
2. continue the `cap320` run long enough to determine whether it plateaus or eventually recovers
3. decide whether to compare against a cleaner control-signal design that does not store the phase/marker token as ordinary memory content
