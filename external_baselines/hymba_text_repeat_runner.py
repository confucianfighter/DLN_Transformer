import argparse

from hymba_shakespeare_lm_runner import (
    HymbaLM,
    build_vocab,
    count_params,
    evaluate,
    load_text,
    parse_layer_indices,
    parse_schedule,
)
from tcn_text_repeat_runner import build_vocab as build_repeat_vocab
from tcn_text_repeat_runner import generate_repeat_data, normalize_text

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@torch.no_grad()
def predict_repeat_tokens(model, x, batch_size, temperature, top_k, device):
    model.eval()
    preds = []
    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size].to(device)
        logits = model(xb) / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(2, indices, values)
            logits = filtered
        sampled = torch.multinomial(torch.softmax(logits.reshape(-1, logits.size(-1)), dim=-1), 1)
        preds.append(sampled.reshape(logits.size(0), logits.size(1)).cpu())
    return torch.cat(preds, dim=0)


def build_self_mixed_repeat_batch(
    x,
    y,
    pred,
    interval,
    span_min,
    span_max,
    insert_prob,
    seed,
    pad_id,
    mix_start,
    mix_end,
):
    rng = random.Random(seed)
    mixed_x = []
    mixed_y = []
    seq_len = x.size(1)
    for row in range(x.size(0)):
        input_row = []
        target_row = []
        i = 0
        next_corrupt = rng.randint(max(1, interval // 2), max(1, interval + interval // 2))
        while len(input_row) < seq_len and i < seq_len:
            in_mix_region = mix_start <= i < mix_end
            if in_mix_region and i >= next_corrupt:
                span = rng.randint(span_min, span_max)
                span = min(span, seq_len - i)
                if rng.random() < insert_prob:
                    for j in range(span):
                        if len(input_row) >= seq_len:
                            break
                        input_row.append(int(pred[row, i + j]))
                        target_row.append(-100)
                else:
                    for j in range(span):
                        if len(input_row) >= seq_len:
                            break
                        input_row.append(int(pred[row, i + j]))
                        target_row.append(-100)
                    i += span
                next_corrupt += rng.randint(max(1, interval // 2), max(1, interval + interval // 2))
                continue
            input_row.append(int(x[row, i]))
            target_row.append(int(y[row, i]))
            i += 1
        while len(input_row) < seq_len:
            input_row.append(pad_id)
            target_row.append(-100)
        mixed_x.append(input_row)
        mixed_y.append(target_row)
    return torch.tensor(mixed_x, dtype=torch.long), torch.tensor(mixed_y, dtype=torch.long)


def corrupt_repeat_inputs(x, y, mem_len, n_chars, noise_prob, ablate_suffix, seed):
    if noise_prob <= 0 and ablate_suffix <= 0:
        return x, y
    if noise_prob < 0 or noise_prob > 1:
        raise ValueError("repeat_noise_prob must be in [0, 1]")
    if ablate_suffix < 0 or ablate_suffix > mem_len:
        raise ValueError("repeat_ablate_suffix must be in [0, mem_len]")
    corrupted = x.clone()
    masked_y = y.clone()
    repeat_start = corrupted.size(1) - mem_len
    repeat_region = corrupted[:, repeat_start:]
    target_region = masked_y[:, repeat_start:]
    generator = torch.Generator(device=corrupted.device)
    generator.manual_seed(seed)
    if noise_prob > 0:
        noise_mask = torch.rand(repeat_region.shape, generator=generator, device=corrupted.device) < noise_prob
        random_tokens = torch.randint(
            1,
            n_chars + 1,
            repeat_region.shape,
            generator=generator,
            device=corrupted.device,
            dtype=corrupted.dtype,
        )
        repeat_region[noise_mask] = random_tokens[noise_mask]
        target_region[noise_mask] = -100
    if ablate_suffix > 0:
        repeat_region[:, -ablate_suffix:] = 0
        target_region[:, -ablate_suffix:] = -100
    corrupted[:, repeat_start:] = repeat_region
    masked_y[:, repeat_start:] = target_region
    return corrupted, masked_y


def ablate_source_inputs(x, mem_len, source_ablate_suffix):
    if source_ablate_suffix <= 0:
        return x
    if source_ablate_suffix > mem_len:
        raise ValueError("source_ablate_suffix must be in [0, mem_len]")
    ablated = x.clone()
    ablated[:, mem_len - source_ablate_suffix : mem_len] = 0
    return ablated


@torch.no_grad()
def apply_wrong_guess_mix(model, x, y, mem_len, mix_prob, seed):
    if mix_prob <= 0:
        return x, y, 0, 0
    if mix_prob > 1:
        raise ValueError("wrong_guess_mix_prob must be in [0, 1]")
    logits = model(x)
    pred = logits.argmax(dim=-1)
    mixed_x = x.clone()
    mixed_y = y.clone()
    repeat_start = x.size(1) - mem_len
    pred_region = pred[:, repeat_start:]
    target_region = y[:, repeat_start:]
    wrong_mask = (target_region != -100) & (pred_region != target_region)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    sample_mask = torch.rand(wrong_mask.shape, generator=generator, device=x.device) < mix_prob
    replace_mask = wrong_mask & sample_mask
    mixed_x[:, repeat_start:][replace_mask] = pred_region[replace_mask]
    mixed_y[:, repeat_start:][replace_mask] = -100
    return mixed_x, mixed_y, replace_mask.sum().item(), wrong_mask.sum().item()


def epoch_ablate_suffix(args, epoch, start_epoch):
    suffix = args.repeat_ablate_suffix + (epoch - start_epoch) * args.repeat_ablate_suffix_step_per_epoch
    if args.repeat_ablate_suffix_max >= 0:
        suffix = min(suffix, args.repeat_ablate_suffix_max)
    return suffix


def epoch_source_ablate_suffix(args, epoch, start_epoch):
    suffix = args.source_ablate_suffix + (epoch - start_epoch) * args.source_ablate_suffix_step_per_epoch
    if args.source_ablate_suffix_max >= 0:
        suffix = min(suffix, args.source_ablate_suffix_max)
    return suffix


def epoch_noise_prob(args, epoch, start_epoch):
    noise_prob = args.repeat_noise_prob + (epoch - start_epoch) * args.repeat_noise_step_per_epoch
    if args.repeat_noise_max >= 0:
        noise_prob = min(noise_prob, args.repeat_noise_max)
    return noise_prob


def repeat_evaluate(model, x, y, criterion, batch_size, device, mem_len, source_ablate_suffix=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    recall_correct = 0
    recall_tokens = 0
    exact_correct = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = ablate_source_inputs(x[start : start + batch_size], mem_len, source_ablate_suffix).to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            pred = logits.argmax(dim=-1)
            total_loss += loss.item() * yb.numel()
            total_tokens += yb.numel()
            total_correct += (pred == yb).sum().item()
            recall_pred = pred[:, -mem_len:]
            recall_target = yb[:, -mem_len:]
            recall_correct += (recall_pred == recall_target).sum().item()
            recall_tokens += recall_target.numel()
            exact_correct += (recall_pred == recall_target).all(dim=1).sum().item()
    return {
        "loss": total_loss / total_tokens,
        "all_acc": total_correct / total_tokens,
        "recall_acc": recall_correct / recall_tokens,
        "exact_acc": exact_correct / x.size(0),
    }


def build_itos(data_path):
    text = normalize_text(Path(data_path).read_text(encoding="utf-8"))
    _, itos = build_repeat_vocab(text)
    return itos


def decode_ids(ids, itos, marker_id):
    chars = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == 0:
            chars.append("_")
        elif token_id == marker_id:
            chars.append("|")
        else:
            chars.append(itos.get(token_id, "?"))
    return "".join(chars)


@torch.no_grad()
def write_non_exact_samples(
    model,
    x,
    y,
    batch_size,
    device,
    mem_len,
    itos,
    marker_id,
    output_path,
    max_samples,
    source_ablate_suffix=0,
):
    if max_samples <= 0:
        return []
    model.eval()
    samples = []
    for start in range(0, x.size(0), batch_size):
        xb_cpu = ablate_source_inputs(x[start : start + batch_size], mem_len, source_ablate_suffix)
        xb = xb_cpu.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu()
        recall_pred = pred[:, -mem_len:]
        recall_target = y[start : start + batch_size, -mem_len:]
        exact = (recall_pred == recall_target).all(dim=1)
        for row in range(xb.size(0)):
            if bool(exact[row]):
                continue
            sample_idx = start + row
            source = ablate_source_inputs(x[sample_idx : sample_idx + 1], mem_len, source_ablate_suffix)[0, :mem_len]
            target = recall_target[row]
            predicted = recall_pred[row]
            mismatches = (predicted != target).nonzero(as_tuple=False).flatten().tolist()
            samples.append(
                {
                    "index": sample_idx,
                    "mismatches": mismatches,
                    "source": decode_ids(source.tolist(), itos, marker_id),
                    "target": decode_ids(target.tolist(), itos, marker_id),
                    "predicted": decode_ids(predicted.tolist(), itos, marker_id),
                }
            )
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"index={sample['index']} mismatches={sample['mismatches']}\n")
            handle.write(f"source:    {sample['source']}\n")
            handle.write(f"target:    {sample['target']}\n")
            handle.write(f"predicted: {sample['predicted']}\n\n")
    return samples


def parse_layer_map(spec):
    mapping = {}
    if not spec:
        return mapping
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        dst, src = item.split(":")
        mapping[int(dst) - 1] = int(src) - 1
    return mapping


def remap_checkpoint_state(source_state, target_state, layer_map):
    remapped = {}
    used_target_keys = set()
    for target_key in target_state:
        if target_key.startswith("layers."):
            parts = target_key.split(".", 2)
            target_idx = int(parts[1])
            suffix = parts[2]
            if target_idx not in layer_map:
                continue
            source_key = f"layers.{layer_map[target_idx]}.{suffix}"
        else:
            source_key = target_key
        if source_key in source_state and source_state[source_key].shape == target_state[target_key].shape:
            remapped[target_key] = source_state[source_key]
            used_target_keys.add(target_key)
    return remapped, used_target_keys


def apply_freezing(model, train_layers_spec):
    if not train_layers_spec:
        return "all"
    train_layers = parse_layer_indices(train_layers_spec, len(model.layers))
    for param in model.parameters():
        param.requires_grad = False
    for idx in train_layers:
        for param in model.layers[idx].parameters():
            param.requires_grad = True
    for module in [model.norm, model.head, model.rate_transition]:
        for param in module.parameters():
            param.requires_grad = True
    return ",".join(str(idx + 1) for idx in sorted(train_layers))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_x, train_y, test_x, test_y, n_classes, n_chars, marker_id = generate_repeat_data(
        args.data_path,
        args.mem_len,
        args.blank_len,
        args.n_train,
        args.n_test,
        args.split,
        args.seed,
        args.span_mode,
        args.chunk_span_len,
    )
    itos = build_itos(args.data_path)
    multirate_schedule = parse_schedule(args.multirate_schedule)
    if multirate_schedule is not None and len(multirate_schedule) != args.n_layers:
        raise ValueError("multirate_schedule length must match n_layers")
    model = HymbaLM(
        vocab_size=n_classes,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout=args.dropout,
        seq_len=train_x.size(1),
        branch_scale_init=args.branch_scale_init,
        n_meta_tokens=args.n_meta_tokens,
        window_size=args.window_size,
        full_attention_layers=parse_layer_indices(args.full_attention_layers, args.n_layers),
        attention_context_cap=args.attention_context_cap,
        multirate_schedule=multirate_schedule,
    ).to(device)
    params = count_params(model)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)
    os.makedirs(args.output_dir, exist_ok=True)

    seq_total = train_x.size(1)
    dense_tokens = seq_total * args.n_layers
    if multirate_schedule:
        active_tokens = sum((seq_total + args.n_meta_tokens + rate - 1) // rate for rate in multirate_schedule)
    else:
        active_tokens = dense_tokens

    print(f"device={device}", flush=True)
    print(f"params={params}", flush=True)
    print(
        f"config=hymba_text_repeat n_chars:{n_chars} n_classes:{n_classes} marker_id:{marker_id} "
        f"blank_len:{args.blank_len} mem_len:{args.mem_len} seq_total:{seq_total} "
        f"span_mode:{args.span_mode} chunk_span_len:{args.chunk_span_len if args.chunk_span_len > 0 else max(args.mem_len, 1024)} "
        f"layers:{args.n_layers} embed_dim:{args.embed_dim} d_state:{args.d_state} "
        f"heads:{args.n_heads} kv_heads:{args.n_kv_heads} meta_tokens:{args.n_meta_tokens} "
        f"window_size:{args.window_size} attention_context_cap:{args.attention_context_cap} "
        f"full_attention_layers:{args.full_attention_layers} "
        f"multirate:{multirate_schedule} active_tokens:{active_tokens} dense_tokens:{dense_tokens} "
        f"active_ratio:{active_tokens / dense_tokens:.3f} self_mix_interval:{args.self_mix_interval} "
        f"self_mix_insert_prob:{args.self_mix_insert_prob} weight_decay:{args.weight_decay} "
        f"label_smoothing:{args.label_smoothing} train_layers:{args.train_layers or 'all'} "
        f"repeat_noise_prob:{args.repeat_noise_prob} repeat_noise_step_per_epoch:{args.repeat_noise_step_per_epoch} "
        f"repeat_noise_max:{args.repeat_noise_max} repeat_ablate_suffix:{args.repeat_ablate_suffix} "
        f"repeat_ablate_suffix_step_per_epoch:{args.repeat_ablate_suffix_step_per_epoch} "
        f"repeat_ablate_suffix_max:{args.repeat_ablate_suffix_max} "
        f"source_ablate_suffix:{args.source_ablate_suffix} "
        f"source_ablate_suffix_step_per_epoch:{args.source_ablate_suffix_step_per_epoch} "
        f"source_ablate_suffix_max:{args.source_ablate_suffix_max} "
        f"wrong_guess_mix_prob:{args.wrong_guess_mix_prob} "
        f"early_stop_exact_below:{args.early_stop_exact_below} "
        f"init_from:{args.init_from or 'none'}",
        flush=True,
    )

    best_recall = 0.0
    best_exact = 0.0
    start_epoch = 1
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device)
        layer_map = parse_layer_map(args.init_layer_map)
        if not layer_map:
            raise ValueError("init_from requires init_layer_map for cross-depth initialization")
        loaded_state, used_keys = remap_checkpoint_state(checkpoint["model"], model.state_dict(), layer_map)
        missing, unexpected = model.load_state_dict(loaded_state, strict=False)
        print(
            f"initialized_from={args.init_from} mapped_keys={len(used_keys)} "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)} layer_map:{args.init_layer_map}",
            flush=True,
        )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_recall = checkpoint.get("best_recall", best_recall)
        best_exact = checkpoint.get("best_exact", best_exact)
        start_epoch = checkpoint.get("epoch", 0) + 1 if args.resume_epoch else 1
        print(
            f"resumed_from={args.resume} checkpoint_epoch={checkpoint.get('epoch', 'unknown')} "
            f"start_epoch={start_epoch} best_recall={best_recall:.2%} best_exact={best_exact:.2%}",
            flush=True,
        )

    train_layers_desc = apply_freezing(model, args.train_layers)
    trainable_params = count_params(model)
    print(f"train_layers_resolved={train_layers_desc} trainable_params={trainable_params}", flush=True)
    optimizer = getattr(optim, args.optim)(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        current_ablate_suffix = epoch_ablate_suffix(args, epoch, start_epoch)
        current_source_ablate_suffix = epoch_source_ablate_suffix(args, epoch, start_epoch)
        current_noise_prob = epoch_noise_prob(args, epoch, start_epoch)
        model.train()
        order = torch.randperm(train_x.size(0))
        pred_train = None
        if args.self_mix_interval > 0:
            mix_start = args.self_mix_start if args.self_mix_start >= 0 else 0
            mix_end = args.self_mix_end if args.self_mix_end > 0 else train_x.size(1)
            pred_train = predict_repeat_tokens(
                model,
                train_x,
                args.eval_batch_size,
                args.self_mix_temperature,
                args.self_mix_top_k,
                device,
            )
            print(
                f"self_mix_refresh epoch={epoch} interval={args.self_mix_interval} "
                f"span={args.self_mix_span_min}-{args.self_mix_span_max} "
                f"insert_prob={args.self_mix_insert_prob} region={mix_start}:{mix_end}",
                flush=True,
            )
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_recall_correct = 0
        epoch_recall_tokens = 0
        epoch_wrong_guess_replaced = 0
        epoch_wrong_guess_candidates = 0
        epoch_started = time.time()
        for batch_idx, start in enumerate(range(0, train_x.size(0), args.batch_size), start=1):
            idx = order[start : start + args.batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            xb = ablate_source_inputs(xb, args.mem_len, current_source_ablate_suffix)
            xb, yb = corrupt_repeat_inputs(
                xb,
                yb,
                args.mem_len,
                n_chars,
                current_noise_prob,
                current_ablate_suffix,
                args.seed + epoch * 100000 + batch_idx,
            )
            if pred_train is not None:
                xb, yb = build_self_mixed_repeat_batch(
                    xb,
                    yb,
                    pred_train[idx],
                    args.self_mix_interval,
                    args.self_mix_span_min,
                    args.self_mix_span_max,
                    args.self_mix_insert_prob,
                    args.seed + epoch * 100000 + batch_idx,
                    n_chars,
                    mix_start,
                    mix_end,
                )
            xb = xb.to(device)
            yb = yb.to(device)
            if args.wrong_guess_mix_prob > 0:
                xb, yb, replaced, candidates = apply_wrong_guess_mix(
                    model,
                    xb,
                    yb,
                    args.mem_len,
                    args.wrong_guess_mix_prob,
                    args.seed + epoch * 100000 + batch_idx + 17,
                )
                epoch_wrong_guess_replaced += replaced
                epoch_wrong_guess_candidates += candidates
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                recall_pred = pred[:, -args.mem_len:]
                recall_target = yb[:, -args.mem_len:]
                valid = recall_target != -100
                epoch_recall_correct += ((recall_pred == recall_target) & valid).sum().item()
                epoch_recall_tokens += valid.sum().item()
            valid_tokens = (yb != -100).sum().item()
            epoch_loss += loss.item() * valid_tokens
            epoch_tokens += valid_tokens
            if batch_idx % args.log_interval == 0:
                train_recall = epoch_recall_correct / max(1, epoch_recall_tokens)
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{(train_x.size(0) + args.batch_size - 1) // args.batch_size} "
                    f"loss={epoch_loss / max(1, epoch_tokens):.6f} recall_acc={train_recall:.2%}",
                    flush=True,
                )
        metrics = repeat_evaluate(
            model,
            test_x,
            test_y,
            criterion,
            args.eval_batch_size,
            device,
            args.mem_len,
            current_source_ablate_suffix,
        )
        is_best_recall = metrics["recall_acc"] >= best_recall
        is_best_exact = metrics["exact_acc"] >= best_exact
        best_recall = max(best_recall, metrics["recall_acc"])
        best_exact = max(best_exact, metrics["exact_acc"])
        print(
            f"epoch={epoch} train_loss={epoch_loss / max(1, epoch_tokens):.6f} "
            f"train_recall_acc={epoch_recall_correct / max(1, epoch_recall_tokens):.2%} "
            f"test_loss={metrics['loss']:.6f} all_acc={metrics['all_acc']:.2%} "
            f"recall_acc={metrics['recall_acc']:.2%} exact_acc={metrics['exact_acc']:.2%} "
            f"repeat_noise_prob={current_noise_prob:.4f} repeat_ablate_suffix={current_ablate_suffix} "
            f"source_ablate_suffix={current_source_ablate_suffix} "
            f"wrong_guess_replaced={epoch_wrong_guess_replaced}/{epoch_wrong_guess_candidates} "
            f"best_recall={best_recall:.2%} best_exact={best_exact:.2%} "
            f"epoch_seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "params": params,
            "epoch": epoch,
            "metrics": metrics,
            "best_recall": best_recall,
            "best_exact": best_exact,
            "repeat_noise_prob": current_noise_prob,
            "repeat_ablate_suffix": current_ablate_suffix,
            "source_ablate_suffix": current_source_ablate_suffix,
            "wrong_guess_mix_prob": args.wrong_guess_mix_prob,
            "wrong_guess_replaced": epoch_wrong_guess_replaced,
            "wrong_guess_candidates": epoch_wrong_guess_candidates,
            "active_tokens": active_tokens,
            "dense_tokens": dense_tokens,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pt"))
        if is_best_recall:
            torch.save(checkpoint, os.path.join(args.output_dir, "best_recall_checkpoint.pt"))
        if is_best_exact:
            torch.save(checkpoint, os.path.join(args.output_dir, "best_exact_checkpoint.pt"))
        if args.sample_non_exact > 0:
            sample_path = os.path.join(args.output_dir, f"non_exact_samples_epoch_{epoch:03d}.txt")
            samples = write_non_exact_samples(
                model,
                test_x,
                test_y,
                args.eval_batch_size,
                device,
                args.mem_len,
                itos,
                marker_id,
                sample_path,
                args.sample_non_exact,
                current_source_ablate_suffix,
            )
            if args.print_non_exact > 0:
                print(
                    f"non_exact_samples epoch={epoch} path={sample_path} showing={min(args.print_non_exact, len(samples))}",
                    flush=True,
                )
                for sample in samples[: args.print_non_exact]:
                    print(f"sample index={sample['index']} mismatches={sample['mismatches']}", flush=True)
                    print(f"  target:    {sample['target']}", flush=True)
                    print(f"  predicted: {sample['predicted']}", flush=True)
        if args.early_stop_exact_below >= 0 and metrics["exact_acc"] < args.early_stop_exact_below:
            print(
                f"early_stop exact_acc={metrics['exact_acc']:.2%} "
                f"below_threshold={args.early_stop_exact_below:.2%}",
                flush=True,
            )
            break
    print(f"duration_seconds={time.time() - started:.1f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./runs_hymba_text_repeat")
    parser.add_argument("--data_path", default="./data/tiny_shakespeare.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--span_mode", choices=["sentence", "chunk"], default="sentence")
    parser.add_argument("--chunk_span_len", type=int, default=0)
    parser.add_argument("--blank_len", type=int, default=128)
    parser.add_argument("--mem_len", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--n_meta_tokens", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--attention_context_cap", type=int, default=0)
    parser.add_argument("--full_attention_layers", default="1,4,8")
    parser.add_argument("--multirate_schedule", default="")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--branch_scale_init", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--optim", default="RMSprop")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_epoch", action="store_true")
    parser.add_argument("--init_from", default="")
    parser.add_argument("--init_layer_map", default="")
    parser.add_argument("--train_layers", default="")
    parser.add_argument("--repeat_noise_prob", type=float, default=0.0)
    parser.add_argument("--repeat_noise_step_per_epoch", type=float, default=0.0)
    parser.add_argument("--repeat_noise_max", type=float, default=-1.0)
    parser.add_argument("--repeat_ablate_suffix", type=int, default=0)
    parser.add_argument("--repeat_ablate_suffix_step_per_epoch", type=int, default=0)
    parser.add_argument("--repeat_ablate_suffix_max", type=int, default=-1)
    parser.add_argument("--source_ablate_suffix", type=int, default=0)
    parser.add_argument("--source_ablate_suffix_step_per_epoch", type=int, default=0)
    parser.add_argument("--source_ablate_suffix_max", type=int, default=-1)
    parser.add_argument("--wrong_guess_mix_prob", type=float, default=0.0)
    parser.add_argument("--early_stop_exact_below", type=float, default=-1.0)
    parser.add_argument("--sample_non_exact", type=int, default=0)
    parser.add_argument("--print_non_exact", type=int, default=0)
    parser.add_argument("--self_mix_interval", type=int, default=0)
    parser.add_argument("--self_mix_span_min", type=int, default=1)
    parser.add_argument("--self_mix_span_max", type=int, default=2)
    parser.add_argument("--self_mix_insert_prob", type=float, default=0.5)
    parser.add_argument("--self_mix_temperature", type=float, default=0.8)
    parser.add_argument("--self_mix_top_k", type=int, default=1)
    parser.add_argument("--self_mix_start", type=int, default=-1)
    parser.add_argument("--self_mix_end", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
