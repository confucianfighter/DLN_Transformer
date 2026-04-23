import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


TCN_REPO = Path(__file__).resolve().parent / "TCN"
sys.path.insert(0, str(TCN_REPO))

from TCN.copy_memory.model import TCN  # noqa: E402
from tcn_copy_multirate_runner import MultirateTCNCopy, active_token_estimate, parse_schedule, parse_skip_pairs  # noqa: E402


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def sentence_spans(text):
    spans = []
    start = 0
    for match in re.finditer(r"[.!?;:]", text):
        end = match.end()
        if end - start > 0:
            spans.append(text[start:end].strip())
        start = end
    if start < len(text):
        spans.append(text[start:].strip())
    return [span for span in spans if span]


def chunk_spans(text, chunk_span_len):
    if chunk_span_len <= 0:
        raise ValueError("chunk_span_len must be positive")
    spans = []
    for start in range(0, len(text), chunk_span_len):
        span = text[start : start + chunk_span_len].strip()
        if span:
            spans.append(span)
    return spans


def build_repeat_spans(text, mem_len, span_mode, chunk_span_len):
    if span_mode == "sentence":
        spans = sentence_spans(text)
        if not any(len(span) >= mem_len for span in spans):
            spans = [text]
        return spans
    if span_mode == "chunk":
        return chunk_spans(text, chunk_span_len)
    raise ValueError(f"Unsupported span_mode: {span_mode}")


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: idx + 1 for idx, ch in enumerate(chars)}
    itos = {idx + 1: ch for idx, ch in enumerate(chars)}
    return stoi, itos


def encode_fragment(fragment, stoi, mem_len):
    encoded = [stoi[ch] for ch in fragment[:mem_len]]
    if len(encoded) < mem_len:
        encoded.extend([0] * (mem_len - len(encoded)))
    return encoded


def make_fragments(spans, stoi, mem_len, n_samples, rng):
    candidates = [span for span in spans if len(span) >= mem_len]
    if not candidates:
        raise ValueError("No candidate spans are long enough for mem_len")
    rows = []
    for _ in range(n_samples):
        span = rng.choice(candidates)
        start = rng.randint(0, len(span) - mem_len)
        rows.append(encode_fragment(span[start : start + mem_len], stoi, mem_len))
    return torch.tensor(rows, dtype=torch.long)


def generate_repeat_data(data_path, mem_len, blank_len, n_train, n_test, split, seed, span_mode="sentence", chunk_span_len=0):
    text = normalize_text(Path(data_path).read_text(encoding="utf-8"))
    split_at = int(len(text) * split)
    train_text = text[:split_at]
    test_text = text[split_at:]
    effective_chunk_span_len = chunk_span_len if chunk_span_len > 0 else max(mem_len, 1024)
    train_spans = build_repeat_spans(train_text, mem_len, span_mode, effective_chunk_span_len)
    test_spans = build_repeat_spans(test_text, mem_len, span_mode, effective_chunk_span_len)
    stoi, itos = build_vocab(text)
    marker_id = len(stoi) + 1
    n_classes = marker_id + 1
    rng = random.Random(seed)

    train_fragments = make_fragments(train_spans, stoi, mem_len, n_train, rng)
    test_fragments = make_fragments(test_spans, stoi, mem_len, n_test, rng)

    def assemble(fragments):
        n = fragments.size(0)
        blanks = torch.zeros((n, blank_len), dtype=torch.long)
        marker = torch.full((n, mem_len + 1), marker_id, dtype=torch.long)
        placeholders = torch.zeros((n, mem_len), dtype=torch.long)
        x = torch.cat((fragments, blanks[:, :-1], marker), dim=1)
        y = torch.cat((placeholders, blanks, fragments), dim=1)
        return x, y

    return (*assemble(train_fragments), *assemble(test_fragments), n_classes, len(stoi), marker_id)


class DenseTextRepeat(nn.Module):
    def __init__(self, n_classes, emb_dim, channels, kernel_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb_dim, padding_idx=0)
        self.tcn = TCN(emb_dim, n_classes, channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2).contiguous()
        return self.tcn(x)


class MultirateTextRepeat(nn.Module):
    def __init__(self, n_classes, emb_dim, channels, schedule, kernel_size, dropout, dilation_mode, skip_pairs, skip_gate_init):
        super().__init__()
        self.embed = nn.Embedding(n_classes, emb_dim, padding_idx=0)
        self.tcn = MultirateTCNCopy(
            emb_dim,
            n_classes,
            channels,
            schedule=schedule,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation_mode=dilation_mode,
            skip_pairs=skip_pairs,
            skip_gate_init=skip_gate_init,
        )

    def forward(self, x):
        x = self.embed(x).transpose(1, 2).contiguous()
        return self.tcn(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, x, y, criterion, batch_size, device, mem_len):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    recall_correct = 0
    recall_tokens = 0
    exact_correct = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start : start + batch_size].to(device)
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
    channel_sizes = [args.nhid] * args.levels
    schedule = parse_schedule(args.multirate_schedule)
    skip_pairs = parse_skip_pairs(args.skip_pairs)
    if args.model == "dense":
        model = DenseTextRepeat(n_classes, args.emb_dim, channel_sizes, args.ksize, args.dropout).to(device)
        active_tokens = train_x.size(1) * args.levels
        dense_tokens = active_tokens
    else:
        if len(schedule) != args.levels:
            raise ValueError("multirate_schedule length must equal levels")
        model = MultirateTextRepeat(
            n_classes,
            args.emb_dim,
            channel_sizes,
            schedule,
            args.ksize,
            args.dropout,
            args.dilation_mode,
            skip_pairs,
            args.skip_gate_init,
        ).to(device)
        active_tokens = active_token_estimate(train_x.size(1), schedule)
        dense_tokens = train_x.size(1) * args.levels

    params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"device={device}", flush=True)
    print(f"params={params}", flush=True)
    print(
        f"config=tcn_text_repeat model:{args.model} data:{args.data_path} n_chars:{n_chars} "
        f"n_classes:{n_classes} marker_id:{marker_id} blank_len:{args.blank_len} mem_len:{args.mem_len} "
        f"span_mode:{args.span_mode} chunk_span_len:{args.chunk_span_len if args.chunk_span_len > 0 else max(args.mem_len, 1024)} "
        f"seq_total:{train_x.size(1)} levels:{args.levels} emb_dim:{args.emb_dim} nhid:{args.nhid} "
        f"ksize:{args.ksize} dropout:{args.dropout} schedule:{schedule if args.model != 'dense' else 'dense'} "
        f"active_tokens:{active_tokens} dense_tokens:{dense_tokens} active_ratio:{active_tokens / dense_tokens:.3f} "
        f"dilation_mode:{args.dilation_mode} skip_pairs:{args.skip_pairs}",
        flush=True,
    )

    best_recall = 0.0
    best_exact = 0.0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_recall = checkpoint.get("best_recall", 0.0)
        best_exact = checkpoint.get("best_exact", 0.0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(
            f"resumed={args.resume} start_epoch={start_epoch} "
            f"best_recall={best_recall:.2%} best_exact={best_exact:.2%}",
            flush=True,
        )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        order = torch.randperm(train_x.size(0))
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_recall_correct = 0
        epoch_recall_tokens = 0
        epoch_started = time.time()
        for batch_idx, start in enumerate(range(0, train_x.size(0), args.batch_size), start=1):
            idx = order[start : start + args.batch_size]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
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
                epoch_recall_correct += (recall_pred == recall_target).sum().item()
                epoch_recall_tokens += recall_target.numel()
            epoch_loss += loss.item() * yb.numel()
            epoch_tokens += yb.numel()
            if batch_idx % args.log_interval == 0:
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{(train_x.size(0) + args.batch_size - 1) // args.batch_size} "
                    f"loss={epoch_loss / epoch_tokens:.6f} recall_acc={epoch_recall_correct / epoch_recall_tokens:.2%}",
                    flush=True,
                )

        metrics = evaluate(model, test_x, test_y, criterion, args.eval_batch_size, device, args.mem_len)
        best_recall = max(best_recall, metrics["recall_acc"])
        best_exact = max(best_exact, metrics["exact_acc"])
        print(
            f"epoch={epoch} train_loss={epoch_loss / epoch_tokens:.6f} "
            f"train_recall_acc={epoch_recall_correct / epoch_recall_tokens:.2%} "
            f"test_loss={metrics['loss']:.6f} all_acc={metrics['all_acc']:.2%} "
            f"recall_acc={metrics['recall_acc']:.2%} exact_acc={metrics['exact_acc']:.2%} "
            f"best_recall={best_recall:.2%} best_exact={best_exact:.2%} "
            f"epoch_seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "params": params,
                "epoch": epoch,
                "best_recall": best_recall,
                "best_exact": best_exact,
                "active_tokens": active_tokens,
                "dense_tokens": dense_tokens,
            },
            os.path.join(args.output_dir, "checkpoint.pt"),
        )
    print(f"duration_seconds={time.time() - started:.1f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./runs_tcn_text_repeat")
    parser.add_argument("--data_path", default="./data/tiny_shakespeare.txt")
    parser.add_argument("--model", choices=["dense", "multirate"], default="dense")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--span_mode", choices=["sentence", "chunk"], default="sentence")
    parser.add_argument("--chunk_span_len", type=int, default=0)
    parser.add_argument("--blank_len", type=int, default=128)
    parser.add_argument("--mem_len", type=int, default=64)
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--nhid", type=int, default=16)
    parser.add_argument("--ksize", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--multirate_schedule", default="1,1,2,4,4,2,1,1")
    parser.add_argument("--dilation_mode", choices=["tcn", "constant"], default="tcn")
    parser.add_argument("--skip_pairs", default="")
    parser.add_argument("--skip_gate_init", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optim", default="RMSprop")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
