from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import ModelConfig, build_model
from tokenizer import PositionalCharTokenizer


def load_tokenizer(vocab_path: Path) -> PositionalCharTokenizer:
    tokenizer = PositionalCharTokenizer()
    tokenizer.load_vocab(vocab_path)
    return tokenizer


def build_training_stream(
    tokenizer: PositionalCharTokenizer,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stream = tokenizer.text_to_stream(text)
    token_ids = torch.tensor([tokenizer.token_to_id(item.token) for item in stream], dtype=torch.long)

    newline_positions = []
    char_positions_in_line = []
    newline_index = 0
    char_pos_in_line = 0
    for item in stream:
        newline_positions.append(newline_index)
        char_positions_in_line.append(char_pos_in_line)
        if item.token == "\n":
            newline_index += 1
            char_pos_in_line = 0
        else:
            char_pos_in_line += 1

    return (
        token_ids,
        torch.tensor(newline_positions, dtype=torch.long),
        torch.tensor(char_positions_in_line, dtype=torch.long),
    )


def get_batch(
    token_ids: torch.Tensor,
    newline_positions: torch.Tensor,
    char_positions_in_line: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = token_ids.size(0) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([token_ids[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([token_ids[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    newlines = torch.stack([newline_positions[s : s + seq_len] for s in starts]).to(device)
    char_positions = torch.stack([char_positions_in_line[s : s + seq_len] for s in starts]).to(device)
    seq_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    return x, y, seq_pos, newlines, char_positions


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    newline_positions: torch.Tensor,
    char_positions_in_line: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_steps: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_steps):
        x, y, seq_pos, newlines, char_positions = get_batch(
            token_ids, newline_positions, char_positions_in_line, batch_size, seq_len, device
        )
        _, loss = model(x, seq_pos, newlines, char_positions, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact Mistral-style baseline on positional char tokens.")
    parser.add_argument("--input", type=Path, default=Path("data/tiny_shakespeare.txt"))
    parser.add_argument("--vocab", type=Path, default=Path("data/positional_char_vocab.json"))
    parser.add_argument("--embedding-init", type=Path, default=Path("data/embedding_init.npy"))
    parser.add_argument("--out", type=Path, default=Path("data/charformer_checkpoint.pt"))
    parser.add_argument("--architecture", type=str, choices=["legacy", "multirate"], default="multirate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-pre-layers", type=int, default=2)
    parser.add_argument("--n-bottleneck-layers", type=int, default=2)
    parser.add_argument("--n-post-layers", type=int, default=2)
    parser.add_argument("--fine-down-cycles", type=int, default=4)
    parser.add_argument("--mid-down-cycles", type=int, default=2)
    parser.add_argument("--core-cycles", type=int, default=1)
    parser.add_argument("--mid-up-cycles", type=int, default=2)
    parser.add_argument("--fine-up-cycles", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=768)
    parser.add_argument("--compression-factor", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_tokenizer(args.vocab)
    text = args.input.read_text(encoding="utf-8")
    token_ids, newline_positions, char_positions_in_line = build_training_stream(tokenizer, text)
    if args.compression_factor > 1 and args.seq_len % args.compression_factor != 0:
        raise ValueError("--seq-len must be divisible by --compression-factor")

    split = int(token_ids.size(0) * args.train_split)
    train_ids, val_ids = token_ids[:split], token_ids[split:]
    train_newlines, val_newlines = newline_positions[:split], newline_positions[split:]
    train_char_positions, val_char_positions = (
        char_positions_in_line[:split],
        char_positions_in_line[split:],
    )

    embedding_init = torch.from_numpy(np.load(args.embedding_init)).to(torch.float32)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size(),
        architecture=args.architecture,
        d_model=args.d_model,
        n_pre_layers=args.n_pre_layers,
        n_bottleneck_layers=args.n_bottleneck_layers,
        n_post_layers=args.n_post_layers,
        fine_down_cycles=args.fine_down_cycles,
        mid_down_cycles=args.mid_down_cycles,
        core_cycles=args.core_cycles,
        mid_up_cycles=args.mid_up_cycles,
        fine_up_cycles=args.fine_up_cycles,
        n_heads=args.n_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        max_seq_len=args.seq_len,
        compression_factor=args.compression_factor,
        dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = build_model(config, embedding_init=embedding_init).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    print(f"device={device}")
    print(f"tokens={token_ids.size(0)} train_tokens={train_ids.size(0)} val_tokens={val_ids.size(0)}")
    if config.architecture == "legacy":
        print(
            f"arch=legacy vocab_size={config.vocab_size} d_model={config.d_model} "
            f"pre={config.n_pre_layers} bottleneck={config.n_bottleneck_layers} "
            f"post={config.n_post_layers} heads={config.n_heads} factor={config.compression_factor}"
        )
    else:
        print(
            f"arch=multirate vocab_size={config.vocab_size} d_model={config.d_model} "
            f"fine_down={config.fine_down_cycles} mid_down={config.mid_down_cycles} "
            f"core={config.core_cycles} mid_up={config.mid_up_cycles} fine_up={config.fine_up_cycles} "
            f"heads={config.n_heads}"
        )
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()
    for step in range(1, args.steps + 1):
        x, y, seq_pos, newlines, char_positions = get_batch(
            train_ids, train_newlines, train_char_positions, args.batch_size, args.seq_len, device
        )
        _, loss = model(x, seq_pos, newlines, char_positions, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = float(loss.item())
            val_loss = estimate_loss(
                model,
                val_ids,
                val_newlines,
                val_char_positions,
                args.batch_size,
                args.seq_len,
                args.eval_steps,
                device,
            )
            elapsed = time.time() - start_time
            toks_per_s = int(step * args.batch_size * args.seq_len / max(elapsed, 1e-6))
            print(
                f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"ppl={math.exp(min(val_loss, 20)):.2f} toks_per_s={toks_per_s}"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "tokenizer_vocab_path": str(args.vocab),
    }
    torch.save(checkpoint, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
