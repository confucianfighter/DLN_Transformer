from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model import ModelConfig, build_model
from tokenizer import PositionalCharTokenizer


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, PositionalCharTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    tokenizer = PositionalCharTokenizer()
    tokenizer.load_vocab(checkpoint["tokenizer_vocab_path"])

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def build_positions(tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_positions = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)

    newline_positions = []
    char_positions_in_line = []
    newline_count = 0
    char_pos = 0
    for token in tokens:
        newline_positions.append(newline_count)
        char_positions_in_line.append(char_pos)
        if token == "\n":
            newline_count += 1
            char_pos = 0
        else:
            char_pos += 1

    newline_positions_t = torch.tensor(newline_positions, dtype=torch.long).unsqueeze(0)
    char_positions_t = torch.tensor(char_positions_in_line, dtype=torch.long).unsqueeze(0)
    return seq_positions, newline_positions_t, char_positions_t


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> int:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: PositionalCharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    tokens = tokenizer.text_to_tokens(prompt)
    if not tokens:
        raise ValueError("Prompt must produce at least one token")

    for _ in range(max_new_tokens):
        ctx_tokens = tokens[-model.config.max_seq_len :]
        if model.config.compression_factor > 1:
            remainder = len(ctx_tokens) % model.config.compression_factor
            if remainder != 0:
                pad_len = model.config.compression_factor - remainder
                ctx_tokens = (["<PAD>"] * pad_len) + ctx_tokens
        input_ids = torch.tensor([[tokenizer.token_to_id(token) for token in ctx_tokens]], device=next(model.parameters()).device)
        seq_positions, newline_positions, char_positions = build_positions(ctx_tokens)
        seq_positions = seq_positions.to(input_ids.device)
        newline_positions = newline_positions.to(input_ids.device)
        char_positions = char_positions.to(input_ids.device)

        logits, _ = model(input_ids, seq_positions, newline_positions, char_positions)
        next_id = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        tokens.append(tokenizer.id_to_token(next_id))

    return tokenizer.render_tokens(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="king richard")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)
    text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        print(f"saved generation to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
