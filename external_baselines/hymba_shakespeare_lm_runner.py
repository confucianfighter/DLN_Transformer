import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


MAMBA_REPO = Path(__file__).resolve().parent / "mamba.py"
sys.path.insert(0, str(MAMBA_REPO))

from mambapy.mamba import MambaBlock, MambaConfig, RMSNorm  # noqa: E402


def parse_schedule(schedule):
    if not schedule:
        return None
    return [int(item.strip()) for item in schedule.split(",") if item.strip()]


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, positions, base=10000.0):
    dim = x.size(-1)
    if dim % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class CausalRoPEAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, dropout, window_size=0, n_meta_tokens=0, attention_context_cap=0):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, 2 * n_kv_heads * self.head_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.window_size = window_size
        self.n_meta_tokens = n_meta_tokens
        self.attention_context_cap = attention_context_cap

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        q = self.q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).view(bsz, seq_len, 2, self.n_kv_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        positions = torch.arange(seq_len, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        effective_window = 0
        if self.window_size > 0:
            effective_window = self.window_size
        if self.attention_context_cap > 0:
            effective_window = self.attention_context_cap if effective_window == 0 else min(effective_window, self.attention_context_cap)
        if effective_window <= 0 or effective_window >= seq_len:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            chunks = []
            query_block = effective_window
            meta_end = min(self.n_meta_tokens, seq_len)
            meta_k = k[:, :, :meta_end, :]
            meta_v = v[:, :, :meta_end, :]
            meta_positions = torch.arange(meta_end, device=x.device)
            for query_start in range(0, seq_len, query_block):
                query_end = min(query_start + query_block, seq_len)
                local_start = max(meta_end, query_start - effective_window + 1)
                q_chunk = q[:, :, query_start:query_end, :]
                local_k = k[:, :, local_start:query_end, :]
                local_v = v[:, :, local_start:query_end, :]
                if meta_end > 0 and local_start < query_end:
                    k_chunk = torch.cat((meta_k, local_k), dim=2)
                    v_chunk = torch.cat((meta_v, local_v), dim=2)
                    k_positions = torch.cat((meta_positions, torch.arange(local_start, query_end, device=x.device)))
                elif meta_end > 0:
                    k_chunk = meta_k
                    v_chunk = meta_v
                    k_positions = meta_positions
                else:
                    k_chunk = local_k
                    v_chunk = local_v
                    k_positions = torch.arange(local_start, query_end, device=x.device)
                q_positions = torch.arange(query_start, query_end, device=x.device)
                causal = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                in_window = k_positions.unsqueeze(0) >= (q_positions.unsqueeze(1) - effective_window + 1)
                meta = k_positions.unsqueeze(0) < meta_end
                allowed = causal & (in_window | meta)
                attn_mask = torch.zeros((q_chunk.size(2), k_chunk.size(2)), dtype=q.dtype, device=x.device)
                attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
                chunk_out = F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                )
                chunks.append(chunk_out)
            y = torch.cat(chunks, dim=2)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(y)


class HymbaBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        d_state,
        n_heads,
        n_kv_heads,
        dropout,
        branch_scale_init,
        window_size,
        n_meta_tokens,
        attention_context_cap,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(embed_dim)
        self.attn_branch_norm = nn.LayerNorm(embed_dim)
        self.ssm_branch_norm = RMSNorm(embed_dim)
        self.ffn_norm = RMSNorm(embed_dim)
        self.mamba = MambaBlock(
            MambaConfig(
                d_model=embed_dim,
                n_layers=1,
                d_state=d_state,
                d_conv=4,
                expand_factor=2,
                use_cuda=False,
            )
        )
        self.attn = CausalRoPEAttention(
            embed_dim,
            n_heads,
            n_kv_heads,
            dropout,
            window_size=window_size,
            n_meta_tokens=n_meta_tokens,
            attention_context_cap=attention_context_cap,
        )
        self.fuse_out = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.ssm_scale = nn.Parameter(torch.full((embed_dim,), float(branch_scale_init)))
        self.attn_scale = nn.Parameter(torch.full((embed_dim,), float(branch_scale_init)))
        self.ffn_scale = nn.Parameter(torch.full((embed_dim,), float(branch_scale_init)))

    def forward(self, x):
        h = self.pre_norm(x)
        ssm_out = self.ssm_branch_norm(self.mamba(h)) * self.ssm_scale
        attn_out = self.attn_branch_norm(self.attn(h)) * self.attn_scale
        fused = self.fuse_out(0.5 * (ssm_out + attn_out))
        x = x + self.dropout(fused)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)) * self.ffn_scale)
        return x


class AdditiveRateTransition(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)

    def downsample2(self, x, skipped_stack):
        if x.size(1) % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        kept = x[:, 0::2]
        skipped = x[:, 1::2]
        skipped_stack.append(skipped)
        return self.norm(kept + skipped)

    def upsample2(self, x, skipped_stack):
        if not skipped_stack:
            raise ValueError("Cannot upsample without stored skipped tokens")
        skipped = skipped_stack.pop()
        if x.shape != skipped.shape:
            raise ValueError(f"Upsample shape mismatch: active={x.shape} skipped={skipped.shape}")
        out = x.new_empty(x.size(0), x.size(1) * 2, x.size(2))
        out[:, 0::2] = x
        out[:, 1::2] = x + skipped
        return self.norm(out)

    def forward(self, x, previous_rate, current_rate, skipped_stack):
        if current_rate == previous_rate:
            return x
        if current_rate == previous_rate * 2:
            return self.downsample2(x, skipped_stack)
        if previous_rate == current_rate * 2:
            return self.upsample2(x, skipped_stack)
        raise ValueError("Multirate Hymba transitions require adjacent 2x rate changes")


class HymbaLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        n_layers,
        d_state,
        n_heads,
        n_kv_heads,
        dropout,
        seq_len,
        branch_scale_init,
        n_meta_tokens,
        window_size,
        full_attention_layers,
        attention_context_cap=0,
        multirate_schedule=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_meta_tokens = n_meta_tokens
        self.attention_context_cap = attention_context_cap
        self.multirate_schedule = multirate_schedule
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.meta_tokens = nn.Parameter(torch.zeros(n_meta_tokens, embed_dim))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                HymbaBlock(
                    embed_dim,
                    d_state,
                    n_heads,
                    n_kv_heads,
                    dropout,
                    branch_scale_init,
                    window_size=0 if idx in full_attention_layers else window_size,
                    n_meta_tokens=n_meta_tokens,
                    attention_context_cap=attention_context_cap,
                )
                for idx in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.head.weight = self.embed.weight
        self.rate_transition = AdditiveRateTransition(embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.meta_tokens, mean=0.0, std=0.02)

    def forward(self, x):
        original_len = x.size(1)
        x = self.drop(self.embed(x))
        if self.n_meta_tokens:
            meta = self.meta_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
            x = torch.cat([meta, x], dim=1)
        if self.multirate_schedule is None:
            for layer in self.layers:
                x = layer(x)
        else:
            skipped_stack = []
            previous_rate = 1
            for rate, layer in zip(self.multirate_schedule, self.layers):
                x = self.rate_transition(x, previous_rate, rate, skipped_stack)
                x = layer(x)
                previous_rate = rate
            if skipped_stack:
                raise ValueError("Multirate schedule ended with unresolved skipped tokens")
        if self.n_meta_tokens:
            x = x[:, self.n_meta_tokens :]
        if x.size(1) > original_len:
            x = x[:, :original_len]
        elif x.size(1) < original_len:
            x = F.pad(x, (0, 0, 0, original_len - x.size(1)))
        return self.head(self.norm(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch(input_ids, target_ids, batch_size, seq_len, device):
    starts = torch.randint(0, input_ids.size(0) - seq_len, (batch_size,))
    offsets = torch.arange(seq_len)
    x = torch.stack([input_ids[start + offsets] for start in starts]).to(device)
    y = torch.stack([target_ids[start + offsets] for start in starts]).to(device)
    return x, y


@torch.no_grad()
def predict_stream(model, token_ids, seq_len, batch_size, temperature, top_k, device):
    model.eval()
    pred = token_ids.clone()
    starts = list(range(0, token_ids.size(0) - seq_len - 1, seq_len))
    offsets = torch.arange(seq_len)
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        x = torch.stack([token_ids[start + offsets] for start in batch_starts]).to(device)
        logits = model(x) / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(2, indices, values)
            logits = filtered
        sampled = torch.multinomial(torch.softmax(logits.reshape(-1, logits.size(-1)), dim=-1), 1)
        sampled = sampled.reshape(logits.size(0), logits.size(1)).cpu()
        for row, start in enumerate(batch_starts):
            pred[start + 1 : start + seq_len + 1] = sampled[row]
    return pred


def build_self_mixed_stream(token_ids, pred_ids, interval, span_min, span_max, insert_prob, seed):
    rng = random.Random(seed)
    input_rows = []
    target_rows = []
    i = 0
    next_corrupt = rng.randint(max(1, interval // 2), max(1, interval + interval // 2))
    while i < token_ids.size(0) - 1:
        if i >= next_corrupt:
            span = rng.randint(span_min, span_max)
            span = min(span, token_ids.size(0) - 1 - i)
            if rng.random() < insert_prob:
                for j in range(span):
                    input_rows.append(int(pred_ids[i + j]))
                    target_rows.append(-100)
            else:
                for j in range(span):
                    input_rows.append(int(pred_ids[i + j]))
                    target_rows.append(-100)
                i += span
            next_corrupt += rng.randint(max(1, interval // 2), max(1, interval + interval // 2))
            continue
        input_rows.append(int(token_ids[i]))
        target_rows.append(int(token_ids[i + 1]))
        i += 1
    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(target_rows, dtype=torch.long)


@torch.no_grad()
def evaluate(model, input_ids, target_ids, criterion, batch_size, seq_len, eval_batches, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for _ in range(eval_batches):
        x, y = get_batch(input_ids, target_ids, batch_size, seq_len, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
    loss = total_loss / total_tokens
    return {"loss": loss, "ppl": math.exp(min(loss, 20.0)), "acc": total_correct / total_tokens}


@torch.no_grad()
def generate(model, prompt, stoi, itos, n_chars, temperature, top_k, device):
    model.eval()
    ids = [stoi[ch] for ch in prompt if ch in stoi] or [0]
    pad_id = stoi.get(" ", ids[0])
    context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n_chars):
        if context.size(1) < model.seq_len:
            pad = torch.full((1, model.seq_len - context.size(1)), pad_id, dtype=torch.long, device=device)
            x = torch.cat([pad, context], dim=1)
        else:
            x = context[:, -model.seq_len :]
        logits = model(x)[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        context = torch.cat([context, next_id], dim=1)
    return "".join(itos[int(idx)] for idx in context[0].tolist())


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    text = load_text(args.input)
    stoi, itos = build_vocab(text)
    token_ids = encode(text, stoi)
    split_at = int(token_ids.size(0) * args.split)
    train_ids = token_ids[:split_at]
    val_ids = token_ids[split_at:]
    train_input_ids = train_ids[:-1]
    train_target_ids = train_ids[1:]
    val_input_ids = val_ids[:-1]
    val_target_ids = val_ids[1:]
    multirate_schedule = parse_schedule(args.multirate_schedule)
    if multirate_schedule is not None and len(multirate_schedule) != args.n_layers:
        raise ValueError("multirate_schedule length must match n_layers")

    model = HymbaLM(
        vocab_size=len(stoi),
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        d_state=args.d_state,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        branch_scale_init=args.branch_scale_init,
        n_meta_tokens=args.n_meta_tokens,
        window_size=args.window_size,
        full_attention_layers=parse_layer_indices(args.full_attention_layers, args.n_layers),
        attention_context_cap=args.attention_context_cap,
        multirate_schedule=multirate_schedule,
    ).to(device)
    params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"device={device}", flush=True)
    print(f"params={params}", flush=True)
    print(
        f"config=hymba_shakespeare_lm vocab:{len(stoi)} seq_len:{args.seq_len} "
        f"layers:{args.n_layers} embed_dim:{args.embed_dim} d_state:{args.d_state} "
        f"heads:{args.n_heads} kv_heads:{args.n_kv_heads} meta_tokens:{args.n_meta_tokens} "
        f"window_size:{args.window_size} attention_context_cap:{args.attention_context_cap} "
        f"full_attention_layers:{args.full_attention_layers} "
        f"multirate:{multirate_schedule} dropout:{args.dropout}",
        flush=True,
    )

    best_val = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        if args.self_mix_interval > 0 and (step == 1 or (step - 1) % args.self_mix_refresh == 0):
            pred_ids = predict_stream(
                model,
                train_ids,
                args.seq_len,
                args.eval_batch_size,
                args.self_mix_temperature,
                args.self_mix_top_k,
                device,
            )
            train_input_ids, train_target_ids = build_self_mixed_stream(
                train_ids,
                pred_ids,
                args.self_mix_interval,
                args.self_mix_span_min,
                args.self_mix_span_max,
                args.self_mix_insert_prob,
                args.seed + step,
            )
            print(
                f"self_mix_refresh step={step} mixed_tokens={train_input_ids.size(0)} "
                f"real_tokens={train_ids.size(0) - 1} interval={args.self_mix_interval} "
                f"insert_prob={args.self_mix_insert_prob}",
                flush=True,
            )
        model.train()
        x, y = get_batch(train_input_ids, train_target_ids, args.batch_size, args.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if step % args.eval_interval == 0 or step == 1:
            train_metrics = evaluate(
                model,
                train_input_ids,
                train_target_ids,
                criterion,
                args.eval_batch_size,
                args.seq_len,
                args.eval_batches,
                device,
            )
            val_metrics = evaluate(
                model,
                val_input_ids,
                val_target_ids,
                criterion,
                args.eval_batch_size,
                args.seq_len,
                args.eval_batches,
                device,
            )
            best_val = min(best_val, val_metrics["loss"])
            print(
                f"step={step}/{args.steps} loss={loss.item():.4f} "
                f"train_loss={train_metrics['loss']:.4f} train_ppl={train_metrics['ppl']:.2f} train_acc={train_metrics['acc']:.2%} "
                f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['ppl']:.2f} val_acc={val_metrics['acc']:.2%} "
                f"best_val={best_val:.4f} elapsed={time.time() - started:.1f}",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "stoi": stoi,
                    "itos": itos,
                    "params": params,
                    "step": step,
                    "best_val": best_val,
                },
                os.path.join(args.output_dir, "checkpoint.pt"),
            )

    print("samples:", flush=True)
    for prompt in args.prompts.split("|"):
        sample = generate(model, prompt, stoi, itos, args.sample_chars, args.temperature, args.top_k, device)
        print(f"--- prompt: {prompt!r} ---", flush=True)
        print(sample, flush=True)
    print(f"duration_seconds={time.time() - started:.1f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./runs_hymba_shakespeare_lm")
    parser.add_argument("--input", default="./data/tiny_shakespeare.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seq_len", type=int, default=256)
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
    parser.add_argument("--optim", default="RMSprop")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--sample_chars", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--prompts", default="ROMEO:|First Citizen:|To be")
    parser.add_argument("--self_mix_interval", type=int, default=0)
    parser.add_argument("--self_mix_refresh", type=int, default=250)
    parser.add_argument("--self_mix_span_min", type=int, default=1)
    parser.add_argument("--self_mix_span_max", type=int, default=2)
    parser.add_argument("--self_mix_insert_prob", type=float, default=0.5)
    parser.add_argument("--self_mix_temperature", type=float, default=0.8)
    parser.add_argument("--self_mix_top_k", type=int, default=10)
    return parser.parse_args()


def parse_layer_indices(spec, n_layers):
    result = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = n_layers + idx
        else:
            idx = idx - 1
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"full_attention layer index out of range: {item}")
        result.add(idx)
    return result


if __name__ == "__main__":
    train(parse_args())
