from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    architecture: str = "legacy"
    d_model: int = 256
    n_layers: int = 6
    n_pre_layers: int = 2
    n_bottleneck_layers: int = 2
    n_post_layers: int = 2
    fine_down_cycles: int = 4
    mid_down_cycles: int = 2
    core_cycles: int = 1
    mid_up_cycles: int = 2
    fine_up_cycles: int = 4
    n_heads: int = 8
    mlp_hidden_dim: int = 768
    max_seq_len: int = 256
    compression_factor: int = 4
    rope_base: float = 10000.0
    dropout: float = 0.0
    use_seq_rope: bool = True
    use_newline_rope: bool = True
    use_line_local_rope: bool = False
    line_pos_vocab_size: int = 512
    line_pos_dim: int = 32

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def free_dim(self) -> int:
        return self.d_model - self.vocab_size


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return self.dropout(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    rotary_dim = x.size(-1)
    if rotary_dim % 2 != 0:
        raise ValueError("Rotary dimension must be even")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos().to(dtype=x.dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=x.dtype).unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class DualRopeSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.track_names = []
        if config.use_seq_rope:
            self.track_names.append("seq")
        if config.use_newline_rope:
            self.track_names.append("newline")
        if config.use_line_local_rope:
            self.track_names.append("line_local")
        if not self.track_names:
            raise ValueError("At least one RoPE track must be enabled")

        if self.head_dim < 2 * len(self.track_names):
            raise ValueError("head_dim is too small for the number of RoPE tracks")
        self.track_dims = self._split_track_dims(self.head_dim, len(self.track_names))
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope_base = config.rope_base

    def forward(
        self,
        x: torch.Tensor,
        seq_positions: torch.Tensor,
        newline_positions: torch.Tensor,
        line_local_positions: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        position_map = {
            "seq": seq_positions,
            "newline": newline_positions,
            "line_local": line_local_positions,
        }

        q_parts = torch.split(q, self.track_dims, dim=-1)
        k_parts = torch.split(k, self.track_dims, dim=-1)
        q = torch.cat(
            [
                apply_rope(q_part, position_map[name], self.rope_base)
                for name, q_part in zip(self.track_names, q_parts, strict=True)
            ],
            dim=-1,
        )
        k = torch.cat(
            [
                apply_rope(k_part, position_map[name], self.rope_base)
                for name, k_part in zip(self.track_names, k_parts, strict=True)
            ],
            dim=-1,
        )

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(attn)

    @staticmethod
    def _split_track_dims(head_dim: int, num_tracks: int) -> list[int]:
        pair_budget = head_dim // 2
        base_pairs = pair_budget // num_tracks
        extra_pairs = pair_budget % num_tracks
        dims = []
        for i in range(num_tracks):
            pairs = base_pairs + (1 if i < extra_pairs else 0)
            dims.append(pairs * 2)
        if sum(dims) != head_dim:
            raise ValueError("Failed to partition head_dim across RoPE tracks")
        return dims


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = DualRopeSelfAttention(config)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden_dim, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_positions: torch.Tensor,
        newline_positions: torch.Tensor,
        line_local_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), seq_positions, newline_positions, line_local_positions)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class CharFormer(nn.Module):
    def __init__(self, config: ModelConfig, embedding_init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        if config.free_dim < 0:
            raise ValueError("d_model must be at least vocab_size")
        if config.line_pos_dim > config.free_dim:
            raise ValueError("line_pos_dim cannot exceed the free dimensions beyond vocab_size")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.line_pos_embedding = nn.Embedding(config.line_pos_vocab_size, config.line_pos_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pre_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_pre_layers)])
        self.bottleneck_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_bottleneck_layers)])
        self.post_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_post_layers)])
        self.use_bottleneck = config.compression_factor > 1
        if self.use_bottleneck:
            self.compress = BottleneckCompressor(config.d_model, config.compression_factor)
            self.expand = BottleneckExpander(config.d_model, config.compression_factor)
        else:
            self.compress = None
            self.expand = None
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if embedding_init is not None:
            if embedding_init.shape != (config.vocab_size, config.d_model):
                raise ValueError(
                    f"embedding_init shape {tuple(embedding_init.shape)} does not match "
                    f"({config.vocab_size}, {config.d_model})"
                )
            with torch.no_grad():
                self.token_embedding.weight.copy_(embedding_init)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_positions: torch.Tensor,
        newline_positions: torch.Tensor,
        char_positions_in_line: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(input_ids)
        clipped_line_positions = char_positions_in_line.clamp(max=self.config.line_pos_vocab_size - 1)
        line_features = self.line_pos_embedding(clipped_line_positions)
        if self.config.line_pos_dim > 0:
            start = self.config.vocab_size
            end = start + self.config.line_pos_dim
            x = x.clone()
            x[..., start:end] = x[..., start:end] + line_features
        x = self.dropout(x)

        for block in self.pre_blocks:
            x = block(x, seq_positions, newline_positions, char_positions_in_line)

        if self.use_bottleneck:
            compressed_x, compressed_seq, compressed_newline, compressed_char = self.compress(
                x,
                seq_positions,
                newline_positions,
                char_positions_in_line,
            )

            for block in self.bottleneck_blocks:
                compressed_x = block(compressed_x, compressed_seq, compressed_newline, compressed_char)

            x = x + self.expand(compressed_x, seq_positions.size(1))

        for block in self.post_blocks:
            x = block(x, seq_positions, newline_positions, char_positions_in_line)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class MultirateCharFormer(nn.Module):
    def __init__(self, config: ModelConfig, embedding_init: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        if config.free_dim < 0:
            raise ValueError("d_model must be at least vocab_size")
        if config.line_pos_dim > config.free_dim:
            raise ValueError("line_pos_dim cannot exceed the free dimensions beyond vocab_size")
        if config.fine_down_cycles % max(config.mid_down_cycles, 1) != 0:
            raise ValueError("fine_down_cycles must be divisible by mid_down_cycles")
        if config.mid_down_cycles % max(config.core_cycles, 1) != 0:
            raise ValueError("mid_down_cycles must be divisible by core_cycles")
        if config.mid_up_cycles <= 0 or config.fine_up_cycles <= 0:
            raise ValueError("mid_up_cycles and fine_up_cycles must be positive")

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.line_pos_embedding = nn.Embedding(config.line_pos_vocab_size, config.line_pos_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.fine_block = DecoderBlock(config)
        self.mid_block = DecoderBlock(config)
        self.core_block = DecoderBlock(config)

        self.fine_to_mid = BottleneckCompressor(config.d_model, 2)
        self.mid_to_core = BottleneckCompressor(config.d_model, 2)
        self.core_to_mid = BottleneckExpander(config.d_model, 2)
        self.mid_to_fine = BottleneckExpander(config.d_model, 2)

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if embedding_init is not None:
            if embedding_init.shape != (config.vocab_size, config.d_model):
                raise ValueError(
                    f"embedding_init shape {tuple(embedding_init.shape)} does not match "
                    f"({config.vocab_size}, {config.d_model})"
                )
            with torch.no_grad():
                self.token_embedding.weight.copy_(embedding_init)

    def _embed_inputs(self, input_ids: torch.Tensor, char_positions_in_line: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        clipped_line_positions = char_positions_in_line.clamp(max=self.config.line_pos_vocab_size - 1)
        line_features = self.line_pos_embedding(clipped_line_positions)
        if self.config.line_pos_dim > 0:
            start = self.config.vocab_size
            end = start + self.config.line_pos_dim
            x = x.clone()
            x[..., start:end] = x[..., start:end] + line_features
        return self.dropout(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_positions: torch.Tensor,
        newline_positions: torch.Tensor,
        char_positions_in_line: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fine = self._embed_inputs(input_ids, char_positions_in_line)

        mid, mid_seq, mid_newline, mid_char = self.fine_to_mid(
            fine,
            seq_positions,
            newline_positions,
            char_positions_in_line,
        )
        core, core_seq, core_newline, core_char = self.mid_to_core(mid, mid_seq, mid_newline, mid_char)

        fine_per_mid = self.config.fine_down_cycles // max(self.config.mid_down_cycles, 1)
        mid_step = 0
        for i in range(self.config.fine_down_cycles):
            fine = self.fine_block(fine, seq_positions, newline_positions, char_positions_in_line)
            if mid_step < self.config.mid_down_cycles and (i + 1) % fine_per_mid == 0:
                mid_refresh, mid_seq, mid_newline, mid_char = self.fine_to_mid(
                    fine,
                    seq_positions,
                    newline_positions,
                    char_positions_in_line,
                )
                mid = mid + mid_refresh
                mid = self.mid_block(mid, mid_seq, mid_newline, mid_char)
                mid_step += 1

        for _ in range(self.config.core_cycles):
            core_refresh, core_seq, core_newline, core_char = self.mid_to_core(mid, mid_seq, mid_newline, mid_char)
            core = core + core_refresh
            core = self.core_block(core, core_seq, core_newline, core_char)

        for _ in range(self.config.mid_up_cycles):
            mid = mid + self.core_to_mid(core, mid_seq.size(1))
            mid = self.mid_block(mid, mid_seq, mid_newline, mid_char)

        for _ in range(self.config.fine_up_cycles):
            fine = fine + self.mid_to_fine(mid, seq_positions.size(1))
            fine = self.fine_block(fine, seq_positions, newline_positions, char_positions_in_line)

        logits = self.lm_head(self.final_norm(fine))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class BottleneckCompressor(nn.Module):
    def __init__(self, d_model: int, compression_factor: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.compression_factor = compression_factor
        self.norm = RMSNorm(d_model * compression_factor)
        self.proj = nn.Linear(d_model * compression_factor, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_positions: torch.Tensor,
        newline_positions: torch.Tensor,
        char_positions_in_line: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        if seq_len % self.compression_factor != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by compression_factor {self.compression_factor}"
            )

        new_len = seq_len // self.compression_factor
        x = x.view(bsz, new_len, self.compression_factor * dim)
        x = self.proj(self.norm(x))

        compressed_seq = seq_positions[:, :: self.compression_factor]
        compressed_newline = newline_positions[:, :: self.compression_factor]
        compressed_char = char_positions_in_line[:, :: self.compression_factor]
        return x, compressed_seq, compressed_newline, compressed_char


class BottleneckExpander(nn.Module):
    def __init__(self, d_model: int, compression_factor: int) -> None:
        super().__init__()
        self.compression_factor = compression_factor
        self.proj = nn.Linear(d_model, d_model * compression_factor, bias=False)

    def forward(self, x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        bsz, compressed_len, dim = x.shape
        expanded = self.proj(x).view(bsz, compressed_len * self.compression_factor, dim)
        if expanded.size(1) != target_seq_len:
            raise ValueError(f"Expanded length {expanded.size(1)} does not match target {target_seq_len}")
        return expanded


def build_model(config: ModelConfig, embedding_init: torch.Tensor | None = None) -> nn.Module:
    if config.architecture == "multirate":
        return MultirateCharFormer(config, embedding_init=embedding_init)
    if config.architecture == "legacy":
        return CharFormer(config, embedding_init=embedding_init)
    raise ValueError(f"Unknown architecture '{config.architecture}'")
