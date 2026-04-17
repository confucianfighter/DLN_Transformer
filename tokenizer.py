from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


TEXT_TOKEN_RE = re.compile(r"[A-Za-z]+|[^A-Za-z]", re.UNICODE)
EDGE_LABELS = ("1", "-1", "2", "-2", "3", "-3")
POSITIONAL_TOKEN_RE = re.compile(r"^([a-z])(1|2|3|middle|-3|-2|-1)$")


@dataclass(frozen=True)
class TokenPiece:
    char: str
    slot: str
    token: str
    char_index: int


@dataclass(frozen=True)
class StreamToken:
    token: str
    is_alpha_piece: bool
    is_word_final: bool


class PositionalCharTokenizer:
    """
    Character tokenizer with fixed edge-aware position labels.

    Resolution order is based on Python-like indexing:
    0, -1, 1, -2, 2, -3, 3, ...

    Labels are assigned in this order:
    1, -1, 2, -2, 3, -3, middle, middle, ...

    Tokens are emitted back in the original reading order.

    Example:
    assignment order for "Thing" -> T1, g-1, h2, n-2, i3
    emitted order                -> T1, h2, i3, n-2, g-1

    Example:
    emitted order for "Position" -> P1, o2, s3, iMiddle, tMiddle, i-3, o-2, n-1
    """

    def __init__(self) -> None:
        self.special_tokens = ["<PAD>", "<UNK>"]
        self._token_to_id: dict[str, int] = {token: i for i, token in enumerate(self.special_tokens)}
        self._id_to_token: list[str] = list(self.special_tokens)

    @classmethod
    def word_to_pieces(cls, word: str) -> list[TokenPiece]:
        if not word:
            return []

        pieces_by_index: list[TokenPiece | None] = [None] * len(word)
        labels = cls._labels_for_length(len(word))

        for char_index, slot in zip(cls._assignment_indices(len(word)), labels, strict=True):
            char = word[char_index]
            token = f"{char}{slot}"
            pieces_by_index[char_index] = TokenPiece(
                char=char,
                slot=slot,
                token=token,
                char_index=char_index,
            )

        return [piece for piece in pieces_by_index if piece is not None]

    @staticmethod
    def _assignment_indices(length: int) -> list[int]:
        indices: list[int] = []
        seen: set[int] = set()
        step = 0

        while len(indices) < length:
            if step == 0:
                candidate = 0
            elif step % 2 == 1:
                candidate = -((step + 1) // 2)
            else:
                candidate = step // 2

            resolved = candidate if candidate >= 0 else length + candidate
            if 0 <= resolved < length and resolved not in seen:
                indices.append(resolved)
                seen.add(resolved)
            step += 1

        return indices

    @staticmethod
    def _labels_for_length(length: int) -> list[str]:
        labels = list(EDGE_LABELS[: min(length, len(EDGE_LABELS))])
        if length > len(labels):
            labels.extend(["middle"] * (length - len(labels)))
        return labels

    @staticmethod
    def split_text(text: str) -> list[str]:
        return TEXT_TOKEN_RE.findall(text)

    @staticmethod
    def _is_alpha_word(token: str) -> bool:
        return token.isalpha()

    def text_to_tokens(self, text: str) -> list[str]:
        return [item.token for item in self.text_to_stream(text)]

    def text_to_stream(self, text: str) -> list[StreamToken]:
        tokens: list[str] = []
        for item in self.split_text(text):
            if self._is_alpha_word(item):
                pieces = self.word_to_pieces(item.lower())
                for i, piece in enumerate(pieces):
                    tokens.append(
                        StreamToken(
                            token=piece.token,
                            is_alpha_piece=True,
                            is_word_final=i == len(pieces) - 1,
                        )
                    )
            else:
                tokens.append(StreamToken(token=item, is_alpha_piece=False, is_word_final=False))
        return tokens

    def fit(self, text_or_tokens: Iterable[str] | str) -> None:
        stream = self.text_to_tokens(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        for token in stream:
            self._ensure_token(token)

    def encode_word(self, word: str) -> list[int]:
        return [self.token_to_id(piece.token) for piece in self.word_to_pieces(word.lower())]

    def encode_text(self, text: str) -> list[int]:
        return [self.token_to_id(token) for token in self.text_to_tokens(text)]

    def decode_ids(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token(token_id) for token_id in ids]

    @staticmethod
    def render_tokens(tokens: Iterable[str]) -> str:
        chars: list[str] = []
        for token in tokens:
            match = POSITIONAL_TOKEN_RE.fullmatch(token)
            if match:
                chars.append(match.group(1))
            elif token in {"<PAD>", "<UNK>"}:
                continue
            else:
                chars.append(token)
        return "".join(chars)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id["<UNK>"])

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def save_vocab(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._id_to_token, indent=2), encoding="utf-8")

    def load_vocab(self, path: str | Path) -> None:
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        self._id_to_token = list(tokens)
        self._token_to_id = {token: i for i, token in enumerate(self._id_to_token)}

    def _ensure_token(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        token_id = len(self._id_to_token)
        self._id_to_token.append(token)
        self._token_to_id[token] = token_id
        return token_id


def build_vocab_from_file(input_path: Path, output_path: Path) -> PositionalCharTokenizer:
    tokenizer = PositionalCharTokenizer()
    text = input_path.read_text(encoding="utf-8")
    tokenizer.fit(text)
    tokenizer.save_vocab(output_path)
    return tokenizer


def sample_excerpt(text: str, target_chars: int = 1500, seed: int = 7) -> str:
    if len(text) <= target_chars:
        return text

    rng = random.Random(seed)
    start = rng.randint(0, max(0, len(text) - target_chars))

    while start > 0 and text[start - 1] not in "\n ":
        start -= 1

    end = min(len(text), start + target_chars)
    while end < len(text) and text[end] not in "\n ":
        end += 1

    return text[start:end].strip()


def write_human_readable_sample(
    tokenizer: PositionalCharTokenizer,
    input_path: Path,
    output_path: Path,
    target_chars: int = 1500,
    seed: int = 7,
) -> Path:
    text = input_path.read_text(encoding="utf-8")
    excerpt = sample_excerpt(text, target_chars=target_chars, seed=seed)
    tokens = tokenizer.text_to_tokens(excerpt)
    token_lines = textwrap.wrap(" ".join(tokens), width=100, break_long_words=False, break_on_hyphens=False)

    lines = [
        "Tiny Shakespeare Sample",
        f"source: {input_path}",
        f"seed: {seed}",
        f"excerpt_chars: {len(excerpt)}",
        "",
        "Original excerpt",
        "----------------",
        excerpt,
        "",
        "Tokenized excerpt",
        "-----------------",
        *token_lines,
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def is_whitespace_token(token: str) -> bool:
    return token.isspace()


def build_transition_statistics(
    tokenizer: PositionalCharTokenizer,
    text: str,
) -> tuple[np.ndarray, np.ndarray]:
    stream = tokenizer.text_to_stream(text)
    vocab_size = tokenizer.vocab_size()
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    for i, current in enumerate(stream[:-1]):
        next_index = i + 1

        if current.is_alpha_piece and current.is_word_final:
            while next_index < len(stream) and is_whitespace_token(stream[next_index].token):
                next_index += 1

        if next_index >= len(stream):
            continue

        current_id = tokenizer.token_to_id(current.token)
        next_id = tokenizer.token_to_id(stream[next_index].token)
        counts[current_id, next_id] += 1

    probs = counts.astype(np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    np.divide(probs, row_sums, out=probs, where=row_sums > 0)
    return counts, probs


def build_embedding_initializer(
    probs: np.ndarray,
    embedding_dim: int = 256,
) -> np.ndarray:
    vocab_size = probs.shape[0]
    if embedding_dim < vocab_size:
        raise ValueError(f"embedding_dim={embedding_dim} must be at least vocab_size={vocab_size}")

    embedding = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    embedding[:, :vocab_size] = probs.astype(np.float32)
    return embedding


def write_transition_report(
    tokenizer: PositionalCharTokenizer,
    counts: np.ndarray,
    probs: np.ndarray,
    output_path: Path,
    top_k: int = 8,
) -> Path:
    lines = [
        "Positional Token Transition Report",
        f"vocab_size: {tokenizer.vocab_size()}",
        "",
    ]

    for token_id, token in enumerate(tokenizer._id_to_token):
        total = int(counts[token_id].sum())
        if total == 0:
            continue

        top_ids = np.argsort(probs[token_id])[::-1][:top_k]
        entries = []
        for next_id in top_ids:
            prob = float(probs[token_id, next_id])
            count = int(counts[token_id, next_id])
            if count == 0:
                continue
            entries.append(f"{tokenizer.id_to_token(int(next_id))} ({prob:.4f}, count={count})")

        if not entries:
            continue

        lines.append(f"{token} -> total={total}")
        lines.extend(f"  {entry}" for entry in entries)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def save_transition_artifacts(
    tokenizer: PositionalCharTokenizer,
    counts: np.ndarray,
    probs: np.ndarray,
    embedding: np.ndarray,
    counts_path: Path,
    probs_path: Path,
    embedding_path: Path,
    report_path: Path,
) -> None:
    np.save(counts_path, counts)
    np.save(probs_path, probs)
    np.save(embedding_path, embedding)
    write_transition_report(tokenizer, counts, probs, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect the positional character tokenizer.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tiny_shakespeare.txt"),
        help="Text file used to build the vocabulary.",
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("data/positional_char_vocab.json"),
        help="Output path for the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--inspect-word",
        type=str,
        default="Thing",
        help="Word to inspect after building the tokenizer.",
    )
    parser.add_argument(
        "--sample-out",
        type=Path,
        default=Path("data/tiny_shakespeare_sample_tokens.txt"),
        help="Output path for a readable tokenized excerpt.",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=1500,
        help="Approximate number of characters to include in the sample excerpt.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=7,
        help="Seed used to choose the sample excerpt.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding width for the initializer matrix.",
    )
    parser.add_argument(
        "--transition-counts-out",
        type=Path,
        default=Path("data/transition_counts.npy"),
        help="Output path for the transition count matrix.",
    )
    parser.add_argument(
        "--transition-probs-out",
        type=Path,
        default=Path("data/transition_probs.npy"),
        help="Output path for the transition probability matrix.",
    )
    parser.add_argument(
        "--embedding-init-out",
        type=Path,
        default=Path("data/embedding_init.npy"),
        help="Output path for the embedding initializer matrix.",
    )
    parser.add_argument(
        "--transition-report-out",
        type=Path,
        default=Path("data/transition_report.txt"),
        help="Output path for a readable top-continuation report.",
    )
    args = parser.parse_args()

    input_text = args.input.read_text(encoding="utf-8")
    tokenizer = build_vocab_from_file(args.input, args.vocab_out)
    pieces = tokenizer.word_to_pieces(args.inspect_word.lower())
    sample_path = write_human_readable_sample(
        tokenizer,
        input_path=args.input,
        output_path=args.sample_out,
        target_chars=args.sample_chars,
        seed=args.sample_seed,
    )
    counts, probs = build_transition_statistics(tokenizer, input_text)
    embedding = build_embedding_initializer(probs, embedding_dim=args.embedding_dim)
    save_transition_artifacts(
        tokenizer,
        counts,
        probs,
        embedding,
        counts_path=args.transition_counts_out,
        probs_path=args.transition_probs_out,
        embedding_path=args.embedding_init_out,
        report_path=args.transition_report_out,
    )

    print(f"built vocab with {tokenizer.vocab_size()} tokens")
    print(f"inspect word: {args.inspect_word}")
    print("tokens:", [piece.token for piece in pieces])
    print("ids:", tokenizer.encode_word(args.inspect_word))
    print(f"saved vocab to {args.vocab_out}")
    print(f"saved sample to {sample_path}")
    print(f"saved transition counts to {args.transition_counts_out}")
    print(f"saved transition probabilities to {args.transition_probs_out}")
    print(f"saved embedding initializer to {args.embedding_init_out}")
    print(f"saved transition report to {args.transition_report_out}")


if __name__ == "__main__":
    main()
