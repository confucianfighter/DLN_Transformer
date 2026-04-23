import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

BASELINE_DIR = Path(__file__).resolve().parent / "external_baselines"
sys.path.insert(0, str(BASELINE_DIR))

from hymba_text_repeat_runner import (
    HymbaLM,
    build_itos,
    parse_layer_indices,
    parse_schedule,
    repeat_evaluate,
    write_non_exact_samples,
)
from tcn_text_repeat_runner import generate_repeat_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_path", default="./data/tiny_shakespeare.txt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _, _, test_x, test_y, n_classes, _, marker_id = generate_repeat_data(
        args.data_path,
        ckpt_args["mem_len"],
        ckpt_args["blank_len"],
        ckpt_args["n_train"],
        ckpt_args["n_test"],
        ckpt_args["split"],
        ckpt_args["seed"],
        ckpt_args.get("span_mode", "sentence"),
        ckpt_args.get("chunk_span_len", 0),
    )
    model = HymbaLM(
        vocab_size=n_classes,
        embed_dim=ckpt_args["embed_dim"],
        n_layers=ckpt_args["n_layers"],
        d_state=ckpt_args["d_state"],
        n_heads=ckpt_args["n_heads"],
        n_kv_heads=ckpt_args["n_kv_heads"],
        dropout=ckpt_args["dropout"],
        seq_len=test_x.size(1),
        branch_scale_init=ckpt_args["branch_scale_init"],
        n_meta_tokens=ckpt_args["n_meta_tokens"],
        window_size=ckpt_args["window_size"],
        full_attention_layers=parse_layer_indices(ckpt_args["full_attention_layers"], ckpt_args["n_layers"]),
        multirate_schedule=parse_schedule(ckpt_args["multirate_schedule"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    metrics = repeat_evaluate(
        model,
        test_x,
        test_y,
        criterion,
        ckpt_args["eval_batch_size"],
        device,
        ckpt_args["mem_len"],
    )
    itos = build_itos(args.data_path)
    write_non_exact_samples(
        model,
        test_x,
        test_y,
        ckpt_args["eval_batch_size"],
        device,
        ckpt_args["mem_len"],
        itos,
        marker_id,
        args.output,
        args.max_samples,
    )
    print(metrics)


if __name__ == "__main__":
    main()
