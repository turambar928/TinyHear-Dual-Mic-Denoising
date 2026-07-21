#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def c_vector(name: str, values) -> str:
    flat = values.detach().cpu().float().reshape(-1).numpy()
    body = ", ".join(f"{float(v):.9e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};\n"


def c_matrix(name: str, values) -> str:
    arr = values.detach().cpu().float().numpy()
    rows = []
    for row in arr:
        rows.append("{" + ", ".join(f"{float(v):.9e}f" for v in row) + "}")
    return f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{{', '.join(rows)}}};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--out", default="c_reference/generated/tiny_gate_params.h")
    args = parser.parse_args()

    ckpt = torch.load(args.gate, map_location="cpu")
    cfg = ckpt["config"]
    state = ckpt["model"]
    input_dim = int(cfg["input_dim"])
    hidden = int(cfg["hidden"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "#pragma once\n\n",
        f"#define TINY_GATE_INPUT_DIM {input_dim}\n",
        f"#define TINY_GATE_HIDDEN {hidden}\n\n",
        c_matrix("kTinyGateFc1Weight", state["net.0.weight"]),
        c_vector("kTinyGateFc1Bias", state["net.0.bias"]),
        c_vector("kTinyGateFc2Weight", state["net.2.weight"]),
        f"static const float kTinyGateFc2Bias = {float(state['net.2.bias'].reshape(-1)[0]):.9e}f;\n",
    ]
    out.write_text("".join(header), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
