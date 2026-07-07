from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from diffusers import UNet2DConditionModel


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="Lykon/dreamshaper-8",
    )

    parser.add_argument(
        "--subfolder",
        type=str,
        default="unet",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/sd1-5_pruned/sd1-5_unet_config.json",
    )

    return parser.parse_args()


def main():
    # Load variables from .env
    load_dotenv()

    token = os.getenv("HF_TOKEN")

    if token is None:
        raise RuntimeError(
            "HF_TOKEN not found. Create a .env file with:\n"
            "HF_TOKEN=hf_your_token_here"
        )

    # Debug only: print partial token, never full token
    print("HF token loaded:", token[:8] + "...")

    args = parse_args()

    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder=args.subfolder,
        torch_dtype=torch.float32,
        token=token,
    )

    cfg = dict(unet.config)

    print("============================================")
    print("SD1-5 U-Net inspection")
    print("Model ID:", args.model_id)
    print("Subfolder:", args.subfolder)
    print("Parameters:", count_parameters(unet) / 1e6, "M")
    print("============================================")

    keys = [
        "sample_size",
        "in_channels",
        "out_channels",
        "down_block_types",
        "up_block_types",
        "block_out_channels",
        "layers_per_block",
        "cross_attention_dim",
        "attention_head_dim",
        "num_attention_heads",
        "transformer_layers_per_block",
        "use_linear_projection",
        "norm_num_groups",
        "norm_eps",
        "act_fn",
        "resnet_time_scale_shift",
        "class_embed_type",
        "addition_embed_type",
        "mid_block_type",
    ]

    for key in keys:
        print(f"{key}: {cfg.get(key)}")

    print("============================================")
    print("Down blocks")
    print("============================================")

    for i, block in enumerate(unet.down_blocks):
        print(f"down_blocks.{i}: {block.__class__.__name__}")

        if hasattr(block, "resnets"):
            print("  resnets:", len(block.resnets))

        if hasattr(block, "attentions"):
            print("  attentions:", len(block.attentions))

        if hasattr(block, "downsamplers") and block.downsamplers is not None:
            print("  downsamplers:", len(block.downsamplers))

    print("============================================")
    print("Mid block")
    print("============================================")

    print("mid_block:", unet.mid_block.__class__.__name__)

    if hasattr(unet.mid_block, "resnets"):
        print("  resnets:", len(unet.mid_block.resnets))

    if hasattr(unet.mid_block, "attentions"):
        print("  attentions:", len(unet.mid_block.attentions))

    print("============================================")
    print("Up blocks")
    print("============================================")

    for i, block in enumerate(unet.up_blocks):
        print(f"up_blocks.{i}: {block.__class__.__name__}")

        if hasattr(block, "resnets"):
            print("  resnets:", len(block.resnets))

        if hasattr(block, "attentions"):
            print("  attentions:", len(block.attentions))

        if hasattr(block, "upsamplers") and block.upsamplers is not None:
            print("  upsamplers:", len(block.upsamplers))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Saved config to:", output_path)


if __name__ == "__main__":
    main()