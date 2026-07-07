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
        "--teacher-model-id",
        type=str,
        default="Lykon/dreamshaper-8",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sd1-5_pruned/depth1",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--layers-per-block",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--transformer-layers-per-block",
        type=int,
        default=1,
    )

    return parser.parse_args()


def build_student_config(
    teacher_config,
    sample_size: int,
    layers_per_block: int,
    transformer_layers_per_block: int,
) -> dict:
    cfg = dict(teacher_config)

    cfg["sample_size"] = sample_size
    cfg["layers_per_block"] = layers_per_block
    cfg["transformer_layers_per_block"] = transformer_layers_per_block

    return cfg


def copy_exact_matching_weights(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
):
    teacher_sd = teacher.state_dict()
    student_sd = student.state_dict()

    new_sd = {}
    loaded = []
    skipped = []

    loaded_numel = 0
    total_numel = 0

    for key, student_value in student_sd.items():
        total_numel += student_value.numel()

        if key in teacher_sd and teacher_sd[key].shape == student_value.shape:
            new_sd[key] = teacher_sd[key]
            loaded.append(key)
            loaded_numel += student_value.numel()
        else:
            new_sd[key] = student_value
            skipped.append(key)

    student.load_state_dict(new_sd, strict=True)

    stats = {
        "loaded_tensors": len(loaded),
        "skipped_tensors": len(skipped),
        "loaded_parameters": loaded_numel,
        "total_student_parameters": total_numel,
        "loaded_parameter_ratio": loaded_numel / max(1, total_numel),
        "loaded_keys": loaded,
        "skipped_keys": skipped,
    }

    return stats


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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading teacher:", args.teacher_model_id)

    teacher = UNet2DConditionModel.from_pretrained(
        args.teacher_model_id,
        subfolder="unet",
        torch_dtype=torch.float32,
    )

    teacher_config = dict(teacher.config)

    student_config = build_student_config(
        teacher_config=teacher_config,
        sample_size=args.sample_size,
        layers_per_block=args.layers_per_block,
        transformer_layers_per_block=args.transformer_layers_per_block,
    )

    print("Building student U-Net")
    student = UNet2DConditionModel.from_config(student_config)

    print("Teacher params:", count_parameters(teacher) / 1e6, "M")
    print("Student params:", count_parameters(student) / 1e6, "M")

    print("Copying exact matching weights")
    stats = copy_exact_matching_weights(
        teacher=teacher,
        student=student,
    )

    print("Loaded tensors:", stats["loaded_tensors"])
    print("Skipped tensors:", stats["skipped_tensors"])
    print("Loaded parameter ratio:", stats["loaded_parameter_ratio"])

    student.save_pretrained(output_dir)

    with open(output_dir / "pruning_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(output_dir / "teacher_config.json", "w", encoding="utf-8") as f:
        json.dump(teacher_config, f, indent=2)

    with open(output_dir / "student_config.json", "w", encoding="utf-8") as f:
        json.dump(student_config, f, indent=2)

    print("Saved pruned student U-Net to:", output_dir)


if __name__ == "__main__":
    main()