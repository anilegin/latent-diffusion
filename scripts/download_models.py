from __future__ import annotations

import os
from pathlib import Path


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def download_torchvision_vgg16() -> None:
    """
    Required by LPIPS when using lpips_net='vgg'.

    This downloads:
        ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    """
    print_header("Downloading torchvision VGG16 weights")

    import torch
    from torchvision.models import vgg16, VGG16_Weights

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()

    print("VGG16 downloaded/loaded successfully.")
    print("Torch cache:", torch.hub.get_dir())


def download_lpips_weights() -> None:
    """
    Required by lpips.LPIPS(net='vgg').

    LPIPS itself has small learned linear calibration weights.
    Depending on the package version, these may already be inside site-packages,
    but instantiating LPIPS here verifies everything is available.
    """
    print_header("Initializing LPIPS VGG model")

    import lpips

    model = lpips.LPIPS(net="vgg")
    model.eval()

    print("LPIPS VGG initialized successfully.")


def download_clip() -> None:
    """
    This downloads Hugging Face CLIP tokenizer + text encoder.
    """
    print_header("Downloading CLIP text encoder for later LDM training")

    from transformers import CLIPTextModel, CLIPTokenizer

    model_id = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id)

    print("CLIP tokenizer loaded:", tokenizer.__class__.__name__)
    print("CLIP text encoder loaded:", text_encoder.__class__.__name__)


def verify_offline_lpips() -> None:
    """
    Test LPIPS under offline-ish environment.
    This should not attempt network if caches are ready.
    """
    print_header("Verifying LPIPS works")

    import torch
    import lpips

    loss_fn = lpips.LPIPS(net="vgg").eval()

    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        value = loss_fn(x, y)

    print("LPIPS forward OK. Value:", float(value.mean()))


def verify_clip_offline() -> None:
    print_header("Verifying CLIP works")

    from transformers import CLIPTextModel, CLIPTokenizer

    model_id = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_id, local_files_only=True)

    batch = tokenizer(
        ["a dog sitting on a bench"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    out = text_encoder(**batch)

    print("CLIP offline load OK.")
    print("last_hidden_state:", tuple(out.last_hidden_state.shape))


def main() -> None:
    print_header("Cache locations")

    home = Path.home()
    print("HOME:", home)
    print("TORCH_HOME:", os.environ.get("TORCH_HOME", "not set"))
    print("HF_HOME:", os.environ.get("HF_HOME", "not set"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE", "not set"))

    download_torchvision_vgg16()
    download_lpips_weights()
    verify_offline_lpips()

    # Needed later for text-conditioned LDM.
    download_clip()
    verify_clip_offline()

    print_header("All required models are cached successfully")


if __name__ == "__main__":
    main()