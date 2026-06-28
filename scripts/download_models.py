from __future__ import annotations

import os
from pathlib import Path


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_model_list() -> None:
    print_header("Models cached by this script")

    models = [
        ("1. VGG16", "LPIPS backbone"),
        ("2. LPIPS VGG", "perceptual metric weights"),
        ("3. InceptionV3", "rFID/FID feature extractor"),
        ("4. CLIP ViT-B/32", "text encoder for LDM"),
    ]

    for name, comment in models:
        print(f"{name:18s} # {comment}")


def download_torchvision_vgg16() -> None:
    """
    Required by LPIPS when using lpips_net='vgg'.

    This downloads:
        ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
    """
    print_header("1. VGG16")

    import torch
    from torchvision.models import vgg16, VGG16_Weights

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()

    print("VGG16 downloaded/loaded successfully.")
    print("Torch cache:", torch.hub.get_dir())


def download_lpips_weights() -> None:
    """
    Required by lpips.LPIPS(net='vgg').

    LPIPS has small learned calibration weights.
    Instantiating it verifies the weights are available.
    """
    print_header("2. LPIPS VGG")

    import lpips

    model = lpips.LPIPS(net="vgg")
    model.eval()

    print("LPIPS VGG initialized successfully.")


def download_torchvision_inception_v3() -> None:
    """
    Required by evaluate_vae_reconstruction.py when rFID/FID is enabled.

    This downloads:
        ~/.cache/torch/hub/checkpoints/inception_v3_google-*.pth
    """
    print_header("3. InceptionV3")

    import torch
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
        aux_logits=True,
    )
    model.eval()

    print("InceptionV3 downloaded/loaded successfully.")
    print("Torch cache:", torch.hub.get_dir())


def download_clip() -> None:
    """
    Required later for text-conditioned LDM training.
    """
    print_header("4. CLIP ViT-B/32")

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
    print_header("Verifying LPIPS")

    import torch
    import lpips

    loss_fn = lpips.LPIPS(net="vgg").eval()

    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        value = loss_fn(x, y)

    print("LPIPS forward OK. Value:", float(value.mean()))


def verify_offline_inception_v3() -> None:
    """
    Test InceptionV3 under offline-ish environment.
    This should not attempt network if caches are ready.
    """
    print_header("Verifying InceptionV3")

    import torch
    import torch.nn.functional as F
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
        aux_logits=True,
    )
    model.fc = torch.nn.Identity()
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    x = F.interpolate(
        x,
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        out = model(x)

    if isinstance(out, tuple):
        out = out[0]

    print("InceptionV3 forward OK.")
    print("Feature shape:", tuple(out.shape))


def verify_clip_offline() -> None:
    print_header("Verifying CLIP")

    from transformers import CLIPTextModel, CLIPTokenizer

    model_id = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        local_files_only=True,
    )

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

    print_model_list()

    download_torchvision_vgg16()
    download_lpips_weights()
    download_torchvision_inception_v3()
    download_clip()

    verify_offline_lpips()
    verify_offline_inception_v3()
    verify_clip_offline()

    print_header("All required models are cached successfully")


if __name__ == "__main__":
    main()