import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMG1_PATH = "data/raw/musk1.jpg"
DEFAULT_IMG2_PATH = "data/raw/musk2.jpg"
# eps 為 [0,1] 區間的擾動上限，跟其他 noise/blur 版本維持「sigma list」式的多級設定。
DEFAULT_EPS_LIST = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12]
DEFAULT_OUTPUT_DIR = "results/fgsm"
DEFAULT_SURROGATE = "resnet18"
DEFAULT_TARGET_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply FGSM adversarial perturbation to image files and save the results."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Input image paths. If omitted, --img1/--img2 or the default paths are used.",
    )
    parser.add_argument("--img1", type=str, default=DEFAULT_IMG1_PATH)
    parser.add_argument("--img2", type=str, default=DEFAULT_IMG2_PATH)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=DEFAULT_EPS_LIST,
        help="FGSM epsilon values in [0,1] domain (越大擾動越強).",
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default=DEFAULT_SURROGATE,
        choices=["resnet18", "simple_cnn"],
        help="替身模型 (用來算梯度): resnet18 (pretrained) 或 simple_cnn (隨機權重).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_TARGET_SIZE),
        metavar=("H", "W"),
        help="替身模型輸入尺寸（梯度經 bilinear 縮放回原圖）.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated images.",
    )
    parser.add_argument(
        "--save-noise-map",
        action="store_true",
        help="額外輸出 noise map (heatmap overlay).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (給 simple_cnn fallback 用，pretrained 不受影響).",
    )
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{path}")
    return image


def bgr_to_rgb_float01(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def rgb_float01_to_bgr_uint8(rgb_float01: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb_float01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def build_simple_cnn(num_classes: int = 1000):
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return SimpleCNN()


def build_surrogate(model_name: str, seed: int = 42):
    """
    Prefer pretrained torchvision model. Fallback to a small untrained CNN
    if torchvision weights cannot be downloaded.
    """
    import torch

    torch.manual_seed(seed)

    name = model_name.lower().strip()
    if name == "resnet18":
        try:
            import torchvision.models as models
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            return model, weights
        except Exception as exc:
            print(
                f"[warning] 無法載入 pretrained resnet18 ({exc})，改用 simple_cnn fallback。",
                file=sys.stderr,
            )

    return build_simple_cnn(num_classes=1000), None


def fgsm_attack(
    image_bgr: np.ndarray,
    eps: float,
    surrogate,
    weights,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
) -> np.ndarray:
    """
    對單張 BGR 圖片做 untargeted FGSM 擾動，回傳 BGR uint8 對抗樣本。
    """
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if eps == 0:
        return image_bgr.copy()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cpu"
    surrogate = surrogate.to(device).eval()

    rgb = bgr_to_rgb_float01(image_bgr)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    x.requires_grad_(True)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def normalize_for_model(x_resized: torch.Tensor) -> torch.Tensor:
        if weights is None:
            return x_resized
        meta = getattr(weights, "meta", {}) or {}
        mean_val = meta.get("mean", meta.get("image_mean", imagenet_mean))
        std_val = meta.get("std", meta.get("image_std", imagenet_std))
        mean = torch.tensor(mean_val, device=x_resized.device).view(1, 3, 1, 1)
        std = torch.tensor(std_val, device=x_resized.device).view(1, 3, 1, 1)
        return (x_resized - mean) / std

    with torch.no_grad():
        x_res = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
        x_res = normalize_for_model(x_res)
        y = int(surrogate(x_res).argmax(dim=1).item())

    if x.grad is not None:
        x.grad.zero_()
    x_res = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
    x_res = normalize_for_model(x_res)
    logits = surrogate(x_res)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([y], device=device))
    loss.backward()

    grad_sign = x.grad.detach().sign()
    x_adv = torch.clamp(x.detach() + eps * grad_sign, 0.0, 1.0)

    adv_rgb = x_adv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return rgb_float01_to_bgr_uint8(adv_rgb)


def save_noise_map(adv_bgr: np.ndarray, base_bgr: np.ndarray, out_path: str) -> None:
    diff = np.abs(adv_bgr.astype(np.int32) - base_bgr.astype(np.int32)).astype(np.float32)
    heat = diff.max(axis=2)
    heat_norm = heat / (heat.max() + 1e-12)
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_bgr, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(out_path, overlay)


def resolve_eps_levels(args) -> list[tuple[str, float]]:
    return [(f"eps_{eps:g}", float(eps)) for eps in args.eps]


def output_path_for(input_path: str, output_dir: str, eps_label: str) -> str:
    source = Path(input_path)
    safe_label = eps_label.replace(".", "p")
    return str(Path(output_dir) / f"{source.stem}_fgsm_{safe_label}{source.suffix}")


def resolve_input_images(args) -> list[str]:
    if args.images:
        return args.images
    return [args.img1, args.img2]


def main():
    args = parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        input_images = resolve_input_images(args)
        eps_levels = resolve_eps_levels(args)

        surrogate, weights = build_surrogate(args.surrogate, seed=args.seed)

        for image_path in input_images:
            image = load_image(image_path)

            for eps_label, eps in eps_levels:
                adv_image = fgsm_attack(
                    image,
                    eps=eps,
                    surrogate=surrogate,
                    weights=weights,
                    target_size=tuple(args.target_size),
                )
                output_path = output_path_for(image_path, args.output_dir, eps_label)

                if not cv2.imwrite(output_path, adv_image):
                    raise OSError(f"無法寫入圖片：{output_path}")

                print(f"已輸出：{output_path}（eps={eps:g}）")

                if args.save_noise_map:
                    noise_path = output_path.replace("_fgsm_", "_fgsm_noise_")
                    save_noise_map(adv_image, image, noise_path)
                    print(f"已輸出 noise map：{noise_path}")

    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
