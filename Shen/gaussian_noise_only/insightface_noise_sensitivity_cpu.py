import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMG1_PATH = "data/raw/musk1.jpg"
DEFAULT_IMG2_PATH = "data/raw/musk2.jpg"
DEFAULT_SIGMAS = [5, 10, 20, 40, 60, 80, 100]
DEFAULT_OUTPUT_DIR = "results/gaussian_noise"


def percent_to_sigma(percent: float) -> float:
    if percent < 0:
        raise ValueError("percent must be non-negative")
    return 255 * (percent / 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to image files and save the results."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Input image paths. If omitted, --img1/--img2 or the default paths are used.",
    )
    parser.add_argument("--img1", type=str, default=DEFAULT_IMG1_PATH)
    parser.add_argument("--img2", type=str, default=DEFAULT_IMG2_PATH)
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=DEFAULT_SIGMAS,
        help="Gaussian noise standard deviation values.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        nargs="+",
        help="Noise strength as a percentage of the 0-255 pixel range. Example: --percent 30",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise generation.",
    )
    return parser.parse_args()


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def resolve_noise_levels(args) -> list[tuple[str, float]]:
    if args.percent:
        return [(f"{percent:g}pct", percent_to_sigma(percent)) for percent in args.percent]
    return [(f"sigma_{sigma:g}", sigma) for sigma in args.sigma]


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{path}")
    return image


def output_path_for(input_path: str, output_dir: str, noise_label: str) -> str:
    source = Path(input_path)
    safe_label = noise_label.replace(".", "p")
    return str(Path(output_dir) / f"{source.stem}_gaussian_{safe_label}{source.suffix}")


def resolve_input_images(args) -> list[str]:
    if args.images:
        return args.images
    return [args.img1, args.img2]


def main():
    args = parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        input_images = resolve_input_images(args)
        noise_levels = resolve_noise_levels(args)

        for image_index, image_path in enumerate(input_images):
            image = load_image(image_path)

            for noise_index, (noise_label, sigma) in enumerate(noise_levels):
                seed = args.seed + image_index * 100_000 + noise_index
                noisy_image = add_gaussian_noise(image, sigma=sigma, seed=seed)
                output_path = output_path_for(image_path, args.output_dir, noise_label)

                if not cv2.imwrite(output_path, noisy_image):
                    raise OSError(f"無法寫入圖片：{output_path}")

                print(f"已輸出：{output_path}（sigma={sigma:g}）")

    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
