import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMG1_PATH = "data/raw/musk1.jpg"
DEFAULT_IMG2_PATH = "data/raw/musk2.jpg"
# sigma 對應 cv2.GaussianBlur 的標準差，數值越大越糊。
# 跟 gaussian_noise 版本的 sigma 概念一致，方便兩個實驗互相對照。
DEFAULT_SIGMAS = [1, 2, 3, 5, 8, 12, 16]
DEFAULT_OUTPUT_DIR = "results/gaussian_blur"


def sigma_to_kernel_size(sigma: float) -> int:
    if sigma <= 0:
        return 1
    k = int(round(sigma * 3)) * 2 + 1
    return max(k, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Gaussian blur to image files and save the results."
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
        help="Gaussian blur sigma values (larger = more blur).",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        nargs="+",
        help="Optional explicit odd kernel sizes paired with --sigma. "
             "If omitted kernel size is derived from sigma automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated images.",
    )
    return parser.parse_args()


def apply_gaussian_blur(
    image: np.ndarray, sigma: float, kernel_size: int | None = None
) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return image.copy()

    if kernel_size is None:
        kernel_size = sigma_to_kernel_size(sigma)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def resolve_blur_levels(args) -> list[tuple[str, float, int | None]]:
    sigmas = args.sigma
    kernels = args.kernel
    if kernels is not None and len(kernels) != len(sigmas):
        raise ValueError(
            f"--kernel 數量 ({len(kernels)}) 必須與 --sigma 數量 ({len(sigmas)}) 相同"
        )

    levels: list[tuple[str, float, int | None]] = []
    for index, sigma in enumerate(sigmas):
        kernel_size = kernels[index] if kernels is not None else None
        levels.append((f"sigma_{sigma:g}", float(sigma), kernel_size))
    return levels


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{path}")
    return image


def output_path_for(input_path: str, output_dir: str, blur_label: str) -> str:
    source = Path(input_path)
    safe_label = blur_label.replace(".", "p")
    return str(Path(output_dir) / f"{source.stem}_blur_{safe_label}{source.suffix}")


def resolve_input_images(args) -> list[str]:
    if args.images:
        return args.images
    return [args.img1, args.img2]


def main():
    args = parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        input_images = resolve_input_images(args)
        blur_levels = resolve_blur_levels(args)

        for image_path in input_images:
            image = load_image(image_path)

            for blur_label, sigma, kernel_size in blur_levels:
                blurred_image = apply_gaussian_blur(
                    image, sigma=sigma, kernel_size=kernel_size
                )
                output_path = output_path_for(image_path, args.output_dir, blur_label)

                if not cv2.imwrite(output_path, blurred_image):
                    raise OSError(f"無法寫入圖片：{output_path}")

                used_kernel = kernel_size or sigma_to_kernel_size(sigma)
                print(
                    f"已輸出：{output_path}（sigma={sigma:g}, kernel={used_kernel}）"
                )

    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
