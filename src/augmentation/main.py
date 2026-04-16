from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "dataset" / "asl_dataset_2"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "asl_dataset_3"
LETTER_CLASS_NAMES = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
IMAGES_PER_LETTER = 1000
IMAGE_SIZE = (400, 400)
MASK_THRESHOLD = 45.0
MASK_BLUR_RADIUS = 6
SEED = 42


def list_images(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def estimate_background_color(image_array: np.ndarray, border_width: int = 12) -> np.ndarray:
    height, width, _ = image_array.shape
    border_width = max(1, min(border_width, height // 4, width // 4))

    top = image_array[:border_width, :, :]
    bottom = image_array[-border_width:, :, :]
    left = image_array[:, :border_width, :]
    right = image_array[:, -border_width:, :]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def build_foreground_mask(image_array: np.ndarray) -> Image.Image:
    background_color = estimate_background_color(image_array)
    color_distance = np.linalg.norm(
        image_array.astype(np.float32) - background_color.astype(np.float32),
        axis=2,
    )

    mask_array = np.where(color_distance >= MASK_THRESHOLD, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_array)
    mask = mask.filter(ImageFilter.MaxFilter(size=3))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=MASK_BLUR_RADIUS))
    return mask


def make_gradient_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    color_a = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    color_b = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)

    axis = random.choice(["x", "y", "diag"])
    if axis == "x":
        ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        base = np.repeat(ramp, height, axis=0)
    elif axis == "y":
        ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        base = np.repeat(ramp, width, axis=1)
    else:
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        base = ((x + y) / 2.0)[:, :, None]

    image = color_a * (1.0 - base) + color_b * base
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def make_noise_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    palette = np.array(
        [[random.randint(0, 255) for _ in range(3)] for _ in range(3)],
        dtype=np.float32,
    )
    weights = np.random.rand(height, width, 3).astype(np.float32)
    weights /= weights.sum(axis=2, keepdims=True)
    image = weights @ palette
    background = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    return background.filter(ImageFilter.GaussianBlur(radius=random.randint(2, 6)))


def make_striped_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    color_a = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    color_b = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
    stripe_width = random.randint(20, 60)
    vertical = random.choice([True, False])

    if vertical:
        for start in range(0, width, stripe_width):
            color = color_a if (start // stripe_width) % 2 == 0 else color_b
            image[:, start:start + stripe_width, :] = color
    else:
        for start in range(0, height, stripe_width):
            color = color_a if (start // stripe_width) % 2 == 0 else color_b
            image[start:start + stripe_width, :, :] = color

    return Image.fromarray(image).filter(
        ImageFilter.GaussianBlur(radius=random.randint(1, 3))
    )


def create_random_background(size: tuple[int, int]) -> Image.Image:
    generator = random.choice(
        [make_gradient_background, make_noise_background, make_striped_background]
    )
    return generator(size)


def get_images_for_letter(source_root: Path, letter: str) -> list[Path]:
    letter_root = source_root / letter
    if not letter_root.exists():
        raise ValueError(f"Missing source folder for letter '{letter}': {letter_root}")

    images = list_images(letter_root)
    if not images:
        raise ValueError(f"No images found for letter '{letter}' in {letter_root}")
    return images


def augment_image(image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        image_array = np.asarray(image_rgb)
        mask = build_foreground_mask(image_array)
        random_background = create_random_background(image_rgb.size)
        composited = Image.composite(image_rgb, random_background, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composited.save(output_path)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    total_target = len(LETTER_CLASS_NAMES) * IMAGES_PER_LETTER

    for letter in LETTER_CLASS_NAMES:
        source_images = get_images_for_letter(SOURCE_ROOT, letter)
        letter_output_root = OUTPUT_ROOT / letter
        letter_output_root.mkdir(parents=True, exist_ok=True)

        if len(source_images) >= IMAGES_PER_LETTER:
            selected_images = random.sample(source_images, IMAGES_PER_LETTER)
        else:
            selected_images = [random.choice(source_images) for _ in range(IMAGES_PER_LETTER)]

        print(f"Generating {IMAGES_PER_LETTER} images for letter '{letter}'...")
        for letter_index, image_path in enumerate(selected_images, start=1):
            output_path = letter_output_root / f"{image_path.stem}_augmented_{letter_index:04d}.png"
            augment_image(image_path=image_path, output_path=output_path)
            total_saved += 1

            if (
                letter_index == 1
                or letter_index % 100 == 0
                or letter_index == IMAGES_PER_LETTER
            ):
                print(
                    f"  [{letter_index}/{IMAGES_PER_LETTER}] saved {output_path.name} "
                    f"| overall {total_saved}/{total_target}"
                )

    print(f"Dataset created in: {OUTPUT_ROOT.resolve()}")
    print(f"Total augmented images saved: {total_saved}")


if __name__ == "__main__":
    main()
