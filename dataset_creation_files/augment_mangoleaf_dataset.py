import argparse
import random
from pathlib import Path

import numpy as np
import math
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _rotate(img: Image.Image, degrees: int) -> Image.Image:
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def _translate(img: Image.Image, x_shift: int, y_shift: int) -> Image.Image:
    return ImageChops.offset(img, x_shift, y_shift)


def _center_crop_resize(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = img.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def _shear_x(img: Image.Image, shear_factor: float) -> Image.Image:
    w, h = img.size
    matrix = (1, shear_factor, -shear_factor * w / 2, 0, 1, 0)
    return img.transform((w, h), Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def _shear_y(img: Image.Image, shear_factor: float) -> Image.Image:
    w, h = img.size
    matrix = (1, 0, 0, shear_factor, 1, -shear_factor * h / 2)
    return img.transform((w, h), Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def _shear_factor_from_degrees(degrees: float) -> float:
    return math.tan(math.radians(degrees))


def _gaussian_noise(img: Image.Image, std_dev: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std_dev, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _salt_pepper_noise(img: Image.Image, amount: float = 0.02, salt_vs_pepper: float = 0.5) -> Image.Image:
    arr = np.array(img)
    out = arr.copy()

    total_pixels = arr.shape[0] * arr.shape[1]
    num_salt = int(total_pixels * amount * salt_vs_pepper)
    num_pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))

    salt_coords_y = np.random.randint(0, arr.shape[0], num_salt)
    salt_coords_x = np.random.randint(0, arr.shape[1], num_salt)
    pepper_coords_y = np.random.randint(0, arr.shape[0], num_pepper)
    pepper_coords_x = np.random.randint(0, arr.shape[1], num_pepper)

    out[salt_coords_y, salt_coords_x] = 255
    out[pepper_coords_y, pepper_coords_x] = 0

    return Image.fromarray(out)


def _cutout(img: Image.Image, ratio: float = 0.2) -> Image.Image:
    w, h = img.size
    cut_w = int(w * ratio)
    cut_h = int(h * ratio)

    x1 = random.randint(0, max(0, w - cut_w))
    y1 = random.randint(0, max(0, h - cut_h))
    x2 = x1 + cut_w
    y2 = y1 + cut_h

    arr = np.array(img)
    arr[y1:y2, x1:x2] = 0
    return Image.fromarray(arr)


def get_augmentations():
    shear_15 = _shear_factor_from_degrees(15)
    return {
        # "flip_horizontal": lambda im: ImageOps.mirror(im),
        # "flip_vertical": lambda im: ImageOps.flip(im),
        # "rotate_15": lambda im: _rotate(im, 15),
        # "rotate_30": lambda im: _rotate(im, 30),
        # "rotate_45": lambda im: _rotate(im, 45),
        # "translate_right": lambda im: _translate(im, int(im.size[0] * 0.1), 0),
        # "translate_down": lambda im: _translate(im, 0, int(im.size[1] * 0.1)),
        # "center_crop_zoom": lambda im: _center_crop_resize(im, 0.85),
        # "shear_horizontal_15deg": lambda im: _shear_x(im, shear_15),
        # "shear_vertical_15deg": lambda im: _shear_y(im, shear_15),
        "brightness_up": lambda im: ImageEnhance.Brightness(im).enhance(1.35),#
        "brightness_down": lambda im: ImageEnhance.Brightness(im).enhance(0.7),#
        "contrast_up": lambda im: ImageEnhance.Contrast(im).enhance(1.4),#
        "contrast_down": lambda im: ImageEnhance.Contrast(im).enhance(0.75),#
        "color_up": lambda im: ImageEnhance.Color(im).enhance(1.35),#
        "color_down": lambda im: ImageEnhance.Color(im).enhance(0.7),#
        "sharpness_up": lambda im: ImageEnhance.Sharpness(im).enhance(2.0),#
        "gaussian_blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.5)),#
        "median_blur": lambda im: im.filter(ImageFilter.MedianFilter(size=3)),#
        "edge_enhance": lambda im: im.filter(ImageFilter.EDGE_ENHANCE_MORE),#
        "autocontrast": lambda im: ImageOps.autocontrast(im),#
        "equalize": lambda im: ImageOps.equalize(im),#
        "posterize": lambda im: ImageOps.posterize(im, bits=4),#
        # "solarize": lambda im: ImageOps.solarize(im, threshold=128),
        # "invert": lambda im: ImageOps.invert(im),
        "grayscale_3ch": lambda im: ImageOps.grayscale(im).convert("RGB"),#
        "gaussian_noise": lambda im: _gaussian_noise(im, std_dev=14.0),#
        "salt_pepper_noise": lambda im: _salt_pepper_noise(im, amount=0.02),#
        # "cutout": lambda im: _cutout(im, ratio=0.2),
    }


def iter_images(root: Path):
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if class_dirs:
        for cls_dir in class_dirs:
            for img_path in cls_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield cls_dir.name, img_path
        return

    for img_path in root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root.name, img_path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_augmentation(input_dir: Path, output_dir: Path):
    augmentations = get_augmentations()

    image_count = 0
    for class_name, img_path in iter_images(input_dir):
        image_count += 1
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            
            for aug_name, aug_fn in augmentations.items():
                out_class_dir = output_dir / class_name
                ensure_dir(out_class_dir)

                augmented = aug_fn(im)
                
                # Save with the augmentation name appended to prevent name collision
                out_name = f"{img_path.stem}_{aug_name}{img_path.suffix}"
                
                # Check format based on original extension (default to JPEG if saving fails)
                save_kwargs = {}
                if img_path.suffix.lower() in {'.jpg', '.jpeg'}:
                    save_kwargs['quality'] = 95
                    
                augmented.save(out_class_dir / out_name, **save_kwargs)

        if image_count % 100 == 0:
            print(f"Processed {image_count} images...")

    print("--------------------------------")
    print(f"Total source images processed: {image_count}")
    print(f"Total augmentation types: {len(augmentations)}")
    print(f"Expected augmented images: {image_count * len(augmentations)}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Apply comprehensive data augmentation to MangoLeafBD dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="final train image",
        help="Path to original dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="augmented",
        help="Path to save augmented dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_dir(output_dir)
    run_augmentation(input_dir, output_dir)


if __name__ == "__main__":
    main()
