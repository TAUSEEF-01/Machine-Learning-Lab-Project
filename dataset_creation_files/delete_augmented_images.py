import argparse
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Matches filenames like: originalname_<suffix>.jpg
TARGET_SUFFIXES = {
    "flip_horizontal",
    "flip_vertical",
    "translate_down",
    "translate_right",
    "center_crop_zoom",
    "cutout",
    "invert",
    "rotate_15",
    "rotate_30",
    "rotate_45",
    "shear_horizontal_15deg",
    "shear_vertical_15deg",
    "solarize",
}


def should_delete(path: Path, suffixes: set[str]) -> bool:
    stem = path.stem
    return any(stem.endswith(f"_{suffix}") for suffix in suffixes)


def collect_targets(root: Path, suffixes: set[str]) -> list[Path]:
    targets: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if should_delete(file_path, suffixes):
            targets.append(file_path)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete augmented images based on filename suffixes"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Downloads\DUCSE Documents\4-1\Mine\CSE-4125 Introduction to Machine Learning\Lab\Final Project\Augmented_final train image",
        help="Root folder containing augmented class folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting",
    )
    parser.add_argument(
        "--extra-suffix",
        action="append",
        default=[],
        help="Additional suffix to delete (use multiple times)",
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root folder not found: {root}")

    suffixes = set(TARGET_SUFFIXES)
    suffixes.update(s.strip() for s in args.extra_suffix if s.strip())

    targets = collect_targets(root, suffixes)

    if args.dry_run:
        print(f"Dry run: {len(targets)} files would be deleted.")
        for path in targets[:25]:
            print(path)
        if len(targets) > 25:
            print(f"... and {len(targets) - 25} more")
        return

    deleted = 0
    for path in targets:
        path.unlink(missing_ok=True)
        deleted += 1

    print(f"Deleted {deleted} files from: {root}")


if __name__ == "__main__":
    main()
