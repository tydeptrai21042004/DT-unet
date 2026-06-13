#!/usr/bin/env python3
"""Prepare supported polyp datasets for the HF-U-Net benchmark.

Supported input modes:
1) Existing extracted dataset folder with images/ and masks/
2) Zip archive containing a supported dataset
3) Optional direct download URL (including Google Drive links via gdown if installed)
4) Automatic default download when a direct URL is configured in the dataset registry
5) Automatic KaggleHub download when a Kaggle dataset handle is configured

Outputs a benchmark-friendly layout:
    data/
      raw/<dataset>/images
      raw/<dataset>/masks
      processed/images_<size>
      processed/masks_<size>
      processed/metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import get_dataset_spec, normalize_dataset_name
from src.datasets.kvasir_seg_dataset import _dir_name_variants, _resolve_image_mask_dirs, canonical_sample_id, looks_like_mask_stem

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def _has_image_mask_dirs(path: Path) -> bool:
    return _resolve_image_mask_dirs(path) is not None



def _find_dataset_root(root: Path, dataset_name: str) -> Optional[Path]:
    normalized = normalize_dataset_name(dataset_name)
    if _has_image_mask_dirs(root):
        return root

    for token in _dir_name_variants(normalized):
        for prefix in (
            root,
            root / "raw",
            root / "dataset",
            root / "datasets",
            root / "TrainDataset",
            root / "TestDataset",
            root / "train",
            root / "test",
        ):
            candidate = prefix / token
            if _has_image_mask_dirs(candidate):
                return candidate

    keywords = {token.lower() for token in _dir_name_variants(normalized)}
    for cand in root.rglob("*"):
        if not cand.is_dir() or not _has_image_mask_dirs(cand):
            continue
        path_text = cand.as_posix().lower()
        if normalized == "custom" or any(keyword in path_text for keyword in keywords):
            return cand
    return None



def _is_image(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS



def _iter_images(directory: Path) -> List[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and _is_image(p))


def _collect_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[str, Path, Path]]:
    image_map = {}
    mask_map = {}
    for image_path in _iter_images(image_dir):
        key = canonical_sample_id(image_path.stem)
        # When image_dir == mask_dir, skip mask-like files from the image map.
        if image_dir == mask_dir and looks_like_mask_stem(image_path.stem):
            continue
        image_map.setdefault(key, image_path)
    for mask_path in _iter_images(mask_dir):
        key = canonical_sample_id(mask_path.stem)
        # In separate mask folders, exact-stem masks are valid; in shared BUSI-like
        # folders, require a mask-like suffix so original images are not treated as masks.
        if image_dir == mask_dir and not looks_like_mask_stem(mask_path.stem):
            continue
        mask_map.setdefault(key, mask_path)

    pairs: List[Tuple[str, Path, Path]] = []
    missing: List[str] = []
    for sample_id, image_path in sorted(image_map.items()):
        mask_path = mask_map.get(sample_id)
        if mask_path is None:
            missing.append(sample_id)
            continue
        pairs.append((sample_id, image_path, mask_path))
    if not pairs:
        raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
    if missing:
        print(f"[WARN] Missing masks for {len(missing)} images. They will be skipped.", file=sys.stderr)
    return pairs



def _copy_raw_pairs(pairs: Sequence[Tuple[str, Path, Path]], raw_images: Path, raw_masks: Path) -> None:
    raw_images.mkdir(parents=True, exist_ok=True)
    raw_masks.mkdir(parents=True, exist_ok=True)
    for sample_id, image_path, mask_path in pairs:
        shutil.copy2(image_path, raw_images / f"{sample_id}{image_path.suffix.lower()}")
        shutil.copy2(mask_path, raw_masks / f"{sample_id}{mask_path.suffix.lower()}")



def _resize_pair(image_path: Path, mask_path: Path, out_image: Path, out_mask: Path, size: int) -> Tuple[int, int]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    orig_h, orig_w = image.height, image.width

    image_resized = image.resize((size, size), resample=Image.BILINEAR)
    mask_resized = mask.resize((size, size), resample=Image.NEAREST)
    mask_binary = mask_resized.point(lambda x: 255 if x > 127 else 0)

    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    image_resized.save(out_image)
    mask_binary.save(out_mask)
    return orig_h, orig_w



def _write_metadata(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _extract_zip(zip_path: Path, extract_dir: Path, dataset_name: str) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    dataset_root = _find_dataset_root(extract_dir, dataset_name)
    if dataset_root is None:
        spec = get_dataset_spec(dataset_name)
        raise FileNotFoundError(
            f"Could not find extracted dataset={spec.name} under {extract_dir}. Expected images/ and masks/ folders."
        )
    return dataset_root



def _resolve_kaggle_dataset_root(
    kaggle_root: Path,
    extract_dir: Path,
    dataset_name: str,
) -> Optional[Path]:
    """Locate a dataset inside a KaggleHub download, extracting nested ZIPs if needed."""
    dataset_root = _find_dataset_root(kaggle_root, dataset_name)
    if dataset_root is not None:
        return dataset_root

    zip_files = sorted(path for path in kaggle_root.rglob("*.zip") if path.is_file())
    if not zip_files:
        return None

    extract_dir.mkdir(parents=True, exist_ok=True)
    for index, zip_path in enumerate(zip_files):
        destination = extract_dir / f"{index:02d}_{zip_path.stem}"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(destination)

    return _find_dataset_root(extract_dir, dataset_name)


def _download_kaggle_dataset(handle: str) -> Path:
    """Download a public Kaggle dataset and return its local directory.

    Kaggle notebooks authenticate KaggleHub automatically. Outside Kaggle, the
    usual Kaggle credentials/environment configuration may be required.
    """
    try:
        import kagglehub
    except Exception as exc:  # pragma: no cover - exercised through dependency failure only
        raise RuntimeError(
            "Kaggle dataset download requested but kagglehub is not installed. "
            "Install kagglehub or pass --source-dir/--zip-path/--download-url."
        ) from exc

    result = kagglehub.dataset_download(handle)
    if not result:
        raise RuntimeError(f"KaggleHub returned an empty path for dataset: {handle}")

    dataset_path = Path(result).expanduser().resolve()
    if not dataset_path.is_dir():
        raise RuntimeError(
            f"KaggleHub download path is not a directory for dataset={handle}: {dataset_path}"
        )
    return dataset_path


def _maybe_download(url: str, dst: Path, *, verify: bool = True) -> Path:
    if "drive.google.com" in url:
        try:
            import gdown
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Google Drive download requested but gdown is not installed. Install gdown or pass --zip-path/--source-dir."
            ) from exc
        dst.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(url=url, output=str(dst), quiet=False, fuzzy=True)
        if not result:
            raise RuntimeError(f"Failed to download Google Drive URL: {url}")
        return dst

    try:
        import requests
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests is required for download mode. Install it or pass --zip-path/--source-dir.") from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, verify=verify) as response:
        response.raise_for_status()
        with dst.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst



def prepared_dataset_exists(data_root: Path, image_size: int, dataset_name: str = "kvasir_seg") -> bool:
    dataset_name = normalize_dataset_name(dataset_name)
    legacy = (data_root / "processed" / f"images_{image_size}").is_dir() and (data_root / "processed" / f"masks_{image_size}").is_dir()
    dataset_specific = (
        (data_root / "processed" / dataset_name / f"images_{image_size}").is_dir()
        and (data_root / "processed" / dataset_name / f"masks_{image_size}").is_dir()
    )
    return bool(dataset_specific or legacy)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a supported binary segmentation dataset for benchmark training.")
    parser.add_argument("--dataset", type=str, default="kvasir_seg", help="Dataset key. Built-in datasets may use a direct URL or KaggleHub handle; custom datasets require an existing local image/mask layout.")
    parser.add_argument("--data-root", type=str, default="data", help="Benchmark data root.")
    parser.add_argument("--source-dir", type=str, default=None, help="Path to an extracted dataset folder or its parent.")
    parser.add_argument("--zip-path", type=str, default=None, help="Path to a dataset zip archive.")
    parser.add_argument("--download-url", type=str, default=None, help="Optional URL to download a zip archive.")
    parser.add_argument("--download-dst", type=str, default=None, help="Optional destination path for the downloaded zip.")
    parser.add_argument("--kaggle-handle", type=str, default=None, help="Optional Kaggle dataset handle, for example owner/dataset. Overrides the registry handle.")
    parser.add_argument("--image-size", type=int, default=352, help="Output square size for processed images/masks.")
    parser.add_argument("--skip-raw-copy", action="store_true", help="Do not copy files into data/raw/<dataset>.")
    parser.add_argument("--force", action="store_true", help="Rebuild processed outputs even if they already exist.")
    parser.add_argument("--allow-insecure-download", action="store_true", help="Disable TLS certificate verification for dataset download.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    if dataset_name == "custom":
        raise ValueError("Custom datasets are not auto-preparable. Use your own images/masks layout and split files.")

    spec = get_dataset_spec(dataset_name)
    data_root = Path(args.data_root)
    raw_root = data_root / "raw" / spec.canonical_dir
    processed_root = data_root / "processed" / dataset_name
    processed_images = processed_root / f"images_{args.image_size}"
    processed_masks = processed_root / f"masks_{args.image_size}"

    if prepared_dataset_exists(data_root, args.image_size, dataset_name=dataset_name) and not args.force:
        print(f"Dataset already prepared for image_size={args.image_size} at: {processed_root}")
        print(f"Processed images: {processed_images}")
        print(f"Processed masks : {processed_masks}")
        return

    dataset_root: Optional[Path] = None

    if args.source_dir:
        dataset_root = _find_dataset_root(Path(args.source_dir), dataset_name)
        if dataset_root is None:
            raise FileNotFoundError(f"Could not locate images/ and masks/ under source-dir: {args.source_dir}")
    elif args.zip_path:
        dataset_root = _extract_zip(Path(args.zip_path), data_root / "_tmp_extract", dataset_name)
    else:
        explicit_download_url = args.download_url
        explicit_kaggle_handle = args.kaggle_handle
        existing_root = _find_dataset_root(data_root, dataset_name)

        if existing_root is not None:
            dataset_root = existing_root
        elif explicit_download_url:
            download_dst = Path(args.download_dst) if args.download_dst else data_root / "downloads" / f"{dataset_name}.zip"
            print(f"Downloading {dataset_name} from explicit URL: {explicit_download_url}")
            allow_insecure = bool(
                args.allow_insecure_download
                or os.environ.get("ALLOW_INSECURE_DOWNLOAD", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
            )
            zip_path = _maybe_download(explicit_download_url, download_dst, verify=not allow_insecure)
            dataset_root = _extract_zip(zip_path, data_root / "_tmp_extract", dataset_name)
        elif explicit_kaggle_handle:
            print(f"Downloading {dataset_name} from Kaggle: {explicit_kaggle_handle}")
            kaggle_root = _download_kaggle_dataset(explicit_kaggle_handle)
            dataset_root = _resolve_kaggle_dataset_root(
                kaggle_root,
                data_root / "_tmp_kaggle_extract" / dataset_name,
                dataset_name,
            )
            if dataset_root is None:
                raise FileNotFoundError(
                    f"Kaggle dataset {explicit_kaggle_handle} was downloaded to {kaggle_root}, "
                    f"but no compatible image/mask layout was found for dataset={dataset_name}."
                )
        elif spec.default_download_url:
            download_dst = Path(args.download_dst) if args.download_dst else data_root / "downloads" / f"{dataset_name}.zip"
            print(f"Downloading {dataset_name} from registry URL: {spec.default_download_url}")
            allow_insecure = bool(
                args.allow_insecure_download
                or os.environ.get("ALLOW_INSECURE_DOWNLOAD", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
            )
            zip_path = _maybe_download(spec.default_download_url, download_dst, verify=not allow_insecure)
            dataset_root = _extract_zip(zip_path, data_root / "_tmp_extract", dataset_name)
        elif spec.kaggle_handle:
            print(f"Downloading {dataset_name} from registry Kaggle handle: {spec.kaggle_handle}")
            kaggle_root = _download_kaggle_dataset(spec.kaggle_handle)
            dataset_root = _resolve_kaggle_dataset_root(
                kaggle_root,
                data_root / "_tmp_kaggle_extract" / dataset_name,
                dataset_name,
            )
            if dataset_root is None:
                raise FileNotFoundError(
                    f"Kaggle dataset {spec.kaggle_handle} was downloaded to {kaggle_root}, "
                    f"but no compatible image/mask layout was found for dataset={dataset_name}."
                )
        else:
            raise ValueError(
                f"No automatic download source is configured for dataset={dataset_name}. "
                "Use --source-dir, --zip-path, --download-url, or --kaggle-handle."
            )

    resolved_dirs = _resolve_image_mask_dirs(dataset_root)
    if resolved_dirs is None:
        raise FileNotFoundError(f"Could not resolve compatible image/mask folders inside dataset root: {dataset_root}")
    image_dir = resolved_dirs.image_dir
    mask_dir = resolved_dirs.mask_dir
    pairs = _collect_pairs(image_dir, mask_dir)

    if not args.skip_raw_copy:
        _copy_raw_pairs(pairs, raw_root / "images", raw_root / "masks")

    metadata_rows: List[dict] = []
    for sample_id, image_path, mask_path in pairs:
        out_image = processed_images / f"{sample_id}.png"
        out_mask = processed_masks / f"{sample_id}.png"
        orig_h, orig_w = _resize_pair(image_path, mask_path, out_image, out_mask, args.image_size)
        metadata_rows.append(
            {
                "id": sample_id,
                "dataset": dataset_name,
                "image_path": str(out_image.as_posix()),
                "mask_path": str(out_mask.as_posix()),
                "orig_height": orig_h,
                "orig_width": orig_w,
                "proc_height": args.image_size,
                "proc_width": args.image_size,
            }
        )

    _write_metadata(metadata_rows, processed_root / "metadata.csv")
    print(f"Prepared {len(metadata_rows)} samples for dataset={dataset_name} at: {processed_root}")
    print(f"Processed images: {processed_images}")
    print(f"Processed masks : {processed_masks}")


if __name__ == "__main__":
    main()
