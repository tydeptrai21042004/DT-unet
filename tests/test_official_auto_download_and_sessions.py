from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_kvasir_seg
from src.datasets import DATASET_SPECS, get_dataset_spec


def test_removed_datasets_are_not_registered():
    assert "busi" not in DATASET_SPECS
    assert "drive" not in DATASET_SPECS
    assert "custom" not in DATASET_SPECS


def test_new_registry_uses_public_institutional_sources():
    instrument = get_dataset_spec("kvasir_instrument")
    hyper = get_dataset_spec("hyper_kvasir_seg")
    montgomery = get_dataset_spec("montgomery_lung")
    assert instrument.default_download_url == "https://datasets.simula.no/downloads/kvasir-instrument.zip"
    assert hyper.default_download_url.endswith("hyper-kvasir-segmented-images.zip")
    assert "lhncbc.nlm.nih.gov" in montgomery.official_source_url
    assert "58.6 GB" in hyper.description


def test_prepare_main_auto_downloads_kvasir_instrument(monkeypatch, tmp_path: Path):
    extracted = tmp_path / "Kvasir-Instrument"
    images = extracted / "images"
    masks = extracted / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(images / "tool.jpg")
    Image.new("L", (8, 8), color=255).save(masks / "tool.png")

    monkeypatch.setattr(prepare_kvasir_seg, "_maybe_download", lambda *a, **k: tmp_path / "instrument.zip")
    monkeypatch.setattr(prepare_kvasir_seg, "_extract_zip", lambda *a, **k: extracted)
    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "parse_args", lambda: Namespace(
        dataset="kvasir_instrument", data_root=str(data_root), source_dir=None,
        zip_path=None, download_url=None, download_dst=None, image_size=32,
        skip_raw_copy=False, force=False, allow_insecure_download=False,
    ))
    prepare_kvasir_seg.main()
    assert (data_root / "processed/kvasir_instrument/images_32/tool.png").is_file()
    assert (data_root / "processed/kvasir_instrument/masks_32/tool.png").is_file()


def test_montgomery_left_and_right_masks_are_merged(monkeypatch, tmp_path: Path):
    def fake_index(url: str, destination: Path, *, verify=True):
        destination.mkdir(parents=True, exist_ok=True)
        if destination.name == "CXR_png":
            Image.new("L", (8, 8), 100).save(destination / "MCUCXR_0001_0.png")
        elif destination.name == "leftMask":
            im = Image.new("L", (8, 8), 0); im.putpixel((1, 1), 255); im.save(destination / "MCUCXR_0001_0.png")
        else:
            im = Image.new("L", (8, 8), 0); im.putpixel((6, 6), 255); im.save(destination / "MCUCXR_0001_0.png")
    monkeypatch.setattr(prepare_kvasir_seg, "_download_index_files", fake_index)
    root = prepare_kvasir_seg._prepare_montgomery_official(tmp_path)
    merged = Image.open(root / "masks/MCUCXR_0001_0.png").convert("L")
    assert merged.getpixel((1, 1)) == 255
    assert merged.getpixel((6, 6)) == 255


def test_balanced_sessions_use_new_datasets_and_are_valid():
    shared = PROJECT_ROOT / "run_hc_balanced_session.sh"
    subprocess.run(["bash", "-n", str(shared)], check=True)
    text = shared.read_text(encoding="utf-8")
    assert 'NEW_DATASET="kvasir_instrument"' in text
    assert 'NEW_DATASET="montgomery_lung"' in text
    assert 'EXISTING_DATASET="hyper_kvasir_seg"' in text
    assert "busi" not in text.lower()
    assert "drive" not in text.lower()
    for session in range(1, 5):
        subprocess.run(["bash", "-n", str(PROJECT_ROOT / f"run_hc_session_{session}.sh")], check=True)
