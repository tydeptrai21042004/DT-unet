from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_all, prepare_kvasir_seg
from src.datasets import get_dataset_spec


def _zip_tree(source: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def test_cross_domain_registry_uses_official_sources_not_kaggle():
    isic = get_dataset_spec("isic2018")
    busi = get_dataset_spec("busi")

    assert isic.official_source_url == "https://challenge.isic-archive.com/data/"
    assert isic.official_download_urls == (
        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    )
    assert busi.official_source_url == "https://scholar.cu.edu.eg/?q=afahmy/pages/dataset"
    assert busi.official_download_urls == (
        "https://scholar.cu.edu.eg/Dataset_BUSI.zip",
    )
    assert not hasattr(isic, "kaggle_handle")
    assert not hasattr(busi, "kaggle_handle")


def test_official_archive_downloader_merges_isic_archives(monkeypatch, tmp_path: Path):
    image_source = tmp_path / "image_source"
    mask_source = tmp_path / "mask_source"
    image_dir = image_source / "ISIC2018_Task1-2_Training_Input"
    mask_dir = mask_source / "ISIC2018_Task1_Training_GroundTruth"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_dir / "ISIC_0000001.jpg")
    Image.new("L", (8, 8), color=255).save(mask_dir / "ISIC_0000001_segmentation.png")

    image_zip = tmp_path / "images.zip"
    mask_zip = tmp_path / "masks.zip"
    _zip_tree(image_source, image_zip)
    _zip_tree(mask_source, mask_zip)

    mapping = {"official://images.zip": image_zip, "official://masks.zip": mask_zip}
    calls: list[str] = []

    def fake_download(url: str, dst: Path, *, verify: bool = True) -> Path:
        calls.append(url)
        shutil.copy2(mapping[url], dst)
        return dst

    monkeypatch.setattr(prepare_kvasir_seg, "_maybe_download", fake_download)
    resolved = prepare_kvasir_seg._download_official_archives(
        tuple(mapping),
        tmp_path / "downloads",
        tmp_path / "extract",
        "isic2018",
    )

    assert calls == list(mapping)
    dirs = prepare_kvasir_seg._resolve_image_mask_dirs(resolved)
    assert dirs is not None
    assert (dirs.image_dir / "ISIC_0000001.jpg").is_file()
    assert (dirs.mask_dir / "ISIC_0000001_segmentation.png").is_file()


def test_benchmark_prepare_cmd_relies_on_registry_official_sources():
    args = Namespace(
        dataset="isic2018",
        data_root="data",
        image_size=352,
        source_dir=None,
        zip_path=None,
        download_url=None,
        download_dst=None,
        allow_insecure_download=False,
    )
    cmd = benchmark_all.build_prepare_cmd(args, py="python")
    assert "--kaggle-handle" not in cmd
    assert "--download-url" not in cmd
    assert cmd[cmd.index("--dataset") + 1] == "isic2018"


def test_prepare_main_auto_downloads_isic_from_official_registry(monkeypatch, tmp_path: Path):
    downloaded = tmp_path / "downloaded_isic"
    image_dir = downloaded / "ISIC2018_Task1-2_Training_Input"
    mask_dir = downloaded / "ISIC2018_Task1_Training_GroundTruth"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(image_dir / "ISIC_0000001.jpg")
    Image.new("L", (8, 8), color=255).save(mask_dir / "ISIC_0000001_segmentation.png")

    calls: list[tuple[str, ...]] = []

    def fake_official(urls, downloads_dir, extract_dir, dataset_name, *, verify=True):
        calls.append(tuple(urls))
        return downloaded

    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "_download_official_archives", fake_official)
    monkeypatch.setattr(
        prepare_kvasir_seg,
        "parse_args",
        lambda: Namespace(
            dataset="isic2018",
            data_root=str(data_root),
            source_dir=None,
            zip_path=None,
            download_url=None,
            download_dst=None,
            image_size=32,
            skip_raw_copy=False,
            force=False,
            allow_insecure_download=False,
        ),
    )

    prepare_kvasir_seg.main()
    assert calls == [get_dataset_spec("isic2018").official_download_urls]
    assert (data_root / "processed" / "isic2018" / "images_32" / "isic_0000001.png").is_file()
    assert (data_root / "processed" / "isic2018" / "masks_32" / "isic_0000001.png").is_file()


def test_prepare_main_auto_downloads_busi_from_official_registry(monkeypatch, tmp_path: Path):
    downloaded = tmp_path / "downloaded_busi" / "Dataset_BUSI_with_GT"
    class_dir = downloaded / "benign"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(class_dir / "benign (1).png")
    Image.new("L", (8, 8), color=255).save(class_dir / "benign (1)_mask.png")

    calls: list[tuple[str, ...]] = []

    def fake_official(urls, downloads_dir, extract_dir, dataset_name, *, verify=True):
        calls.append(tuple(urls))
        return downloaded

    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "_download_official_archives", fake_official)
    monkeypatch.setattr(
        prepare_kvasir_seg,
        "parse_args",
        lambda: Namespace(
            dataset="busi",
            data_root=str(data_root),
            source_dir=None,
            zip_path=None,
            download_url=None,
            download_dst=None,
            image_size=32,
            skip_raw_copy=False,
            force=False,
            allow_insecure_download=False,
        ),
    )

    prepare_kvasir_seg.main()
    assert calls == [get_dataset_spec("busi").official_download_urls]
    assert (data_root / "processed" / "busi" / "images_32" / "benign__1.png").is_file()
    assert (data_root / "processed" / "busi" / "masks_32" / "benign__1.png").is_file()


def test_balanced_session_shell_scripts_are_valid_and_do_not_reference_kaggle_downloads():
    shared = PROJECT_ROOT / "run_hc_balanced_session.sh"
    subprocess.run(["bash", "-n", str(shared)], check=True)
    text = shared.read_text(encoding="utf-8")

    assert "proposal_hf_unet" not in text
    assert "proposal_hc_unet_no_gate" in text
    assert "isic2018" in text
    assert "busi" in text
    assert "official archives" in text
    assert "kagglehub" not in text.lower()
    assert "kaggle-handle" not in text.lower()

    for session in range(1, 5):
        wrapper = PROJECT_ROOT / f"run_hc_session_{session}.sh"
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        assert f"run_hc_balanced_session.sh\" {session}" in wrapper.read_text(encoding="utf-8")


def test_repository_no_longer_depends_on_kagglehub():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "kagglehub" not in requirements
    for relative in (
        "scripts/prepare_kvasir_seg.py",
        "scripts/benchmark_all.py",
        "scripts/benchmark_multi_seed.py",
        "scripts/run_hc_ablation.py",
        "run.sh",
        "run_hc_ablation.sh",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8").lower()
        assert "kagglehub" not in text
        assert "kaggle-handle" not in text
