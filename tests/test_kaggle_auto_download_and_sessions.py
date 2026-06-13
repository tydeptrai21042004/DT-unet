from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import benchmark_all, prepare_kvasir_seg
from src.datasets import get_dataset_spec


def test_cross_domain_registry_has_public_kaggle_handles():
    assert get_dataset_spec("isic2018").kaggle_handle == (
        "tschandl/isic2018-challenge-task1-data-segmentation"
    )
    assert get_dataset_spec("busi").kaggle_handle == (
        "sabahesaraki/breast-ultrasound-images-dataset"
    )


def test_kagglehub_download_helper_returns_directory(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_download(handle: str) -> str:
        calls.append(handle)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(dataset_download=fake_download),
    )

    result = prepare_kvasir_seg._download_kaggle_dataset("owner/dataset")
    assert result == tmp_path.resolve()
    assert calls == ["owner/dataset"]


def test_benchmark_prepare_cmd_injects_isic_registry_kaggle_handle():
    args = Namespace(
        dataset="isic2018",
        data_root="data",
        image_size=352,
        source_dir=None,
        zip_path=None,
        download_url=None,
        download_dst=None,
        kaggle_handle=None,
    )
    cmd = benchmark_all.build_prepare_cmd(args, py="python")
    assert "--kaggle-handle" in cmd
    assert get_dataset_spec("isic2018").kaggle_handle in cmd


def test_explicit_kaggle_handle_overrides_registry_handle():
    args = Namespace(
        dataset="busi",
        data_root="data",
        image_size=352,
        source_dir=None,
        zip_path=None,
        download_url=None,
        download_dst=None,
        kaggle_handle="custom-owner/custom-busi",
    )
    cmd = benchmark_all.build_prepare_cmd(args, py="python")
    index = cmd.index("--kaggle-handle")
    assert cmd[index + 1] == "custom-owner/custom-busi"


def test_balanced_session_shell_scripts_are_valid_and_do_not_run_hf_proposal():
    shared = PROJECT_ROOT / "run_hc_balanced_session.sh"
    subprocess.run(["bash", "-n", str(shared)], check=True)
    text = shared.read_text(encoding="utf-8")

    assert "proposal_hf_unet" not in text
    assert "proposal_hc_unet_no_gate" in text
    assert "isic2018" in text
    assert "busi" in text
    assert "hc_reference" in text
    assert "hc_without_hc_branch" in text
    assert "hc_shared_kernel" in text
    assert "hc_learnable_h" in text
    assert "hc_kernel5" in text
    assert "hc_identity_projection" in text
    assert "hc_no_channel_expansion" in text

    for session in range(1, 5):
        wrapper = PROJECT_ROOT / f"run_hc_session_{session}.sh"
        subprocess.run(["bash", "-n", str(wrapper)], check=True)
        assert f"run_hc_balanced_session.sh\" {session}" in wrapper.read_text(
            encoding="utf-8"
        )


def test_prepare_main_auto_downloads_isic_from_registry(monkeypatch, tmp_path: Path):
    from PIL import Image

    downloaded = tmp_path / "downloaded_isic"
    image_dir = downloaded / "ISIC2018_Task1-2_Training_Input"
    mask_dir = downloaded / "ISIC2018_Task1_Training_GroundTruth"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(
        image_dir / "ISIC_0000001.jpg"
    )
    Image.new("L", (8, 8), color=255).save(
        mask_dir / "ISIC_0000001_segmentation.png"
    )

    calls: list[str] = []

    def fake_download(handle: str) -> Path:
        calls.append(handle)
        return downloaded

    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "_download_kaggle_dataset", fake_download)
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
            kaggle_handle=None,
            image_size=32,
            skip_raw_copy=False,
            force=False,
            allow_insecure_download=False,
        ),
    )

    prepare_kvasir_seg.main()

    assert calls == [get_dataset_spec("isic2018").kaggle_handle]
    assert (data_root / "processed" / "isic2018" / "images_32" / "isic_0000001.png").is_file()
    assert (data_root / "processed" / "isic2018" / "masks_32" / "isic_0000001.png").is_file()


def test_prepare_main_auto_downloads_busi_from_registry(monkeypatch, tmp_path: Path):
    from PIL import Image

    downloaded = tmp_path / "downloaded_busi" / "Dataset_BUSI_with_GT"
    class_dir = downloaded / "benign"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(
        class_dir / "benign (1).png"
    )
    Image.new("L", (8, 8), color=255).save(
        class_dir / "benign (1)_mask.png"
    )

    calls: list[str] = []

    def fake_download(handle: str) -> Path:
        calls.append(handle)
        return downloaded.parent

    data_root = tmp_path / "data"
    monkeypatch.setattr(prepare_kvasir_seg, "_download_kaggle_dataset", fake_download)
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
            kaggle_handle=None,
            image_size=32,
            skip_raw_copy=False,
            force=False,
            allow_insecure_download=False,
        ),
    )

    prepare_kvasir_seg.main()

    assert calls == [get_dataset_spec("busi").kaggle_handle]
    assert (data_root / "processed" / "busi" / "images_32" / "benign__1.png").is_file()
    assert (data_root / "processed" / "busi" / "masks_32" / "benign__1.png").is_file()


def test_kaggle_root_resolver_extracts_nested_zip(tmp_path: Path):
    import zipfile
    from PIL import Image

    source = tmp_path / "source"
    image_dir = source / "ISIC2018_Task1-2_Training_Input"
    mask_dir = source / "ISIC2018_Task1_Training_GroundTruth"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_dir / "ISIC_0000001.jpg")
    Image.new("L", (8, 8), color=255).save(
        mask_dir / "ISIC_0000001_segmentation.png"
    )

    kaggle_root = tmp_path / "kaggle"
    kaggle_root.mkdir()
    archive_path = kaggle_root / "isic_payload.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file_path in source.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source))

    resolved = prepare_kvasir_seg._resolve_kaggle_dataset_root(
        kaggle_root,
        tmp_path / "extracted",
        "isic2018",
    )
    assert resolved is not None
    assert resolved.name == "extracted" or "isic" in resolved.as_posix().lower()
    pair = prepare_kvasir_seg._resolve_image_mask_dirs(resolved)
    assert pair is not None
