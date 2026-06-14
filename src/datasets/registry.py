from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: Tuple[str, ...]
    canonical_dir: str
    default_download_url: Optional[str] = None
    official_download_urls: Tuple[str, ...] = ()
    official_source_url: Optional[str] = None
    description: str = ""


DATASET_SPECS = {
    "kvasir_seg": DatasetSpec(
        name="kvasir_seg",
        aliases=("kvasir_seg", "kvasir-seg", "kvasir"),
        canonical_dir="Kvasir-SEG",
        default_download_url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        description="Kvasir-SEG polyp segmentation dataset.",
    ),
    "cvc_clinicdb": DatasetSpec(
        name="cvc_clinicdb",
        aliases=("cvc_clinicdb", "cvc-clinicdb", "clinicdb", "cvc612", "cvc-612"),
        canonical_dir="CVC-ClinicDB",
        # PraNet public training bundle: TrainDataset.zip contains Kvasir-SEG and CVC-ClinicDB.
        # The preparation script searches inside the archive and extracts only the requested dataset.
        default_download_url="https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing",
        description="CVC-ClinicDB polyp segmentation dataset.",
    ),
    "etis": DatasetSpec(
        name="etis",
        aliases=("etis", "etis-larib", "etis_larib", "etis-laribpolypdb", "etis_laribpolypdb"),
        canonical_dir="ETIS-LaribPolypDB",
        # PraNet public testing bundle: TestDataset.zip contains ETIS-LaribPolypDB,
        # CVC-ColonDB, CVC-300, CVC-ClinicDB, and Kvasir test subsets.
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="ETIS-LaribPolypDB polyp segmentation dataset.",
    ),
    "cvc_colondb": DatasetSpec(
        name="cvc_colondb",
        aliases=("cvc_colondb", "cvc-colondb", "colondb", "cvc-colon"),
        canonical_dir="CVC-ColonDB",
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="CVC-ColonDB polyp segmentation dataset.",
    ),
    "cvc_300": DatasetSpec(
        name="cvc_300",
        aliases=("cvc_300", "cvc-300", "cvc300"),
        canonical_dir="CVC-300",
        default_download_url="https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing",
        description="CVC-300 polyp segmentation dataset.",
    ),

    "isic2018": DatasetSpec(
        name="isic2018",
        aliases=("isic2018", "isic-2018", "isic", "isic_task1", "isic2018_task1", "isic_2018_task_1"),
        canonical_dir="ISIC2018",
        default_download_url=None,
        official_download_urls=(
            "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
            "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        ),
        official_source_url="https://challenge.isic-archive.com/data/",
        description="ISIC 2018 Task 1 binary skin-lesion boundary segmentation dataset. Automatically downloaded from the official ISIC Challenge archive.",
    ),
    "busi": DatasetSpec(
        name="busi",
        aliases=("busi", "dataset_busi", "dataset_busi_with_gt", "breast_ultrasound", "breast-ultrasound", "breast_ultrasound_images_dataset"),
        canonical_dir="Dataset_BUSI_with_GT",
        default_download_url=None,
        official_download_urls=(
            "https://scholar.cu.edu.eg/Dataset_BUSI.zip",
        ),
        official_source_url="https://scholar.cu.edu.eg/?q=afahmy/pages/dataset",
        description="BUSI breast ultrasound lesion segmentation dataset. Automatically downloaded from the official Cairo University dataset page; the generic loader pairs *_mask files with corresponding images.",
    ),
    "drive": DatasetSpec(
        name="drive",
        aliases=("drive", "drive_db", "drive-db", "digital_retinal_images_for_vessel_extraction", "retinal_drive"),
        canonical_dir="DRIVE",
        default_download_url=None,
        description="DRIVE retinal vessel segmentation dataset. The generic loader pairs *_manual masks with corresponding fundus images.",
    ),
    "custom": DatasetSpec(
        name="custom",
        aliases=("custom", "custom_binary_seg", "custom_segmentation"),
        canonical_dir="custom",
        default_download_url=None,
        description="Custom binary segmentation dataset with matching images/masks and split files.",
    ),
}

_ALIAS_TO_NAME: Dict[str, str] = {}
for _name, _spec in DATASET_SPECS.items():
    for _alias in _spec.aliases:
        _ALIAS_TO_NAME[_alias.lower()] = _name

SUPPORTED_DATASETS = tuple(sorted(DATASET_SPECS))


def normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return "kvasir_seg"
    value = str(name).strip().lower().replace(" ", "_")
    if not value:
        return "kvasir_seg"
    try:
        return _ALIAS_TO_NAME[value]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported datasets: {supported}") from exc



def get_dataset_spec(name: Optional[str] = None) -> DatasetSpec:
    return DATASET_SPECS[normalize_dataset_name(name)]


__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "SUPPORTED_DATASETS",
    "normalize_dataset_name",
    "get_dataset_spec",
]
