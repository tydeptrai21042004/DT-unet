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
    "kvasir_instrument": DatasetSpec(
        name="kvasir_instrument",
        aliases=("kvasir_instrument", "kvasir-instrument", "instrument", "kvasir_tool"),
        canonical_dir="Kvasir-Instrument",
        default_download_url="https://datasets.simula.no/downloads/kvasir-instrument.zip",
        official_source_url="https://datasets.simula.no/kvasir-instrument/",
        description="Kvasir-Instrument gastrointestinal endoscopic tool binary segmentation dataset.",
    ),
    "hyper_kvasir_seg": DatasetSpec(
        name="hyper_kvasir_seg",
        aliases=("hyper_kvasir_seg", "hyper-kvasir-seg", "hyper_kvasir_segmentation", "hyper-kvasir-segmentation"),
        canonical_dir="HyperKvasir-SEG",
        default_download_url="https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip",
        official_source_url="https://datasets.simula.no/hyper-kvasir/",
        description="The segmentation-only HyperKvasir subset; the full 58.6 GB archive is intentionally not downloaded.",
    ),
    "montgomery_lung": DatasetSpec(
        name="montgomery_lung",
        aliases=("montgomery_lung", "montgomery-lung", "montgomery", "montgomery_cxr", "montgomery-cxr"),
        canonical_dir="Montgomery-Lung",
        default_download_url=None,
        official_source_url="https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/",
        description="NLM Montgomery County chest X-ray lung-field segmentation dataset; left and right masks are merged automatically.",
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
