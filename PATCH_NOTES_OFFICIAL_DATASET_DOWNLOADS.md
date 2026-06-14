# Official dataset download update

- Removed KaggleHub as the automatic source for ISIC 2018 and BUSI.
- ISIC 2018 Task 1 now downloads the official training input and ground-truth ZIP archives from the ISIC Challenge S3 archive.
- BUSI now downloads `Dataset_BUSI.zip` from the official Cairo University Scholars host.
- Multiple official archives are extracted into a shared directory before image/mask resolution.
- Downloads use retries, temporary files, ZIP validation, and cached-archive reuse.
- Local `--source-dir`, `--zip-path`, and explicit `--download-url` workflows remain supported.
- The four balanced session runners require no Kaggle dataset attachment and no Kaggle API token; outbound internet access must remain enabled.
