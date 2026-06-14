# Kaggle Session 3 and 4 (BUSI)

The official Cairo University BUSI host may return HTTP 403 to Kaggle. Download the original `Dataset_BUSI.zip` through the official dataset page, upload the unchanged ZIP as a private Kaggle notebook input, then run:

```bash
%%bash
set -euo pipefail
cd /kaggle/working
rm -rf DT-unet

git clone --depth 1 \
  https://github.com/tydeptrai21042004/DT-unet.git \
  DT-unet

cd DT-unet
export DEVICE=cuda
export INSTALL_DEPS=1
export RUN_TESTS=1
export AUTO_FIND_KAGGLE_INPUT=1

bash run_hc_session_4.sh
```

The runner automatically searches `/kaggle/input` for `Dataset_BUSI.zip` or another ZIP filename containing `BUSI`.

For an explicit path:

```bash
export BUSI_ZIP_PATH=/kaggle/input/<dataset-slug>/Dataset_BUSI.zip
bash run_hc_session_4.sh
```
