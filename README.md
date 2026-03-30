# DLCV HW1

## Introduction

影像分類作業專案，使用 PyTorch 與 torchvision 實作 100 類別分類。專案支援單模型訓練、單模型推論、集成推論，以及 Optuna 超參數搜尋；目前內含多個已訓練 checkpoint 與對應輸出檔。

主要功能：

- 預訓練 backbone：`resnet18`、`resnet34`、`resnet50`、`resnext50_32x4d`
- 結構變體：`eca_*`、`se_*`、`gmlp_resnet50`
- 訓練策略：RandAugment、MixUp、CutMix、label smoothing、warmup + cosine decay、early stopping
- 測試策略：horizontal flip TTA、multi-checkpoint ensemble
- 實驗追蹤：Weights & Biases
- 調參：Optuna

## Project Structure

```text
.
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── resnet18.pth
│   ├── resnet50.pth
│   ├── eca_resnet50.pth
│   ├── se_resnet50.pth
│   ├── gmlp_resnet50.pth
│   └── optuna_trials/
│       └── gmlp_resnet50_trial_*.pth
├── output/
│   ├── prediction.csv
│   ├── ensemble_prediction.csv
│   └── gmlp_resnet50_trial_46_prediction.csv
├── src/
│   ├── dataset.py
│   ├── main.py
│   ├── model.py
│   └── trainer.py
├── pyproject.toml
└── uv.lock
```

## Environment Setup

本專案使用 Python 3.11 以上。

### 使用 `uv`

```bash
uv sync
```

### 或使用 `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pillow tqdm wandb
```

如果要使用 `mode=tune`，還需要另外安裝：

```bash
pip install optuna
```

## Dataset Format

程式預設資料根目錄為 `data/`，並假設結構如下：

```text
data/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 99/
├── val/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 99/
└── test/
    ├── xxx.jpg
    ├── yyy.jpg
    └── ...
```

- `train/` 與 `val/` 需符合 `torchvision.datasets.ImageFolder` 格式
- 類別資料夾名稱會直接作為 label 名稱，建議使用 `0` 到 `99`
- `test/` 底下直接放圖片，不需要子資料夾

輸出 CSV 格式為：

```csv
image_name,pred_label
001a74bd-6679-4709-aa0b-d10277f057e6,17
002fe951-e857-4ebf-8de4-53c89b9f324e,42
```

## Usage

所有指令都在 repo root 執行。

### 1. Train

```bash
python src/main.py --mode train --model resnet50
```

常用訓練參數：

```bash
python src/main.py \
  --mode train \
  --model gmlp_resnet50 \
  --epochs 28 \
  --batch-size 256 \
  --image-size 288 \
  --lr 0.00156 \
  --weight-decay 6.3e-06 \
  --dropout 0.27 \
  --backbone-lr-scale 0.2 \
  --label-smoothing 0.1187 \
  --randaugment-magnitude 7
```

訓練完成後，checkpoint 預設會儲存在：

```text
models/<model>.pth
```

### 2. Resume Training

```bash
python src/main.py \
  --mode train \
  --model resnet50 \
  --resume-train \
  --model-path models/resnet50.pth
```

### 3. Single Model Inference

```bash
python src/main.py \
  --mode test \
  --model gmlp_resnet50 \
  --model-path models/gmlp_resnet50.pth \
  --output-path output/prediction.csv
```

加入 horizontal flip TTA：

```bash
python src/main.py \
  --mode test \
  --model gmlp_resnet50 \
  --model-path models/gmlp_resnet50.pth \
  --tta-horizontal-flip \
  --output-path output/prediction.csv
```

### 4. Train + Test

```bash
python src/main.py --mode all --model resnet50
```

### 5. Ensemble Inference

指定多個 checkpoint：

```bash
python src/main.py \
  --mode test \
  --ensemble-models resnet50 eca_resnet50 se_resnet50 gmlp_resnet50 resnet18 \
  --ensemble-model-paths \
    models/resnet50.pth \
    models/eca_resnet50.pth \
    models/se_resnet50.pth \
    models/gmlp_resnet50.pth \
    models/resnet18.pth \
  --tta-horizontal-flip \
  --output-path output/ensemble_prediction.csv
```

從 `models/optuna_trials/*.pth` 中自動挑選 validation accuracy 最好的前 `k` 個 checkpoint：

```bash
python src/main.py \
  --mode test \
  --model gmlp_resnet50 \
  --ensemble-top-k 5 \
  --ensemble-candidate-glob "models/optuna_trials/*.pth" \
  --ensemble-split test \
  --tta-horizontal-flip \
  --output-path output/ensemble_prediction.csv
```

若只想先在 validation set 評估集成效果：

```bash
python src/main.py \
  --mode test \
  --model gmlp_resnet50 \
  --ensemble-top-k 5 \
  --ensemble-candidate-glob "models/optuna_trials/*.pth" \
  --ensemble-split val
```

### 6. Hyperparameter Tuning

```bash
python src/main.py \
  --mode tune \
  --model gmlp_resnet50 \
  --optuna-trials 20
```

Optuna 每個 trial 的 checkpoint 會存到：

```text
models/optuna_trials/
```

最佳 trial 的 checkpoint 會複製到：

```text
models/<model>.pth
```

## Implementation Notes

- `src/model.py`
  - 封裝 ResNet / ResNeXt backbone
  - 實作 ECA、SE attention block replacement
  - 實作 `gmlp_resnet50` head
- `src/trainer.py`
  - 負責 train / validation / test 流程
  - 支援 AMP、MixUp、CutMix、early stopping、TTA
- `src/dataset.py`
  - 處理 test split，回傳 `(image, filename)`
- `src/main.py`
  - 統一處理 CLI、資料載入、訓練、推論、ensemble、Optuna

## Performance Snapshot

目前 repo 已包含：

- `models/resnet18.pth`
- `models/resnet50.pth`
- `models/eca_resnet50.pth`
- `models/se_resnet50.pth`
- `models/gmlp_resnet50.pth`
- `models/optuna_trials/gmlp_resnet50_trial_*.pth`
- `output/prediction.csv`
- `output/ensemble_prediction.csv`
- `output/gmlp_resnet50_trial_46_prediction.csv`

依 `wandb` 紀錄，現有結果包含：

- 單一最佳 Optuna trial validation accuracy 約 `0.9233`
- Top-5 ensemble validation accuracy 約 `0.9300`

## W&B

程式預設會初始化 Weights & Biases run。若只想離線紀錄，可先設定：

```bash
export WANDB_MODE=offline
```

也可透過參數調整：

```bash
python src/main.py \
  --mode train \
  --model resnet50 \
  --wandb-project dlcv-hw1 \
  --wandb-run-name exp_resnet50 \
  --wandb-tags baseline,resnet50
```

## Notes

- `gmlp_resnet50` 的 token 長度會跟 `--image-size` 綁定，推論時若載入不同 image size 訓練出的 checkpoint，程式會自動從權重推回對應設定
- 若使用 ensemble 且成員來自不同 image size，程式會分別建立對應 dataloader
- `.gitignore` 目前忽略 `data/`、`*.pth`、`wandb/` 等大型或實驗檔案
