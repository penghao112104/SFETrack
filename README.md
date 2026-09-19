# SFETrack: Spatio-temporal MoE based Feature Enhancement for RGBT Tracking

You can download the model and results from [here](https://pan.baidu.com/s/18ivR61BnGO5pjC4Fkvo6Fg?pwd=9ir3)

### Installation

Create and activate a conda environment:

```bash
conda create -n sfetrack python=3.10
conda activate sfetrack
```

Install PyTorch and TorchVision compatible with your CUDA version. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install causal-conv1d mamba-ssm --no-build-isolation
```

### Path Setting

Run the following command to configure the local paths:

```bash
cd <PATH_OF_SFETRACK>
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir <PATH_OF_DATASETS> \
  --save_dir ./output
```

### Training

Download `DropTrack_k700_800E_alldata.pth.tar` from the official [DropTrack](https://github.com/jimmy-dq/DropTrack) repository and place it under `./pretrained/`.

```bash
DATA_DIR=<PATH_OF_LASHER> \
CUDA_VISIBLE_DEVICES=0,1 \
bash train_sfetrack.sh
```

Training logs are saved under `./output/logs/`.

### Testing

Run the following command with the path containing only the test sequences:

```bash
SEQ_HOME=<PATH_OF_TEST_SEQUENCES> \
DATASET_NAME=LasHeR \
CUDA_VISIBLE_DEVICES=0 \
bash test_sfetrack.sh <PATH_OF_CHECKPOINT>
```

`DATASET_NAME` supports `GTOT`, `RGBT210`, `RGBT234`, and `LasHeR`. Tracking results are saved under:

```text
RGBT_workspace/results/<dataset>/SFETrack/
```

We use [RGBT_toolkit_python](https://github.com/Alexadlu/RGBT_toolkit_python) to evaluate the tracking results on GTOT, RGBT210, RGBT234, and LasHeR.

## Acknowledgment

- This repository is based on [BAT](https://github.com/SparkTempest/BAT), which is an excellent work.
- Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library.
