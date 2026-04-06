# MLops-Project-Training

This repository contains PyTorch training pipelines for aesthetic quality prediction using precomputed image embeddings, with MLflow tracking and a minimal Jupyter container.

## Repository Summary

The codebase is organized around two datasets:

- `FLICKR-AES`
- `UHD-IQA`

The current implementation trains multilayer perceptron models on cached embedding vectors instead of training an image encoder end to end. Each pipeline reads a manifest CSV, loads cached `.npy` embeddings, trains a regression model, evaluates on split-specific datasets, and logs metrics and artifacts to MLflow.

## What Is In The Repo

### Training entrypoints

- [src/flickr/train_flickr_global.py]: trains a global aesthetic prediction model for Flickr-AES.
- [src/flickr/train_flickr_personalized.py]: trains a personalized Flickr-AES model with user embeddings.
- [src/uhd/train_uhd_global.py]: trains a global quality prediction model for UHD-IQA.

### Shared utilities

- [src/flickr/flickr_common.py]: dataset classes, MLP model definitions, training loops, evaluation helpers, config loading, and Flickr cache path handling.
- [src/uhd/uhd_common.py]: UHD dataset utilities, model definition, training loop, evaluation helpers, and config loading.

### Configuration

- [configs/flickr/train_global.yaml]: Flickr global training configuration.
- [configs/flickr/train_personalized.yaml]: Flickr personalized training configuration.
- [configs/uhd/train_uhd_global.yaml]: UHD global training configuration.

### Container runtime

- [docker/Dockerfile.jupyter-torch-mlflow-cuda]: minimal Jupyter + PyTorch image for running the training code.
- [docker/docker-compose-mlflow.yaml]: Docker Compose setup for the MLflow tracking server and backing Postgres database.

## Data Expectations

The repo does not include raw datasets or embedding caches. The training code expects:

- a manifest CSV path defined in the YAML config
- an output root directory defined in the YAML config
- a cache directory under `output_root/<cache_subdir>`
- precomputed `.npy` embedding files keyed by the manifest rows

## Outputs

Each training run creates a run directory under the configured `output_root` and typically writes:

- model checkpoints
- per-epoch training history CSVs
- prediction CSVs for evaluation splits
- aggregate metrics CSVs

The scripts also log parameters, metrics, tags, checkpoints, and CSV artifacts to MLflow.

## Docker

[Dockerfile.jupyter-torch-mlflow-cuda] is used to bring up the container in which the training has been done. It is built on top of the `quay.io/jupyter/pytorch-notebook:cuda12-latest` base image, and installs the extra Python packages currently required by this repo (along with permissions required to execute the training scripts)

## MLflow Tracking Server

The repository also includes [docker/docker-compose-mlflow.yaml], which is used to bring up the MLflow tracking service stack. This file has been used directly as provided in the additional resources

## Running Training

Run the scripts by passing the desired config file path. Please ensure that the embeddings directory and manifest file is properly configured.

The images, manifest files and embeddings are available at these links: 
FLICKR-AES - https://drive.google.com/drive/folders/1Ht2qgplMZ4PH-bum35mKYhXHRPilN3HM?usp=sharing
UHD-IQA - https://drive.google.com/drive/folders/1LnW5EPsfbvZYK_5axfDK-2add8d6BPx6?usp=sharing

```bash
python src/flickr/train_flickr_global.py configs/flickr/train_global.yaml
python src/flickr/train_flickr_personalized.py configs/flickr/train_personalized.yaml
python src/uhd/train_uhd_global.py configs/uhd/train_uhd_global.yaml
```

If MLflow tracking should be sent to a remote or local tracking server, set `MLFLOW_TRACKING_URI` before launching training.
