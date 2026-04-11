# PAN'25 "Voight-Kampff" Generative AI Authorship Verification Baselines

LLM detection baselines for PAN 2025 Task 1.

## Quick Start

Run all commands from this repository root.

```bash
cd /pan25_genai_baselines
```

## Local Installation (conda)
```bash
conda create -n pan25 python=3.10 -y
conda activate pan25
pip install -U pip
pip install .
```


## Run Commands (Local)
Activate the conda environment first:

```bash
conda activate pan25
```
Train advanced compression model:
```bash
CUDA_VISIBLE_DEVICES=0 python -m pan25_genai_baselines.cli train-advanced-compression pan25-generative-ai-detection-task1-train/train.jsonl --device cuda:0 --class-weight-mode balanced --class0-weight-multiplier 1.3
```

Run advanced compression inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pan25_genai_baselines.cli advanced-compression pan25-generative-ai-detection-task1-train/val.jsonl demo_out -g pan25-generative-ai-detection-task1-train/val.jsonl --tune-threshold-for acc
```


## Docker Compose Workflow

The repository includes [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

Build image:

```bash
docker compose build
```

Show CLI help:

```bash
docker compose run --rm pan25 --help
```

Run TF-IDF baseline in container:

```bash
docker compose run --rm pan25 tfidf /data/val.jsonl /out
```

Train advanced compression model in container:

```bash
docker compose run --rm pan25 train-advanced-compression /data/train.jsonl -m /out/compression_xgb_model.json
```

Run advanced compression in container:

```bash
docker compose run --rm pan25 advanced-compression -m /out/compression_xgb_model.json /data/val.jsonl /out
```

Evaluate in container:

```bash
docker compose run --rm pan25 advanced-compression -m /out/compression_xgb_model.json /data/val.jsonl /out -g /data/val.jsonl --tune-threshold-for acc
```

## Docker Troubleshooting

If you get `docker: command not found`:

1. Install Docker CLI on the host machine, or use an environment where Docker is already available.
2. If you are inside a dev container, mount host Docker socket (`/var/run/docker.sock`) and use host daemon.
3. Try standalone Compose binary if available:

```bash
docker-compose build
docker-compose run --rm pan25 --help
```

GPU note: `gpus: all` in compose is mainly needed for `binoculars`. Remove this line if you only run CPU baselines.

## Optional: Direct Docker Run

```bash
docker run --rm --gpus=all \
  -v "$(pwd)/pan25-generative-ai-detection-task1-train/val.jsonl:/val.jsonl" \
  -v "$(pwd)/demo_out:/out" \
  ghcr.io/pan-webis-de/pan25-generative-authorship-baselines \
  tfidf /val.jsonl /out
```

`--gpus=all` is only required for binoculars.

## Submit to TIRA

Verify TIRA client:

```bash
tira-cli verify-installation
```

Dry run submission:

```bash
tira-cli code-submission --dry-run --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline tfidf $inputDataset/dataset.jsonl $outputDir'
```

Submit after dry run succeeds:

```bash
tira-cli code-submission --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline tfidf $inputDataset/dataset.jsonl $outputDir'
```

LLM submission example:

```bash
tira-cli code-submission --mount-hf-model meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B --path . --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --command '/usr/local/bin/pan25-baseline binoculars --observer meta-llama/Llama-3.1-8B --performer meta-llama/Llama-3.1-8B-Instruct $inputDataset/dataset.jsonl $outputDir'
```

