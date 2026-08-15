# BharatVox — Tech Stack

Companion to [`METHODS.md`](METHODS.md). This document specifies the complete technical stack for BharatVox development, training, serving, and monitoring.

BharatVox is developed primarily on **Linux**. The stack is chosen so that everything runs on a single local GPU today and scales to multi-GPU/cloud later without rework.

> **Open-source-only policy:** Every tool in the official BharatVox pipeline must be open source (OSI-approved license). No proprietary SaaS, no closed-source dependencies in training, data processing, evaluation, serving, or monitoring. The single acknowledged exception is the **NVIDIA driver/CUDA layer**, which is proprietary but currently unavoidable for GPU training; it is isolated behind PyTorch so the codebase itself stays vendor-neutral (ROCm/CPU paths remain possible).

## 1. Hardware Baseline

| Tier | Hardware | Role |
|---|---|---|
| **Current (Phase 0–1)** | Local Linux machine, NVIDIA RTX 4050 Laptop GPU, **6 GB VRAM** | All development, preprocessing, Phase 0 proof-of-run, BharatVox-Tiny training |
| **Scale-up (Phase 2+)** | Cloud GPU (single T4/L4/A10 → multi-GPU A100/H100 as needed) | Larger data volumes, BharatVox-Mini and beyond |

**6 GB VRAM feasibility notes (design constraints, not afterthoughts):**

- A 2M–10M parameter Conformer-CTC in BF16 with dynamic bucketed batching fits comfortably in 6 GB; the binding constraint is batch size, handled via gradient accumulation (METHODS.md — Training Methodology).
- RTX 4050 (Ada Lovelace) supports **BF16 natively** — BF16 is the default precision from day one.
- Feature extraction (log-Mel) runs on CPU in the dataloader; GPU is reserved for the model.
- `torch.compile` and fused optimizers are used where stable to maximize throughput on limited VRAM.
- Nothing in the codebase may assume a specific GPU; device, precision, and batch settings live in config files only.

## 2. Core ML Stack

| Layer | Tool | Purpose / Notes |
|---|---|---|
| OS | Linux (Ubuntu 22.04+ LTS) | Primary development and training platform |
| Language | Python 3.11+ | Single language across the project |
| DL Framework | **PyTorch 2.x** (BSD-3) | Model, training loop, AMP/BF16, `torch.compile`, DDP for later multi-GPU |
| Audio | **torchaudio** (BSD-2) | Feature extraction, resampling, I/O |
| Audio decoding | **FFmpeg** (LGPL/GPL), soundfile (BSD-3) | Format conversion, robust decoding of heterogeneous source audio |
| Datasets/Hub | **Hugging Face** (`datasets`, `huggingface_hub` — Apache-2.0 libraries) | Dataset download/streaming, checkpoint and dataset-card publishing for open release; the Hub is used only as a public distribution mirror, never as a pipeline dependency |
| Tokenizer | **SentencePiece** (Apache-2.0) | Phase 1 multilingual unigram experiments (Phase 0 is character-level, no tokenizer) |
| Metrics | **jiwer** (Apache-2.0) | WER/CER computation |
| Numerics | NumPy | General numerics |
| Config | **YAML + OmegaConf/Hydra** | Every experiment fully described by a versioned config file (METHODS.md — Reproducibility) |

## 3. Data Engineering

| Concern | Tool | Notes |
|---|---|---|
| Manifests | JSON Lines per split | One record per utterance: audio path, duration, transcript, language, source dataset, license tag |
| Data versioning | Manifest hash + dataset snapshot IDs (optionally **DVC**) | Reproducibility per METHODS.md; DVC adopted if manifest-hash discipline proves insufficient |
| Storage layout | Raw → processed → manifests, strictly separated | Raw data is never modified in place |
| Validation | Automated checks in the preprocessing pipeline | Duration bounds, sample-rate, script/Unicode validation, transcript-audio pairing |

## 4. Experiment Tracking

| Tool | Role |
|---|---|
| **MLflow** (Apache-2.0) | Primary experiment tracker: params, metrics, artifacts, model registry. Self-hosted, runs fine locally, and matches the reproducibility record in METHODS.md — Reproducibility |
| **TensorBoard** (Apache-2.0) | Lightweight in-training curves (loss, LR, grad norm) |

Both are fully open source and self-hosted; no proprietary tracking SaaS is used.

## 5. Containerization and Orchestration

| Tool | Role | When |
|---|---|---|
| **Docker** + NVIDIA Container Toolkit | Reproducible training/inference environment; the training image (CUDA + PyTorch + deps) is version-tagged and referenced in every experiment record | From Phase 0 |
| Docker Compose | Local multi-service dev (MLflow server + training container + monitoring) | From Phase 0 |
| **Kubernetes** | Serving and scaled workloads: inference API deployment, autoscaling, later multi-node training jobs | **Not used in Phase 0–1.** Introduced when serving the streaming prototype (Phase 4) or when training outgrows one machine. A laptop-friendly path (k3s/minikube) is used for serving experiments before any cloud cluster |

All container and orchestration tooling above is open source (Docker Engine/moby: Apache-2.0; Kubernetes: Apache-2.0; k3s: Apache-2.0). Podman is an acceptable drop-in where a daemonless engine is preferred.

> **Honest scoping:** Kubernetes adds no value to a single-GPU training loop. It is in the stack for the serving/scaling stages, and its introduction point is deliberately deferred so Phase 0–1 stay simple and debuggable.

## 6. Monitoring and Observability

| Tool | Role |
|---|---|
| **Prometheus** (Apache-2.0) | Metrics collection (system, GPU, serving) |
| **Grafana** (AGPL-3.0 OSS edition) | Dashboards: GPU utilization/VRAM/temperature during training; latency, real-time factor, throughput, and error rates for the streaming prototype |
| NVIDIA DCGM Exporter (Apache-2.0; wraps the proprietary driver — see CUDA exception above) or `nvidia-smi` exporter locally | GPU metrics into Prometheus |
| Python logging + structured logs | Training-run logs, kept with experiment artifacts |

Training-quality metrics (loss, WER) live in MLflow/TensorBoard; Grafana covers **infrastructure and serving** metrics. The two are not mixed.

## 7. Serving and Streaming (Phase 4)

| Tool | Role |
|---|---|
| **FastAPI** (MIT) | Inference API; WebSocket endpoint for streaming transcription |
| **ONNX Runtime** (MIT) or TorchScript | Exported inference graph; CPU-viable inference for small models |
| Uvicorn | ASGI serving |
| Kubernetes + Prometheus/Grafana | Deployment, autoscaling, latency dashboards (Sections 5–6) |

## 8. Code Quality and CI

| Tool | Role |
|---|---|
| **Git** (GPL-2.0) | Version control; every official experiment pinned to a commit (METHODS.md — Reproducibility). Hosted on GitHub as a public mirror; the repository is fully portable to open-source hosts (GitLab CE, Gitea/Forgejo) at any time |
| CI | GitHub Actions for the public repo, with all logic kept in plain shell/Make targets so CI is portable to open-source runners (Woodpecker, GitLab CE, Gitea Actions) — no vendor-specific lock-in in the pipeline itself |
| **ruff** | Linting + formatting |
| **pytest** | Unit tests — especially normalization and tokenizer round-trip tests per script, which are the highest-risk pure-code components |
| pre-commit | Enforce lint/format/test hooks locally |

## 9. Stack Principle

> **Everything open source. Everything reproducible from `git clone` + one Docker image + one config file + one data manifest.** Any tool that cannot fit this contract — closed-source, SaaS-only, or unreproducible — is not part of the official pipeline.
