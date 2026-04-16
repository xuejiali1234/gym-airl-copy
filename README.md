# gym-airl

AIRL-based lane-change / merge imitation learning project built on NGSIM-style trajectories.

## Overview

This repository contains a training pipeline for highway merging behavior learning with:

- AIRL adversarial imitation learning
- PPO generator training
- Goal-conditioned observation support
- Attention-based social interaction modeling
- Optional safety-aware discriminator regularization

The current training entrypoint is:

- `train_airl_baseline.py`

## Main Structure

- `configs/`: experiment configuration
- `envs/`: gym environment definitions
- `model/`: attention, reward, and safety modules
- `utils/`: dataset loading and preprocessing
- `evaluation/`: offline evaluation scripts
- `plot/`: visualization helpers
- `train_log/`: plotting scripts for training curves

## Quick Start

1. Create / activate your environment.
2. Install dependencies from `requirements.txt`.
3. Place trajectory data under the expected `data/` directory layout.
4. Run training:

```bash
python train_airl_baseline.py
```

## Notes

- Large local artifacts are intentionally not tracked:
  - `data/`
  - `checkpoints/`
  - training logs / generated figures
- The repository keeps source plotting scripts under `train_log/`, but not the generated run folders.

## Current Experiment Features

- Goal-conditioned training
- Attention-based policy / discriminator path
- Safety Q module with scalar-only fusion
- Safety-stage control from config

## Repository Purpose

This repo is intended to track the main project code, configs, docs, and analysis scripts without syncing large model files or local training outputs.
