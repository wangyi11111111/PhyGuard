# LiteTrust-PINN Agent Rules

This repository is the `LiteTrust-PINN` project for sparse traffic state reconstruction.

Core execution rules:

1. Do not start with full experiments, multi-dataset runs, or many baselines.
2. Every stage must pass a smoke test before any expansion.
3. After every meaningful code change or run, update `PROGRESS.md`.
4. The target environment is Windows with a single RTX 4060, so defaults must stay conservative.
5. Prefer lightweight PyTorch implementations over large transformers, neural operators, diffusion models, or RL-heavy pipelines.
6. Do not download large datasets unless the user explicitly allows it.
7. If a real dataset is missing, use a clear fallback path instead of crashing the entire pipeline.
8. If a stage fails, stop expansion and document the failure plus diagnosis before moving on.
