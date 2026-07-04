# LLM Alignment Pipeline

This repository provides a configuration-first pipeline for three stages:

- alignment
- inference
- evaluation

The orchestrator expands a master config into per-stage runs and writes structured outputs for reproducible local experiments.


## Repository Layout

- `alignment/`: alignment config loading, dataset materialization, LLaMA-Factory config generation, training execution, and export helpers
- `inference/`: local Hugging Face inference and JSONL output writing
- `eval/`: offline evaluation, judge-model integration, metric execution, and report export
- `orchestrator/`: master-config validation, run planning, stage execution, and aggregation
- `configs/`: example and scenario configs
- `LLaMA-Factory/`: vendored alignment backend used by the alignment stage

Local-only data, artifacts, caches, runs, and scratch notes are intentionally excluded from the published repo surface.


## Supported Workflow

The maintained workflow is the local Python environment.

### 1. Create an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

### 2. Install pipeline dependencies

```powershell
pip install -r requirements.txt
```

Optional development tools:

```powershell
pip install -r requirements-dev.txt
```

### 3. Install the alignment backend

Alignment runs depend on the vendored `LLaMA-Factory` submodule:

```powershell
git submodule update --init --recursive
pip install -e .\LLaMA-Factory
pip install -r .\LLaMA-Factory\requirements\metrics.txt
```


## Common Commands

Run an orchestrator config:

```powershell
python -m orchestrator.cli --config .\configs\master_config.yaml
```

Download a benchmark split:

```powershell
python -m orchestrator.cli --dataset TIGER-Lab/MMLU-Pro --split test --max-samples 10
```

Run a single stage directly:

```powershell
python -c "from alignment import run_alignment; print(run_alignment(r'.\configs\alignment_config.example.yaml'))"
python -c "from inference import run_inference; run_inference(r'.\configs\inference_config.example.yaml')"
python -c "from eval import run_evaluation; run_evaluation(r'.\configs\eval_config.example.yaml')"
```


## Safe-Refusal Preparation

Prepare the safe-refusal scenario datasets:

```powershell
python -m orchestrator.cli --dataset PKU-Alignment/PKU-SafeRLHF --split train
python -m orchestrator.cli --dataset PKU-Alignment/PKU-SafeRLHF --split test
python -m orchestrator.cli --dataset LibrAI/do-not-answer --split train
python -m orchestrator.cli --dataset nvidia/HelpSteer2 --split train
python -m orchestrator.cli --dataset nvidia/HelpSteer2 --split validation
python prepare_safe_refusal_scenario_data.py
```

Run the main safe-refusal config:

```powershell
python -m orchestrator.cli --config .\configs\orchestrator_config.safe_refusal_qwen3b.yaml
```

Run the family-by-method safe-refusal config:

```powershell
python -m orchestrator.cli --config .\configs\orchestrator_config.family_method_interaction_safe_refusal.yaml
```


## Output Layout

The orchestrator writes run outputs under `runs/orchestrator/<run-id>/`, including:

- `resolved_master_config.yaml`
- `pipeline_state.json`
- run-level aggregate CSVs
- visualization artifacts
- per-branch generated configs
- per-branch alignment, inference, and evaluation outputs
