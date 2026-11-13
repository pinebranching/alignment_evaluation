# LLM Alignment Pipeline

This project runs language-model alignment (training), local inference, and evaluation from YAML configurations. This guide takes a new user from a fresh checkout to a first safe-refusal smoke experiment.

## Before you start

Install Git and Python **3.11**, and have an internet connection for Python packages, datasets, and model weights. The smoke experiment loads a 3B-parameter model, so it needs substantial free disk space and memory even on CPU. Downloads and training can take a long time. The checkout does not include the datasets, model weights, or LlamaFactory training backend.

Run the commands below from the repository root (the directory containing this README), in a terminal with the virtual environment activated. Create a separate environment on each machine; do not copy one between operating systems.

## 1. Clone the repository and create an environment

On Windows, open PowerShell:

```powershell
git clone https://github.com/pinebranching/alignment_evaluation.git
cd alignment_evaluation
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and retry activation. The change applies only to the current PowerShell session.

On Linux or macOS, open a terminal:

```bash
git clone https://github.com/pinebranching/alignment_evaluation.git
cd alignment_evaluation
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

The version printed must be 3.11.x. If the Python command is missing, install Python 3.11 first; some Linux distributions also require a separate Python venv package.

## 2. Install dependencies

The project's `requirements.txt` lists the core Python dependencies and pins a matching PyTorch, torchvision, and torchaudio set. Training also uses LlamaFactory, a separate upstream repository with its own dependencies. Clone the pinned release, then install both sets together in the activated environment:

```text
git clone --branch v0.9.5 --depth 1 https://github.com/hiyouga/LlamaFactory.git LLaMA-Factory
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -e ./LLaMA-Factory -r ./LLaMA-Factory/requirements/metrics.txt
python -m pip check
python -c "import torch, llamafactory, llamafactory.train.tuner; print('Backend imports OK; torch', torch.__version__)"
python -m orchestrator.cli --help
```

`pip check` should report no broken requirements, the import command should print `Backend imports OK`, and the last command should show the CLI options. If any command fails, resolve the error before continuing.

If pip reports that `torch==2.5.1` has no matching distribution, the configured package index may not provide wheels for your platform. Use a [PyTorch 2.5.1 wheel index](https://pytorch.org/get-started/previous-versions/#v251) appropriate for the machine (CPU or a CUDA version supported by its GPU driver) to install the matching `torch`, `torchvision`, and `torchaudio` trio, then rerun the project install command above. On a cluster, run the CUDA import check on a GPU compute node, with the environment's required CUDA modules loaded.

The CPU example does not need a GPU. To use the GPU example, you need an NVIDIA GPU and a CUDA-enabled PyTorch build. Check on the machine that will run the model:

```text
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

For GPU use, this must print `True`. If it prints `False`, select a compatible PyTorch 2.5.1 CUDA installation command from the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/#v251), run it in the activated environment, and repeat the check and `python -m pip check`. The GPU example uses CUDA; on macOS, start with the CPU example.

### Fix a torchaudio CUDA mismatch

LlamaFactory imports `torchaudio` even for text-only training. Its version and CUDA build must match PyTorch. In one GPU run, PyTorch `2.5.1+cu124` was installed alongside torchaudio `2.11.0`, causing `OSError: libcudart.so.13: cannot open shared object file` during training preflight. Base inference succeeded and `pip check` did not catch the mismatch.

If PyTorch is `2.5.1+cu124`, install the matching CUDA 12.4 torchaudio build in the activated environment:

```bash
python -m pip install --force-reinstall --no-deps \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

The `--no-deps` option avoids replacing the existing PyTorch installation. Verify the packages and the training backend on the GPU node before rerunning:

```bash
python -m pip check
python -c "import torch, torchaudio, llamafactory.train.tuner; print('torch:', torch.__version__, 'torchaudio:', torchaudio.__version__, 'CUDA available:', torch.cuda.is_available()); print('Training backend import OK')"
```

For another PyTorch version or CUDA build, choose matching versions and a wheel index from the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) instead of using the version-specific command above.

## 3. Example configurations

Place the example configuration YAML files in `configs/yaml_scripts/`.

## 4. Download and prepare the example data

The following commands download complete dataset splits and prepare the files required by the safe-refusal examples. They can require substantial time and storage. Check each dataset's Hugging Face access and license terms (potentially log in through `hf auth login`). Run these from the repository root with the environment active. The downloads use `--verification-mode no_checks` because upstream file metadata can trigger `ExpectedMoreDownloadedFilesError`; this skips Hugging Face's integrity checks, so confirm the output files are nonempty before preparing the data.

```text
python -m orchestrator.cli --dataset PKU-Alignment/PKU-SafeRLHF --split train --verification-mode no_checks
python -m orchestrator.cli --dataset PKU-Alignment/PKU-SafeRLHF --split test --verification-mode no_checks
python -m orchestrator.cli --dataset LibrAI/do-not-answer --split train --verification-mode no_checks
python -m orchestrator.cli --dataset nvidia/HelpSteer2 --split train --verification-mode no_checks
python -m orchestrator.cli --dataset nvidia/HelpSteer2 --split validation --verification-mode no_checks
```

Wait for each download to finish successfully before starting the next. Check that the downloaded files in `configs/benchmark_data/` are nonempty, then prepare the data:

```text
python exemplary_scenario_runs/prepare_safe_refusal_scenario_data.py
```

The preparation script expects the default download filenames, so use the commands as shown. Raw downloads and prepared evaluation data are stored in `configs/benchmark_data/`; prepared training data is stored in `configs/data/`.

## 5. Run the smoke experiment

After verifying CUDA as described in step 2, use the NVIDIA GPU configuration:

```text
python -u -m orchestrator.cli \
  --config configs/yaml_scripts/orchestrator_config.safe_refusal_qwen3b_gpu_smoke.yaml
```

The first run downloads the model from Hugging Face. Results and logs are written under `runs/<run-id>/`. Check `pipeline_state.json` for stage status and the stage logs for errors. The data, copied configs, backend checkout, and run outputs are ignored by Git. Once the smoke experiment works, the other YAML files in `configs/github_repo/` provide larger examples; inspect their model, data, and hardware requirements before running them.
