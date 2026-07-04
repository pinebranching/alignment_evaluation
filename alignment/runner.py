"""Run LLaMA-Factory train and export commands while capturing logs and failures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from pathlib import Path

import yaml

from .schemas import AlignmentConfig, CommandResult


def _llamafactory_src_dir() -> Path | None:
    """Return the bundled LLaMA-Factory source tree when it exists locally."""

    candidate = Path(__file__).resolve().parents[1] / "LLaMA-Factory" / "src"
    if candidate.is_dir():
        return candidate
    return None


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    """Write a YAML payload to disk for LLaMA-Factory to consume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_train_command(config: AlignmentConfig, config_path: Path) -> list[str]:
    """Build the LLaMA-Factory train command using the configured launcher mode."""

    return _build_backend_command(config, "train", config_path)


def build_export_command(config: AlignmentConfig, config_path: Path) -> list[str]:
    """Build the LLaMA-Factory export command using the configured launcher mode."""

    return _build_backend_command(config, "export", config_path)


def run_command(
    command: list[str],
    log_path: Path,
    config: AlignmentConfig,
    phase: str,
) -> CommandResult:
    """Execute one LLaMA-Factory command and surface progress during long runs."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_command_env(config)
    start_time = time.perf_counter()
    preflight = _run_backend_preflight(config, env)

    with log_path.open("a", encoding="utf-8") as log_handle:
        if log_handle.tell() > 0:
            _emit_log_line(log_handle, f"[alignment] --- restarting {phase} ---")
        _emit_log_line(
            log_handle,
            f"[alignment] Preflight OK for {phase}. python={preflight['python_executable']}",
        )
        _emit_log_line(
            log_handle,
            f"[alignment] versions: python={preflight['python_version']} torch={preflight['torch_version']} llamafactory={preflight['llamafactory_version']}",
        )
        _emit_log_line(
            log_handle,
            f"[alignment] imports: torch={preflight['torch_path']} llamafactory={preflight['llamafactory_path']}",
        )
        _emit_log_line(log_handle, f"[alignment] Starting {phase}. Log: {log_path}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            shell=False,
        )
        assert process.stdout is not None
        queue: Queue[str | None] = Queue()
        reader = threading.Thread(
            target=_enqueue_stdout_lines,
            args=(process.stdout, queue),
            daemon=True,
        )
        reader.start()

        while True:
            try:
                item = queue.get(timeout=config.runtime.heartbeat_interval_seconds)
            except Empty:
                if process.poll() is None:
                    elapsed = time.perf_counter() - start_time
                    _emit_log_line(
                        log_handle,
                        # f"[alignment] {phase} still running after {elapsed:.1f}s with no new output.",
                    )
                    continue
                break

            if item is None:
                break

            log_handle.write(item)
            log_handle.flush()

        return_code = process.wait()

    duration_seconds = time.perf_counter() - start_time
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(command)}. Log: {log_path}"
        )

    return CommandResult(
        command=command,
        log_path=log_path,
        return_code=return_code,
        duration_seconds=duration_seconds,
        launcher_python=preflight["python_executable"],
    )


def _build_command_env(config: AlignmentConfig) -> dict[str, str]:
    """Build the subprocess environment from the high-level runtime config."""

    env = dict(os.environ)
    launcher_python = Path(config.runtime.python_executable or sys.executable).expanduser().resolve()
    launcher_python_dir = str(launcher_python.parent)
    env["PATH"] = _prepend_path_entry(launcher_python_dir, env.get("PATH"))

    env_root = _infer_environment_root(launcher_python)
    if env_root is not None:
        env.setdefault("CONDA_PREFIX", str(env_root))

    llama_factory_src_dir = _llamafactory_src_dir()
    if llama_factory_src_dir is not None:
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(llama_factory_src_dir)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if config.runtime.cpu_threads > 0:
        env["OMP_NUM_THREADS"] = str(config.runtime.cpu_threads)
    if config.runtime.target != "cpu" and config.runtime.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(config.runtime.cuda_visible_devices)
    if config.runtime.force_torchrun:
        env["FORCE_TORCHRUN"] = "1"
        env["NPROC_PER_NODE"] = str(_resolve_nproc_per_node(config))
    env.update(config.runtime.extra_env)
    return env


def _prepend_path_entry(entry: str, existing_path: str | None) -> str:
    """Prepend one PATH entry while avoiding duplicates."""

    if not existing_path:
        return entry

    parts = [part for part in existing_path.split(os.pathsep) if part]
    normalized_entry = os.path.normcase(os.path.normpath(entry))
    filtered_parts = [
        part
        for part in parts
        if os.path.normcase(os.path.normpath(part)) != normalized_entry
    ]
    return os.pathsep.join([entry, *filtered_parts])


def _infer_environment_root(python_executable: Path) -> Path | None:
    """Infer a virtualenv/conda root from the launcher interpreter path."""

    parent = python_executable.parent
    if parent.name.lower() in {"bin", "scripts"}:
        return parent.parent
    return None


def _build_backend_command(config: AlignmentConfig, phase: str, config_path: Path) -> list[str]:
    """Build the backend command in a way that stays bound to the current interpreter."""

    if config.runtime.command_mode == "module":
        python_executable = config.runtime.python_executable or sys.executable
        return [python_executable, "-m", "llamafactory.cli", phase, str(config_path)]

    executable = shutil.which("llamafactory-cli")
    if executable is None:
        raise RuntimeError(
            "The command 'llamafactory-cli' is not available on PATH. Install LLaMA-Factory first."
        )
    return [executable, phase, str(config_path)]


def _resolve_nproc_per_node(config: AlignmentConfig) -> int:
    """Resolve torchrun worker count for single-node LLaMA-Factory launches."""

    if config.runtime.nproc_per_node is not None:
        return config.runtime.nproc_per_node
    if config.runtime.target != "cpu" and config.runtime.cuda_visible_devices:
        return len(config.runtime.cuda_visible_devices)
    return 1


def _run_backend_preflight(config: AlignmentConfig, env: dict[str, str]) -> dict[str, str]:
    """Validate backend imports with the same interpreter used by the command."""

    python_executable = config.runtime.python_executable or sys.executable
    script = (
        "import json, sys\n"
        "if sys.version_info >= (3, 14):\n"
        "    raise RuntimeError('Python 3.14 is not supported for this LLaMA-Factory launch path. Use Python 3.11-3.13.')\n"
        "import torch\n"
        "import llamafactory\n"
        "import llamafactory.train.tuner\n"
        "print(json.dumps({\n"
        "    'python_executable': sys.executable,\n"
        "    'python_version': sys.version.replace('\\n', ' '),\n"
        "    'torch_version': str(getattr(torch, '__version__', 'unknown')),\n"
        "    'torch_path': str(getattr(torch, '__file__', 'unknown')),\n"
        "    'llamafactory_version': str(getattr(llamafactory, '__version__', 'unknown')),\n"
        "    'llamafactory_path': str(getattr(llamafactory, '__file__', 'unknown')),\n"
        "}))\n"
    )
    completed = subprocess.run(
        [python_executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LLaMA-Factory backend preflight failed for "
            f"{python_executable}. stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )

    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "LLaMA-Factory backend preflight did not return parseable JSON. "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        ) from exc


def _enqueue_stdout_lines(stream, queue: Queue[str | None]) -> None:
    """Read process output in a background thread so idle periods can emit heartbeats."""

    try:
        while True:
            raw = stream.readline()
            if not raw:
                break
            if isinstance(raw, str):
                queue.put(raw)
                continue
            queue.put(_decode_process_output(raw))
    finally:
        queue.put(None)


def _decode_process_output(raw: bytes) -> str:
    """Decode backend output robustly across Linux and Windows consoles."""

    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _emit_log_line(log_handle, message: str) -> None:
    """Write one synthetic status line to both stdout and the command log."""

    line = message + "\n"
    print(line, end="")
    log_handle.write(line)
    log_handle.flush()
