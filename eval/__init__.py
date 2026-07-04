import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

def run_evaluation(*args, **kwargs):
    from .service import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)

__all__ = ["run_evaluation"]
