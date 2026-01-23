# %%
import wandb
import pandas as pd
from typing import Optional, Any


def fetch_wandb_runs_dataframe(project: str) -> pd.DataFrame:
    """Fetch all runs from the 'entity/project' wandb project and return as a DataFrame."""
    api = wandb.Api()
    runs = api.runs(project)

    rows = []

    for run in runs:
        row = {}

        # ---- metadata ----
        row["run_id"] = run.id
        row["name"] = run.name
        row["state"] = run.state
        row["created_at"] = run.created_at

        # ---- config ----
        for k, v in run.config.items():
            if not k.startswith("_"):
                row[f"config/{k}"] = v

        # ---- summary ----
        for k, v in run.summary.items():
            row[f"summary/{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _safe_wandb_log(run: Any, key: str, value: object, step: Optional[int] = None) -> None:
    """Try to log a value to wandb/Neptune run. Works with both APIs.

    This is intentionally permissive: if logging fails we silently continue so this file stays usable
    even if wandb/Neptune isn't available at runtime.
    """
    if run is None:
        return
    try:
        # Try wandb API first: run.log({'key': value}, step=step)
        if hasattr(run, 'log') and callable(run.log):
            log_dict = {key: value}
            if step is not None:
                run.log(log_dict, step=step)
            else:
                run.log(log_dict)
            return
    except Exception:
        pass
    
    try:
        # Fallback to Neptune API: run['key'].log(value, step=step)
        target = run[key]
        if step is None:
            target.log(value)
        else:
            target.log(value, step=step)
        return
    except Exception:
        pass
    
    try:
        # Final fallback: direct assignment
        run[key] = value
    except Exception:
        # give up silently
        return