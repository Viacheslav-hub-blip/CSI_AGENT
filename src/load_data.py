import os
from pathlib import Path
import pandas as pd

DEFAULT_DATAFRAMES_DIR = Path(os.getenv("CSI_AGENT_DATAFRAMES_DIR", "DataFrames"))


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataframe format: {path.suffix}")


def _resolve_default_path() -> Path | None:
    env_path = os.getenv("CSI_AGENT_DATAFRAME_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None
    candidates = [
        DEFAULT_DATAFRAMES_DIR / "source_dataframe.pkl",
        DEFAULT_DATAFRAMES_DIR / "current_dataframe.pkl",
    ]
    for pattern in ("*.pkl", "*.csv", "*.xlsx", "*.xls"):
        candidates.extend(sorted(DEFAULT_DATAFRAMES_DIR.glob(pattern)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_data(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    target_path = Path(path) if path else _resolve_default_path()
    if target_path is None:
        return pd.DataFrame()
    return _read_dataframe(target_path)
