"""Parser and accessors for simple plot.in files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].split("!", 1)[0].strip()


@dataclass(frozen=True)
class PlotConfig:
    path: Path
    values: dict[str, str]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def require(self, key: str) -> str:
        norm = key.lower()
        if norm not in self.values or not self.values[norm].strip():
            raise KeyError(f"Missing required plot.in key: {key}")
        return self.values[norm]

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key.lower(), default)

    def get_str(self, key: str, default: str) -> str:
        value = self.get(key)
        return default if value is None else str(value).strip()

    def get_int(self, key: str, default: int) -> int:
        value = self.get(key)
        return int(default if value is None else value)

    def get_float(self, key: str, default: float) -> float:
        value = self.get(key)
        return float(default if value is None else value)

    def get_bool(self, key: str, default: bool) -> bool:
        value = self.get(key)
        if value is None:
            return bool(default)
        token = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean for {key}: {value}")

    def get_list(self, key: str, default: Iterable[str] | None = None) -> list[str]:
        value = self.get(key)
        if value is None:
            return list(default or [])
        return [tok for tok in value.replace(",", " ").split() if tok]

    def path_value(self, key: str, default: str | None = None, must_exist: bool = False) -> Path:
        raw = self.get(key, default)
        if raw is None:
            raise KeyError(f"Missing path key: {key}")
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if must_exist and not path.exists():
            raise FileNotFoundError(f"{key} path does not exist: {path}")
        return path

    def prefixed(self, prefix: str) -> dict[str, str]:
        prefix = prefix.lower()
        head = f"{prefix}."
        return {k[len(head):]: v for k, v in self.values.items() if k.startswith(head)}


def load_plot_config(path: str | Path) -> PlotConfig:
    cfg_path = Path(path).expanduser().resolve()
    values: dict[str, str] = {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = _strip_comment(raw)
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{cfg_path}:{lineno}: expected key = value")
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if not key:
                raise ValueError(f"{cfg_path}:{lineno}: empty key")
            values[key] = value.strip()
    return PlotConfig(path=cfg_path, values=values)
