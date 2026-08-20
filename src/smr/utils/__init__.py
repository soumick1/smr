"""Shared utilities: seeding, config, reporting, geometry."""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import yaml

from . import geometry  # noqa: F401  (re-export)

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def project_root() -> pathlib.Path:
    return _ROOT


def rng(seed: int) -> np.random.Generator:
    """Single entry point for randomness; every module takes an explicit seed."""
    return np.random.default_rng(seed)


def load_config(name: str = "default") -> dict:
    path = _ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class Report:
    """Collects named PASS/FAIL checks with values; writes JSON; prints table."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[dict] = []
        self.t0 = time.time()

    def check(self, name: str, value, passed: bool, criterion: str):
        self.rows.append(
            dict(name=name, value=_jsonable(value), passed=bool(passed),
                 criterion=criterion)
        )
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name:38s} {criterion:34s} value={value}")
        return passed

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)

    def save(self, path):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(title=self.title, elapsed_s=round(time.time() - self.t0, 2),
                       all_passed=self.all_passed, checks=self.rows)
        path.write_text(json.dumps(payload, indent=2))
        print(f"  report -> {path}  (all_passed={self.all_passed})")


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v
