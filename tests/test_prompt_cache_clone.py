"""Tests for prompt cache cloning helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import types
from typing import Any


def _install_fake_mlx_cache_module() -> None:
    """Install a lightweight ``mlx_lm.models.cache`` stub for imports."""

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    fake_cache.can_trim_prompt_cache = lambda cache: bool(cache)
    fake_cache.trim_prompt_cache = lambda cache, num_tokens: min(num_tokens, len(cache))
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    sys.modules["mlx_lm"] = fake_mlx_lm
    sys.modules["mlx_lm.models"] = fake_models
    sys.modules["mlx_lm.models.cache"] = fake_cache


def _load_clone_function() -> Any:
    """Import ``clone_prompt_cache`` while stubbing MLX-backed imports."""

    repo_root = Path(__file__).resolve().parents[1]
    fake_utils_pkg = types.ModuleType("app.utils")
    fake_utils_pkg.__path__ = [str(repo_root / "app" / "utils")]
    sys.modules["app.utils"] = fake_utils_pkg

    _install_fake_mlx_cache_module()
    sys.modules.pop("app.utils.prompt_cache", None)
    module = importlib.import_module("app.utils.prompt_cache")
    module = importlib.reload(module)
    return module.clone_prompt_cache


@dataclass
class _FakeCache:
    """Minimal cache stub implementing the MLX cache clone contract."""

    payload: list[int]
    meta: dict[str, int]

    @property
    def state(self) -> list[int]:
        return self.payload

    @property
    def meta_state(self) -> dict[str, int]:
        return self.meta

    @classmethod
    def from_state(cls, state: list[int], meta_state: dict[str, int]) -> "_FakeCache":
        return cls(payload=state, meta=meta_state)


def test_clone_prompt_cache_returns_independent_cache_objects() -> None:
    """Cloned cache entries should not share mutable state with the source."""

    clone_prompt_cache = _load_clone_function()
    original = [_FakeCache(payload=[1, 2, 3], meta={"offset": 3})]

    cloned = clone_prompt_cache(original)

    assert cloned is not original
    assert isinstance(cloned[0], _FakeCache)
    assert cloned[0] is not original[0]
    assert cloned[0].payload == [1, 2, 3]
    assert cloned[0].meta == {"offset": 3}

    cloned[0].payload.append(4)
    cloned[0].meta["offset"] = 4

    assert original[0].payload == [1, 2, 3]
    assert original[0].meta == {"offset": 3}
