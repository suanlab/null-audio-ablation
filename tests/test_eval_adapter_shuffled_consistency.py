"""Cross-adapter shuffled-mapping consistency (CHECK.md G.3 regression gate).

Permanent guard against the B.1 / L.1 defect: every eval adapter that consumes
``--audio_mode shuffled`` must produce the IDENTICAL deterministic donor
mapping for the same (n, seed). This test imports each adapter's
``build_shuffled_mapping`` and asserts pairwise equality at the two sample
sizes used in the paper (n=98 matched subset, n=548 larger-pool robustness).

If this test ever fails, a cross-model R-Sh comparison in the paper is
silently on different shuffled realities.
"""

# pyright: reportMissingImports=false, reportUnknownVariableType=false

from __future__ import annotations

import ast
import importlib.util
import random
import sys
from pathlib import Path

import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
_VL2 = _REPO.parent / "VideoLLaMA2" / "eval_avqa_5mode.py"

# Import each adapter's build_shuffled_mapping in isolation. We do NOT exec
# the full module main() — only the function definition needs to be exposed.
# To avoid heavy upstream imports (torch, transformers) at test collection
# time, we extract the function by parsing the file rather than execing.


def _extract_build_shuffled_fn(path: Path):
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_shuffled_mapping":
            mod = ast.Module(body=[node], type_ignores=[])
            code = compile(mod, str(path), "exec")
            ns: dict = {"random": random}
            exec(code, ns)
            return ns["build_shuffled_mapping"]
    raise AssertionError(f"build_shuffled_mapping not found in {path}")


ADAPTERS = {
    "in_house": _SCRIPTS / "eval_avqa.py",
    "in_house_fast": _SCRIPTS / "eval_avqa_fast.py",
    "qwenomni": _SCRIPTS / "eval_avqa_qwenomni.py",
    "videosalmonn2": _SCRIPTS / "eval_avqa_videosalmonn2.py",
}
if _VL2.exists():
    ADAPTERS["videollama2"] = _VL2


@pytest.mark.parametrize("n", [98, 548])
def test_all_adapters_share_shuffled_mapping(n: int) -> None:
    """Every adapter's build_shuffled_mapping must agree at (n, seed=42)."""
    mappings = {name: _extract_build_shuffled_fn(p)(n, 42) for name, p in ADAPTERS.items()}
    reference_name, reference = next(iter(mappings.items()))
    for name, m in mappings.items():
        assert m == reference, (
            f"adapter '{name}' shuffled mapping at n={n} differs from "
            f"'{reference_name}'. First-12 diff: {m[:12]} vs {reference[:12]}"
        )


def test_shuffled_mapping_is_derangement_n98() -> None:
    """No sample maps to itself (the derangement contract)."""
    p_any = next(iter(ADAPTERS.values()))
    m = _extract_build_shuffled_fn(p_any)(98, 42)
    assert all(m[i] != i for i in range(98))


def test_shuffled_mapping_first_entries_n98_seed42() -> None:
    """Pin the canonical (n=98, seed=42) mapping prefix so changes are caught."""
    p_any = next(iter(ADAPTERS.values()))
    m = _extract_build_shuffled_fn(p_any)(98, 42)
    assert m[:12] == [47, 84, 4, 36, 67, 42, 13, 2, 73, 44, 22, 60]
