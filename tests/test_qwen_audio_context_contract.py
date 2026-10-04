"""Two invariants in the Qwen2.5-Omni adapter that fail silently when broken.

Both defects these tests pin produced *well-formed* output. Neither raised, neither
logged a per-sample warning, and both would have been scored as ordinary answers:

* **No audio block in the conversation.** The chat template emits
  ``<|audio_bos|><|AUDIO|><|audio_eos|>`` only when the conversation declares audio.
  Without it the rendered prompt has no audio position, the waveform passed to the
  processor rides along as an inert ``input_features`` tensor, and every audio mode
  decodes byte-identically -- which is also exactly what a model that ignores audio
  looks like. The stored pre-fix cells show it: six modes, one prediction hash.

* **Video tokens budgeted against the wrong context.** ``qwen_omni_utils`` sizes a
  video from ``MODEL_SEQ_LEN`` (default 128000); this checkpoint's context is 32768.
  High-resolution clips rendered to ~43.5k tokens. The tokenizer warns once per
  *process*, so a 60-item run overflowing a third of its items emits one line.

These are source-contract tests. They read the adapter's real text rather than
importing it, because importing pulls in transformers and the upstream utils; a
refactor that drops either guard, or that moves the ``qwen_omni_utils`` import above
the ``MODEL_SEQ_LEN`` assignment, fails here instead of in a silently wrong grid.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
QWEN_PATH = REPO_ROOT / "scripts" / "eval_avqa_qwenomni.py"

# Qwen2.5-Omni-7B: thinker_config.text_config.max_position_embeddings.
CHECKPOINT_CONTEXT = 32768


@pytest.fixture(scope="module")
def src() -> str:
    return QWEN_PATH.read_text()


# --- audio actually reaches the model --------------------------------------


def test_template_conversation_declares_audio(src: str) -> None:
    """The conversation used for the template must carry an audio block."""
    match = re.search(r"conv_tmpl\s*=\s*\[(.+?)\n            \]", src, re.DOTALL)
    assert match is not None, "conv_tmpl not found; the template conversation was renamed"
    assert '"type": "audio"' in match.group(1), (
        "conv_tmpl declares no audio block, so apply_chat_template emits no <|AUDIO|> "
        "placeholder and the waveform is ignored"
    )


def test_mm_info_conversation_declares_no_audio(src: str) -> None:
    """process_mm_info must not be handed audio, or it loads a waveform off disk.

    The point of the split is that the harness supplies exactly the waveform
    ``mode_wav()`` built -- silence, noise, a donor clip -- not whatever the video
    happens to contain.
    """
    match = re.search(r"conv_mm\s*=\s*\[(.+?)\n            \]", src, re.DOTALL)
    assert match is not None, "conv_mm not found; the mm-info conversation was renamed"
    assert '"type": "audio"' not in match.group(1), (
        "conv_mm declares audio, so process_mm_info would decode the video's own track "
        "and the audio intervention would no longer control what the model hears"
    )
    assert "process_mm_info(conv_mm" in src, "process_mm_info must run on the video-only copy"


def test_audio_placeholder_guard_present(src: str) -> None:
    """A rendered prompt without <|AUDIO|> must raise, not run."""
    assert '"<|AUDIO|>" not in chat' in src, "the <|AUDIO|> placeholder guard was removed"
    guard = src[src.index('"<|AUDIO|>" not in chat') :]
    assert "raise" in guard[:400], "the <|AUDIO|> check no longer raises"


# --- the sequence fits the context -----------------------------------------


def test_model_seq_len_is_set_before_qwen_omni_utils_is_imported(src: str) -> None:
    """Ordering invariant: the env var is read at qwen_omni_utils import time.

    If the import moves above the assignment the budget silently reverts to the
    128000 default and high-resolution clips overflow again, with no error.
    """
    assign = src.index("MODEL_SEQ_LEN")
    imports = [m.start() for m in re.finditer(r"(?:from|import)\s+qwen_omni_utils", src)]
    assert imports, "no qwen_omni_utils import found; this test needs updating"
    assert assign < min(imports), (
        "MODEL_SEQ_LEN is assigned after qwen_omni_utils is imported, so the module has "
        "already read the 128000 default and the video budget is ~4x the real context"
    )


def test_budget_leaves_headroom_under_the_checkpoint_context(src: str) -> None:
    seq_len = int(re.search(r'os\.environ\.setdefault\("MODEL_SEQ_LEN",\s*"(\d+)"\)', src).group(1))
    limit = int(re.search(r"CONTEXT_LIMIT\s*=\s*(\d+)", src).group(1))
    assert limit == CHECKPOINT_CONTEXT, (
        f"CONTEXT_LIMIT {limit} no longer matches the checkpoint's {CHECKPOINT_CONTEXT}"
    )
    assert seq_len < limit, f"video budget {seq_len} does not fit the context {limit}"


def test_context_guard_runs_before_generate(src: str) -> None:
    """An item that cannot be measured must be recorded as a failure, not mis-decoded."""
    assert "CONTEXT_LIMIT" in src, "the context guard was removed"
    guard = src.index("n_tok + 64 > CONTEXT_LIMIT")
    generate = src.index("model.generate(")
    assert guard < generate, "the context guard runs after generate, which is too late"
    assert "raise RuntimeError" in src[guard : guard + 400], "the context check no longer raises"
