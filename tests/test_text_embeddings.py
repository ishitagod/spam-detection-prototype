"""
pytest suite for features.text_embeddings.

Uses a fake, injectable encoder (not the real sentence-transformers
model) so these tests run fast, deterministically, and without needing
network access or downloaded model weights - only
features/text_embeddings.py's own logic (dedup, row alignment, id_map
construction) is under test here, not MiniLM itself.

Run:
    pytest tests/test_text_embeddings.py -v
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.text_embeddings import (
    _select_device,
    compute_message_embeddings,
    embed_texts,
)

DIM = 4


class FakeModel:
    """
    Deterministic stand-in for a sentence-transformers model: encodes
    each string to a vector derived from its own hash, so identical
    strings always produce identical vectors (needed to meaningfully
    assert dedup correctness) and different strings differ. Tracks every
    call to `.encode()` so tests can assert HOW MANY texts it was asked
    to encode - the actual thing being tested when checking dedup.
    """

    def __init__(self):
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.encode_calls.append(list(texts))
        vectors = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2**32))
            vectors.append(rng.rand(DIM).astype(np.float32))
        return np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, DIM), dtype=np.float32)

    def get_embedding_dimension(self):
        return DIM


BASE = {
    "source": "SMPP", "record_id": "r1", "timestamp": "2026-08-19T10:00:00",
    "text": "hello",
}


def msg(**overrides) -> dict:
    return {**BASE, **overrides}


def test_raises_on_missing_required_column():
    bad = pd.DataFrame([{"source": "SMPP"}])
    with pytest.raises(ValueError):
        compute_message_embeddings(bad, model=FakeModel())


def test_output_row_count_and_order_matches_input():
    rows = [
        msg(record_id="r1", text="alpha"),
        msg(record_id="r2", text="beta"),
        msg(record_id="r3", text="gamma"),
    ]
    embeddings, id_map = compute_message_embeddings(pd.DataFrame(rows), model=FakeModel())
    assert embeddings.shape == (3, DIM)
    assert list(id_map["record_id"]) == ["r1", "r2", "r3"]


def test_identical_text_gets_identical_embedding():
    rows = [
        msg(record_id="r1", text="WIN A PRIZE"),
        msg(record_id="r2", text="something else"),
        msg(record_id="r3", text="WIN A PRIZE"),
    ]
    embeddings, _ = compute_message_embeddings(pd.DataFrame(rows), model=FakeModel())
    assert np.array_equal(embeddings[0], embeddings[2])
    assert not np.array_equal(embeddings[0], embeddings[1])


def test_distinct_text_encoded_exactly_once_not_per_row():
    """The actual dedup guarantee: 5 rows, only 2 distinct texts, the
    fake model must only ever be asked to encode 2 strings - not 5."""
    rows = [
        msg(record_id=f"r{i}", text="WIN A PRIZE" if i % 2 == 0 else "hi there")
        for i in range(5)
    ]
    fake = FakeModel()
    compute_message_embeddings(pd.DataFrame(rows), model=fake)
    assert len(fake.encode_calls) == 1  # embed_texts calls .encode() once, batched
    assert sorted(fake.encode_calls[0]) == ["WIN A PRIZE", "WIN A PRIZE", "hi there"] or \
        sorted(set(fake.encode_calls[0])) == sorted({"WIN A PRIZE", "hi there"})
    assert len(fake.encode_calls[0]) == 2  # 2 distinct texts, not 5 rows


def test_message_key_is_source_pipe_record_id():
    rows = [msg(source="SS7", record_id="12345")]
    _, id_map = compute_message_embeddings(pd.DataFrame(rows), model=FakeModel())
    assert id_map.iloc[0]["message_key"] == "SS7|12345"


def test_same_record_id_different_source_are_different_keys():
    """record_id alone can collide across sources - message_key must
    include source, same reasoning as behavioral.py's sender_id."""
    rows = [
        msg(source="SMPP", record_id="1", text="a"),
        msg(source="SS7", record_id="1", text="b"),
    ]
    _, id_map = compute_message_embeddings(pd.DataFrame(rows), model=FakeModel())
    assert set(id_map["message_key"]) == {"SMPP|1", "SS7|1"}


def test_nan_text_is_treated_as_empty_string_not_a_crash():
    rows = [msg(text=None)]
    embeddings, _ = compute_message_embeddings(pd.DataFrame(rows), model=FakeModel())
    assert embeddings.shape == (1, DIM)


def test_embed_texts_empty_input_returns_zero_rows_correct_dim():
    result = embed_texts([], model=FakeModel())
    assert result.shape == (0, DIM)


def test_embed_texts_output_is_float32():
    result = embed_texts(["hello", "world"], model=FakeModel())
    assert result.dtype == np.float32


def _fake_torch(cuda_available: bool) -> types.ModuleType:
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return fake


# _select_device tested DIRECTLY (unlike everything above, which only goes
# through the public API) - the whole point of this file's FakeModel
# pattern is avoiding a real model load, but _get_model()/_select_device
# are exactly the code path a real (non-injected) run takes BEFORE a model
# object exists, so there's no way to reach it through embed_texts()'s
# `model=` injection without actually loading real weights. Torch is
# faked via sys.modules too (not assumed installed) - same reasoning.
def test_select_device_returns_explicit_request_without_importing_torch(monkeypatch):
    """An explicit request must short-circuit before ever touching torch -
    setting sys.modules['torch'] = None makes any `import torch` raise,
    so this proves the guard rather than just happening to pass."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _select_device("cpu") == "cpu"
    assert _select_device("cuda") == "cuda"


def test_select_device_auto_detects_cuda_when_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    assert _select_device(None) == "cuda"


def test_select_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert _select_device(None) == "cpu"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
