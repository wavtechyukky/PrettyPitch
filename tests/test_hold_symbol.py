"""The hold symbol (recipe ``hold_symbol`` / config ``model.hold_symbol``).

1. ``phoneset.load(hold=...)`` appends the symbol to the phoneme table and the nucleus set,
   leaving every existing id unchanged, and never duplicates it.
2. The checkpoint records the symbol whenever it was used, for either encoder, so hosts can
   tell a model trained on "ay -" labels from one that expects "ay ay".
"""
from types import SimpleNamespace

import pytest

from duration_model import phoneset
from preprocess.vocab import Vocab


def test_hold_is_appended_without_moving_existing_ids():
    base, held = phoneset.load("en_full"), phoneset.load("en_full", hold="-")
    assert held.phonemes == base.phonemes + ("-",)
    assert held.ph2id["-"] == base.n_phonemes
    assert all(held.ph2id[p] == i for i, p in enumerate(base.phonemes))
    assert held.nuclei == base.nuclei | {"-"} and held.is_nucleus("-")
    assert "-" not in held.consonants and held.consonants == base.consonants
    assert held.hold_symbol == "-" and base.hold_symbol is None


def test_digest_mentions_the_hold_symbol_only_when_used():
    assert "hold_symbol" not in phoneset.load("ja").digest()
    assert phoneset.load("ja", hold="-").digest()["hold_symbol"] == "-"


def test_hold_already_in_a_custom_table_is_not_duplicated(tmp_path):
    table = tmp_path / "custom.phonemes"
    table.write_text("\n".join(list(phoneset.load("ja").phonemes) + ["-"]) + "\n", encoding="utf-8")
    ps = phoneset.load("ja", str(table), None, "-")
    assert ps.phonemes.count("-") == 1 and ps.phonemes[-1] == "-" and ps.is_nucleus("-")


@pytest.mark.parametrize("bad", ["", "pau", "br"])
def test_silence_or_empty_hold_symbols_are_refused(bad):
    with pytest.raises(ValueError):
        phoneset.load("ja", hold=bad)


def test_vocab_built_from_the_held_set_gives_the_symbol_the_last_id():
    v = Vocab(list(phoneset.load("ja", hold="-").phonemes))
    assert v.get("-") == v.n_phonemes - 1


def _ckpt_cfg(model: dict):
    from train import ckpt_config
    ps = phoneset.load("en_full", hold=model.get("hold_symbol"))
    cfg = {"lang": "en_full", "audio": {"sr": 44100, "hop": 256}, "pitch": {}, "decomp": {}, "model": model}
    ds = SimpleNamespace(frame_rate=44100 / 256, n_phonemes=ps.n_phonemes)
    return ckpt_config(cfg, ds, Vocab(list(ps.phonemes)), sorted(ps.nuclei), 1, 1)


def test_checkpoint_records_the_hold_symbol_for_either_encoder():
    assert "hold_symbol" not in _ckpt_cfg({}) and "encoder" not in _ckpt_cfg({})
    plain = _ckpt_cfg({"hold_symbol": "-"})
    assert plain["hold_symbol"] == "-" and "encoder" not in plain and "-" in plain["phonemes"]
    notes = _ckpt_cfg({"encoder": "notes", "hold_symbol": "-"})
    assert notes["encoder"] == "notes" and notes["hold_symbol"] == "-"
