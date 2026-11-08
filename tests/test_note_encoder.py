"""``model.encoder: notes`` — the hold-token convention read as two streams.

1. The loop-free frame expansion equals the original InputEncoder loops.
2. Hold tokens fold into the vowel and start notes; rest notes come from pau runs.
3. "wan~" (an + hold) and "wan an" (an an) differ only in the phoneme stream.
4. The hold token never reaches an embedding.
5. Padding inside a batch does not change an item.
6. The encoder exports to ONNX with a dynamic frame count.
7. The default encoder and existing checkpoints are untouched.
"""
import numpy as np
import pytest
import torch

from prettypitch.frontend import frames_for, length_regulate, position_in_token
from prettypitch.models.diffusion import InputEncoder, NoteInputEncoder

PAU, W, AN, K, HOLD = 0, 1, 2, 3, 4
PHONEMES = ["pau", "w", "an", "k", "-"]


def make_encoder():
    torch.manual_seed(0)
    return NoteInputEncoder(len(PHONEMES), nucleus_ids=[AN], pau_id=PAU, hold_id=HOLD).eval()


def wan(second):
    """pau w an <second> pau: "an" for 5 + 10 frames, the second part on a higher note."""
    ids = torch.tensor([[PAU, W, AN, second, PAU]])
    pitch = torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.25]])          # forward-filled, normalised
    durs = torch.tensor([[5, 3, 10, 8, 4]])
    return ids, pitch, durs


def test_frontend_matches_legacy_loops():
    g = torch.Generator().manual_seed(0)
    durs = torch.randint(0, 6, (3, 9), generator=g)
    durs[1, -3:] = 0
    feat = torch.randn(3, 9, 5, generator=g)
    frames = frames_for(durs)
    assert torch.equal(length_regulate(feat, durs, frames), InputEncoder.length_regulate(feat, durs))
    assert torch.allclose(position_in_token(durs, frames), InputEncoder._frame_positions(durs), atol=1e-6)


def test_hold_tokens_fold_and_start_notes():
    ph, ph_dur, n_pitch, n_rest, n_dur = make_encoder().streams(*wan(HOLD))
    assert ph[0, :4].tolist() == [PAU, W, AN, PAU]
    assert ph_dur[0].tolist() == [5, 3, 18, 4, 0]                 # the hold joins "an"
    # notes: rest (pau + the onset w), an, the held part, the closing rest
    assert n_dur[0].tolist() == [8, 10, 8, 4, 0]
    assert n_pitch[0, :4].tolist() == [0.0, 0.0, 0.25, 0.25]
    assert n_rest[0, :4].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_leading_consonant_gets_a_rest_note():
    ids = torch.tensor([[K, AN, PAU]])
    _, _, _, n_rest, n_dur = make_encoder().streams(ids, torch.zeros(1, 3), torch.tensor([[2, 6, 3]]))
    assert n_dur[0, :3].tolist() == [2, 6, 3] and n_rest[0, :3].tolist() == [1.0, 0.0, 1.0]


def test_held_and_resung_differ_only_in_phonemes():
    e = make_encoder()
    held, resung = e.streams(*wan(HOLD)), e.streams(*wan(AN))
    for a, b in zip(held[2:], resung[2:]):
        assert torch.equal(a, b)                                  # same note stream
    assert not torch.equal(held[1], resung[1])                    # one "an" vs two
    with torch.no_grad():
        assert not torch.allclose(e(*wan(HOLD)), e(*wan(AN)))


def test_hold_token_never_embedded():
    e = make_encoder()
    with torch.no_grad():
        e.phon_emb.weight[HOLD] = float("nan")
        assert torch.isfinite(e(*wan(HOLD))).all()


def test_batch_padding_matches_single():
    e = make_encoder()
    ids_a, pitch_a, durs_a = wan(HOLD)
    ids_b = torch.tensor([[PAU, K, AN, PAU, 0]])
    pitch_b = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.0]])
    durs_b = torch.tensor([[4, 2, 6, 3, 0]])
    with torch.no_grad():
        both = e(torch.cat([ids_a, ids_b]), torch.cat([pitch_a, pitch_b]), torch.cat([durs_a, durs_b]))
        a = e(ids_a, pitch_a, durs_a)
        b = e(ids_b[:, :4], pitch_b[:, :4], durs_b[:, :4])
    assert torch.allclose(both[0], a[0], atol=1e-6)
    assert torch.allclose(both[1, :, :15], b[0], atol=1e-6)


def test_encoder_exports_with_dynamic_frames(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    e = make_encoder()

    class Wrap(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, ids, pitch, durs, noise):
            return self.enc(ids, pitch, durs, frames=noise[:, 0, :])

    ids, pitch, durs = wan(HOLD)
    path = str(tmp_path / "enc.onnx")
    names = ["ids", "pitch", "durs", "noise"]
    torch.onnx.export(Wrap(e), (ids, pitch, durs, torch.zeros(1, 2, 30)), path, opset_version=17,
                      input_names=names, output_names=["feat"], dynamo=False,
                      dynamic_axes={"ids": {1: "L"}, "pitch": {1: "L"}, "durs": {1: "L"},
                                    "noise": {2: "T"}, "feat": {2: "T"}})
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    longer = (torch.tensor([[PAU, K, AN, HOLD, HOLD, W, AN, PAU]]),
              torch.tensor([[0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.0, 0.0]]),
              torch.tensor([[7, 2, 9, 4, 6, 3, 12, 5]]))
    for ids, pitch, durs in (wan(HOLD), wan(AN), longer):
        T = int(durs.sum())
        with torch.no_grad():
            ref = e(ids, pitch, durs).numpy()
        got = sess.run(["feat"], {"ids": ids.numpy(), "pitch": pitch.numpy(), "durs": durs.numpy(),
                                  "noise": np.zeros((1, 2, T), np.float32)})[0]
        assert got.shape == ref.shape and np.abs(got - ref).max() < 1e-5


def test_default_encoder_and_checkpoints_unchanged():
    from train import build_model, encoder_kwargs
    cfg = {"n_phonemes": 4, "n_speakers": 1, "spk_dim": 32, "n_styles": 1, "n_steps": 10,
           "phonemes": PHONEMES[:4], "nucleus": ["an"]}
    assert encoder_kwargs(cfg) == {}
    legacy = build_model(cfg)
    assert isinstance(legacy.encoder, InputEncoder)
    notes = build_model({**cfg, "n_phonemes": 5, "phonemes": PHONEMES, "nucleus": ["an", "-"],
                         "encoder": "notes", "hold_symbol": "-"})
    assert isinstance(notes.encoder, NoteInputEncoder)
    assert notes.encoder.hold_id == HOLD and notes.encoder.is_nucleus.tolist() == [False, False, True, False, False]
    # same parameter names outside the encoder, so the rest of a checkpoint maps one to one
    rest = lambda m: sorted(k for k in m.state_dict() if not k.startswith("encoder."))  # noqa: E731
    assert rest(legacy) == rest(notes)
    with pytest.raises(ValueError):
        build_model({**cfg, "encoder": "notes", "hold_symbol": "~"})
