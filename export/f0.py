"""F0 モデル（DDPM / rectified flow 共通）の ONNX エクスポート。

    python -m export.f0 --ckpt checkpoints/f0_3singer.pt --out export/out/f0_3singer.onnx
    python -m export.f0 --ckpt log/<reflow_run>/ckpt_010000.pt --out export/out/f0_reflow.onnx
    python -m export.f0 --bench export/out/f0_3singer.onnx --seconds 240     # メモリ・時間の計測

グラフに入れるもの = `model.sample` と同じ計算（長さ展開 → conditioner → 話者/スタイル →
サンプラ → v/uv ヘッド）。**baseline とピッチ補正はホスト側**（README の方針）。

入力（B は常に 1）:
  tokens     [1, L] int64    音素 id（metadata の `phonemes` の行順）
  pitch_midi [1, L] float32  核の音高 [MIDI、連続値可] を後続の音素へ前方埋めした階段
                             （次の核で変わる。pau など先頭で値が無い所は 0 でよい＝正規化後 −5）
  durations  [1, L] int64    音素ごとのフレーム数（各 ≥ 1、合計 = T）
  spk_id     [1]    int64
  style_mix  [1, n_styles] float32   one-hot なら style_id と同じ
  noise      [1, K, 2, T] float32    標準正規乱数。K = DDPM は n_steps、reflow は 1。
                                     ホストが乱数を握るので seed 再現はホストの責任
出力:
  plain    [1, T] float32   log2 領域の baseline からのずれ（plain_std を掛けた後）
  vibrato  [1, T] float32   同上（vib_std を掛けた後）
  vuv      [1, T] float32   有声確率 [0,1]

ホスト側の復元:  f0 = 2 ** (baseline_log2 + plain_scale * plain + vib_scale * vibrato)
                 voiced = (vuv > 0.5) & ~pau。ピッチ補正（Blaauw + 非対称 Tukey）は plain にだけ。
                 baseline は音高の階段を σ = baseline_sigma_sec のガウシアンで平滑したもの。

検証: (1) ラッパー(torch) と `model.sample`（seed 固定）がビット一致 — 長さ展開を書き換えても
計算が同じであること。(2) ORT と torch ラッパーを複数の長さで比較。(3) --bench で ORT の
ピーク RSS と時間を長さごとに測る（NHVSing / LeapSinger で起きた「中間テンソルが長さに
比例して爆発する」事故の再発防止。この設計は 1D 畳み込みの積み重ねで、長さ×音素数 の
2 次元テンソルを作らない）。
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from export.cons_dur import scan_bytes  # noqa: E402

OPSET = 17
INPUT_NAMES = ['tokens', 'pitch_midi', 'durations', 'spk_id', 'style_mix', 'noise']
OUTPUT_NAMES = ['plain', 'vibrato', 'vuv']


# ── ラッパー ─────────────────────────────────────────────────────────────────

class F0Wrapper(nn.Module):
    """_F0Base（DDPM / reflow）を ONNX 向けの固定入出力に平たくする。

    model.sample と違うのは (a) 長さ展開を Python ループ無しで書く (b) 乱数を入力で受ける、の 2 点。
    それ以外は同じモジュールを同じ順に呼ぶ。
    """

    def __init__(self, model, cfg: dict):
        super().__init__()
        self.m = model
        self.is_reflow = hasattr(model, 'predict_velocity')
        self.center = float(cfg['pitch_center'])
        self.scale = float(cfg['pitch_scale'])
        self.plain_std = float(cfg['plain_std'])
        self.vib_std = float(cfg['vib_std'])
        self.n_noise = 1 if self.is_reflow else int(model.n_steps)

    # InputEncoder.forward の B=1・ループ無し版
    def _frame_features(self, tokens, pitch_midi, durations, noise):
        enc = self.m.encoder
        pitch = (pitch_midi - self.center) / self.scale
        pe = enc.phon_emb(tokens)                                     # [1, L, 64]
        pp = enc.pitch_proj(pitch.unsqueeze(-1))                      # [1, L, 32]
        logdur = torch.log1p(durations.float()).unsqueeze(-1)         # [1, L, 1]
        phon_feat = torch.cat([pe, pp, logdur], dim=-1)               # [1, L, D]

        ends = torch.cumsum(durations, dim=1)                         # [1, L]
        starts = ends - durations
        onehot = torch.zeros_like(noise[:, 0, 0, :])                  # [1, T]（T は noise から）
        onehot = onehot.scatter(1, starts, torch.ones_like(starts, dtype=onehot.dtype))
        idx = torch.cumsum(onehot, dim=1).long() - 1                  # フレーム → 音素 index
        t = torch.cumsum(torch.ones_like(onehot), dim=1) - 1.0        # 0, 1, 2, ...
        frame_feat = torch.gather(phon_feat, 1, idx.unsqueeze(-1).expand(-1, -1, phon_feat.size(-1)))
        d = torch.gather(durations, 1, idx).float()
        s = torch.gather(starts, 1, idx).float()
        pos = (t - s) / torch.clamp(d - 1.0, min=1.0)                 # linspace(0,1,d) と同値
        frame_feat = torch.cat([frame_feat, pos.unsqueeze(-1)], dim=-1)
        return enc.proj(frame_feat.transpose(1, 2))                   # [1, 128, T]

    def _cond(self, tokens, pitch_midi, durations, spk_id, style_mix, noise):
        m = self.m
        c = m.conditioner(self._frame_features(tokens, pitch_midi, durations, noise))
        if m.spk_bank is not None:
            c = c + m.spk_proj(m.spk_bank(spk_id)).unsqueeze(-1)
        if m.style_emb is not None:
            c = c + (style_mix @ m.style_emb.weight).unsqueeze(-1)
        return c

    def _sample(self, cond, noise):
        m = self.m
        x = noise[:, 0]
        if self.is_reflow:
            steps = int(m.num_steps)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((1,), i * dt, dtype=x.dtype, device=x.device)
                x = x + m.predict_velocity(x, cond, t) * dt
            return x
        k = 1
        for i in reversed(range(int(m.n_steps))):
            t = torch.full((1,), i, dtype=torch.long, device=x.device)
            x0 = m.predict_x0(x, cond, t)
            ab = m.alpha_bar[i]
            ab_prev = m.alpha_bar[i - 1] if i > 0 else torch.ones((), device=x.device)
            a, b = m.alphas[i], m.betas[i]
            mu = (torch.sqrt(ab_prev) * b / (1 - ab)) * x0 + (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                sigma = torch.sqrt(b * (1 - ab_prev) / (1 - ab))
                x = mu + sigma * noise[:, k]
                k += 1
            else:
                x = x0
        return x

    def forward(self, tokens, pitch_midi, durations, spk_id, style_mix, noise):
        cond = self._cond(tokens, pitch_midi, durations, spk_id, style_mix, noise)
        x = self._sample(cond, noise)                                 # [1, 2, T]
        vuv = torch.sigmoid(self.m.uv_head(cond))[:, 0]               # [1, T]
        return x[:, 0] * self.plain_std, x[:, 1] * self.vib_std, vuv


# ── 入力の生成 ───────────────────────────────────────────────────────────────

def make_noise(n_noise: int, T: int, seed: int) -> torch.Tensor:
    """model.sample が seed 固定で引く乱数と同じ順（初期 randn → 各 step の randn_like）。"""
    torch.manual_seed(seed)
    draws = [torch.randn(1, 2, T) for _ in range(n_noise)]
    return torch.stack(draws, dim=1)                                  # [1, K, 2, T]


def synthetic_phrase(cfg: dict, seconds: float, seed: int = 0):
    """検証・計測用の適当なフレーズ（pau + 子音/核の繰り返し）。"""
    rng = np.random.default_rng(seed)
    fr = float(cfg['frame_rate'])
    phonemes = list(cfg['phonemes']); nuc = [p for p in cfg['nucleus'] if p in phonemes]
    cons = [p for p in phonemes if p not in nuc and p != 'pau'][:20]
    ph2id = {p: i for i, p in enumerate(phonemes)}
    toks, durs, pitch = [ph2id['pau']], [int(0.25 * fr)], [0.0]
    total = durs[0]; cur = 60.0
    while total < seconds * fr:
        c = cons[rng.integers(len(cons))]; v = nuc[rng.integers(len(nuc))]
        cur = float(np.clip(cur + rng.integers(-3, 4), 52, 76))
        dc = int(max(1, rng.normal(0.07, 0.02) * fr)); dv = int(max(1, rng.normal(0.35, 0.15) * fr))
        toks += [ph2id[c], ph2id[v]]; durs += [dc, dv]; pitch += [pitch[-1], cur]
        total += dc + dv
    toks.append(ph2id['pau']); durs.append(int(0.25 * fr)); pitch.append(pitch[-1])
    return (torch.tensor([toks]), torch.tensor([pitch], dtype=torch.float32), torch.tensor([durs]))


def example_inputs(wrapper: F0Wrapper, cfg: dict, seconds: float, seed: int = 0, spk: int = 2,
                   style_mix=None):
    # 話者の番号はモデルの話者の数に収める（既定の 2 は 3 話者の配布モデル用。1 話者のモデルでは 0）
    bank = getattr(wrapper.m, 'spk_bank', None)
    if bank is not None:
        spk = min(int(spk), bank.num_embeddings - 1)
    tokens, pitch, durs = synthetic_phrase(cfg, seconds, seed)
    T = int(durs.sum())
    n_styles = int(cfg.get('n_styles', 0))
    sm = torch.zeros(1, max(1, n_styles)); sm[0, 0] = 1.0
    if style_mix is not None:
        sm = torch.tensor([style_mix], dtype=torch.float32)
    return (tokens, pitch, durs, torch.tensor([spk]), sm, make_noise(wrapper.n_noise, T, seed))


# ── 検証 ───────────────────────────────────────────────────────────────────

def verify_against_sample(wrapper: F0Wrapper, cfg: dict, seconds: float, seed: int) -> float:
    """ラッパー(torch) と model.sample（同 seed）の最大差。長さ展開の書き換えが正しいことの証明。"""
    tokens, pitch, durs, spk, sm, noise = example_inputs(wrapper, cfg, seconds, seed)
    with torch.no_grad():
        ref_pitch = (pitch - wrapper.center) / wrapper.scale
        torch.manual_seed(seed)
        ref = wrapper.m.sample(tokens, ref_pitch, durs, spk_id=spk, style_mix=sm)[0]   # [3, T]
        out = wrapper(tokens, pitch, durs, spk, sm, noise)
        got = torch.stack([out[0][0] / wrapper.plain_std, out[1][0] / wrapper.vib_std, out[2][0]])
    return float((ref - got).abs().max())


def verify_ort(wrapper: F0Wrapper, cfg: dict, onnx_path: str, seconds_list, seed: int = 0):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    rows = []
    for s in seconds_list:
        ex = example_inputs(wrapper, cfg, s, seed)
        with torch.no_grad():
            ref = [t[0].numpy() for t in wrapper(*ex)]
        got = sess.run(OUTPUT_NAMES, {n: t.numpy() for n, t in zip(INPUT_NAMES, ex)})
        diffs = [float(np.abs(g[0] - r).max()) for g, r in zip(got, ref)]
        rows.append((s, int(ex[2].sum()), diffs))
    return rows


# ── 計測 ───────────────────────────────────────────────────────────────────

def bench(onnx_path: str, seconds: float, repeats: int, threads: int | None, torch_ckpt: str | None):
    """1 プロセスで ORT を回し、ピーク RSS（ru_maxrss）と時間を報告する。呼び出し側で
    子プロセスにして上限付きで監視すること（scratch の mem_watch など）。"""
    import onnxruntime as ort
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    md = {p.key: p.value for p in m.metadata_props}
    cfg = json.loads(md['config'])
    K = int(md['n_noise'])
    fr = float(cfg['frame_rate'])
    tokens, pitch, durs = synthetic_phrase(cfg, seconds, 0)
    T = int(durs.sum())
    sm = np.zeros((1, max(1, int(cfg.get('n_styles', 0)))), np.float32); sm[0, 0] = 1
    spk = min(2, max(1, int(cfg.get('n_speakers', 3))) - 1)      # 3 話者の配布モデルは 2、1 話者は 0
    feed = {'tokens': tokens.numpy(), 'pitch_midi': pitch.numpy(), 'durations': durs.numpy(),
            'spk_id': np.array([spk], np.int64), 'style_mix': sm,
            'noise': np.random.default_rng(0).standard_normal((1, K, 2, T), dtype=np.float32)}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
    sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); sess.run(OUTPUT_NAMES, feed); times.append(time.perf_counter() - t0)
    rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    sec = T / fr
    print(f'[bench ORT] {onnx_path}  {sec:.1f}s / {T} frames / L={tokens.shape[1]} / K={K}  '
          f'run median {np.median(times):.3f}s (RTF {np.median(times)/sec:.4f})  '
          f'peak RSS: before {rss0:.0f} MB → session {rss1:.0f} MB → after run {rss2:.0f} MB  '
          f'(+{rss2 - rss1:.0f} MB for the run, {(rss2 - rss1) / sec:.2f} MB/s)', flush=True)
    if torch_ckpt:
        import infer
        model, tcfg = infer.load_model(torch_ckpt, 'cpu')
        w = F0Wrapper(model, tcfg).eval()
        ex = (tokens, pitch, durs, torch.tensor([spk]), torch.from_numpy(sm), torch.from_numpy(feed['noise']))
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        t0 = time.perf_counter()
        with torch.no_grad():
            w(*ex)
        dt = time.perf_counter() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f'[bench torch] {sec:.1f}s  run {dt:.3f}s (RTF {dt/sec:.4f})  peak RSS {r0:.0f} → {r1:.0f} MB '
              f'(+{r1 - r0:.0f} MB)', flush=True)


# ── メタデータ ───────────────────────────────────────────────────────────────

def metadata(cfg: dict, wrapper: F0Wrapper, ckpt_path: str, maps: dict) -> dict:
    return {
        'model': 'prettypitch-f0',
        'arch': cfg['arch'],
        'sampler': 'reflow (Euler, %d step)' % int(wrapper.m.num_steps) if wrapper.is_reflow
                   else 'ddpm (ancestral, %d steps, x0 prediction)' % int(wrapper.m.n_steps),
        'n_noise': wrapper.n_noise,
        'inputs': INPUT_NAMES,
        'outputs': OUTPUT_NAMES,
        'input_spec': {
            'tokens': '[1, L] int64 phoneme ids (row order of `phonemes`)',
            'pitch_midi': '[1, L] float32 nucleus pitch in MIDI (fractional ok), forward-filled to '
                          'the following phonemes; 0 where no nucleus has occurred yet',
            'durations': '[1, L] int64 frames per phoneme, each >= 1, sum = T',
            'spk_id': '[1] int64',
            'style_mix': '[1, n_styles] float32 (one-hot = style_id)',
            'noise': '[1, n_noise, 2, T] float32 standard normal',
        },
        'output_spec': {
            'plain': '[1, T] log2 deviation from the baseline (plain_std already applied)',
            'vibrato': '[1, T] log2 deviation (vib_std already applied)',
            'vuv': '[1, T] voicing probability',
        },
        'host_reconstruction': 'f0_hz = 2 ** (baseline_log2 + plain_scale * plain + vib_scale * vibrato); '
                               'voiced = (vuv > 0.5) & not pau. baseline_log2 = log2 of the note-pitch '
                               'staircase (Hz) smoothed by a Gaussian of baseline_sigma_sec. '
                               'Pitch correction (Blaauw stage + asymmetric Tukey stage) applies to plain only.',
        'correction_defaults': {'stage1': 'blaauw constant offset per note', 'hann_alpha': 0.8,
                                'hann_window': 'tukey', 'hann_lead_taper': 0.6, 'hann_trail_taper': 0.1},
        'ort_session_hint': 'peak memory grows ~4 MB per second of audio (linear); '
                            'SessionOptions.enable_mem_pattern = False cuts it by ~40% at no speed cost',
        'frame_rate': float(cfg['frame_rate']), 'sample_rate': int(cfg['sample_rate']), 'hop': int(cfg['hop']),
        'pitch_center': float(cfg['pitch_center']), 'pitch_scale': float(cfg['pitch_scale']),
        'plain_std': float(cfg['plain_std']), 'vib_std': float(cfg['vib_std']),
        'baseline_sigma_sec': float(cfg.get('baseline_sigma_sec', 0.02)),
        'phonemes': list(cfg['phonemes']), 'nucleus': list(cfg['nucleus']),
        'n_speakers': int(cfg.get('n_speakers', 0)), 'n_styles': int(cfg.get('n_styles', 0)),
        'spk_map': maps.get('spk_map', {}), 'style_map': maps.get('style_map', {}),
        'lang': cfg.get('lang'),
        'source_ckpt': Path(ckpt_path).name,
        'opset': OPSET,
        'config': {k: v for k, v in cfg.items() if k not in ('decomp_cfg', 'pitch_cfg')},
    }


def load_maps(config_yaml: Path) -> dict:
    try:
        import yaml
        d = yaml.safe_load(config_yaml.read_text(encoding='utf-8')).get('data', {})
        return {'spk_map': dict(d.get('spk_map') or {}), 'style_map': dict(d.get('style_map') or {})}
    except Exception:                                                 # noqa: BLE001
        return {}


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', help='F0 ckpt（slim でも学習 ckpt でも可）')
    ap.add_argument('--out', help='書き出す .onnx。隣に <out>.json も書く')
    ap.add_argument('--config', default=str(_ROOT / 'configs' / 'ja_example.yaml'),
                    help='spk_map / style_map を metadata に入れるための学習 config')
    ap.add_argument('--opset', type=int, default=OPSET)
    ap.add_argument('--no-simplify', action='store_true')
    ap.add_argument('--no-scan', action='store_true')
    ap.add_argument('--bench', help='既存の .onnx を計測するだけ（export しない）')
    ap.add_argument('--seconds', type=float, default=60.0, help='--bench のフレーズ長 [s]')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--torch-ckpt', default=None, help='--bench で torch 側も同じ入力で測る')
    args = ap.parse_args()

    if args.bench:
        bench(args.bench, args.seconds, args.repeats, args.threads, args.torch_ckpt)
        return
    if not (args.ckpt and args.out):
        ap.error('--ckpt と --out が要ります（計測だけなら --bench）')

    import infer
    model, cfg = infer.load_model(args.ckpt, 'cpu')
    wrapper = F0Wrapper(model, cfg).eval()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f'arch={cfg["arch"]}  conditioner={cfg["conditioner"]}  n_noise={wrapper.n_noise}  '
          f'n_speakers={cfg.get("n_speakers")}  n_styles={cfg.get("n_styles")}')

    # (1) ラッパー vs model.sample（seed 固定）— 長さ展開の書き換えが正しいか
    for s in (2.0, 7.0):
        d = verify_against_sample(wrapper, cfg, s, seed=0)
        print(f'[verify] wrapper vs model.sample ({s:.0f}s): 最大差 {d:.2e}')
        if d > 1e-5:
            print('[verify] WARNING: ラッパーが model.sample と一致しません'); sys.exit(2)

    ex = example_inputs(wrapper, cfg, 3.0, seed=0)
    dyn = {'tokens': {1: 'L'}, 'pitch_midi': {1: 'L'}, 'durations': {1: 'L'},
           'style_mix': {}, 'spk_id': {}, 'noise': {3: 'T'},
           'plain': {1: 'T'}, 'vibrato': {1: 'T'}, 'vuv': {1: 'T'}}
    torch.onnx.export(
        wrapper, ex, str(out), opset_version=args.opset,
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        dynamic_axes=dyn, do_constant_folding=True, dynamo=False,
    )

    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            m, ok = simplify(onnx.load(str(out)))
            if ok:
                onnx.save(m, str(out)); print('[onnxsim] simplified')
            else:
                print('[onnxsim] check failed; keeping the unsimplified graph')
        except Exception as e:                                       # noqa: BLE001
            print(f'[onnxsim] skipped ({e.__class__.__name__}: {e})')

    import onnx
    md = metadata(cfg, wrapper, args.ckpt, load_maps(Path(args.config)))
    side = out.with_suffix('.json')
    side.write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding='utf-8')
    m = onnx.load(str(out))
    m.producer_name = 'prettypitch'
    m.doc_string = 'PrettyPitch F0 model: plain / vibrato (log2 deviations) and v/uv from phonemes, durations and note pitch'
    del m.metadata_props[:]
    for k, v in md.items():
        p = m.metadata_props.add(); p.key = k
        p.value = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    onnx.save(m, str(out))
    onnx.checker.check_model(onnx.load(str(out)))

    # (2) ORT vs torch ラッパー
    rows = verify_ort(wrapper, cfg, str(out), seconds_list=(0.6, 3.0, 12.0, 40.0))
    worst = 0.0
    for s, T, diffs in rows:
        worst = max(worst, max(diffs))
        print(f'[verify] ORT vs torch {s:5.1f}s ({T:5d} frames): plain {diffs[0]:.2e}  '
              f'vibrato {diffs[1]:.2e}  vuv {diffs[2]:.2e}   (log2 units / probability)')
    rc = 0
    if worst > 2e-3:
        print('[verify] WARNING: ORT と torch の差が大きい'); rc = 2

    if not args.no_scan:
        needles = [os.path.expanduser('~').encode(), str(Path(args.ckpt).resolve().parent).encode()]
        hits = scan_bytes(str(out), needles)
        if hits:
            print(f'[scan] NG: 焼かれてはいけない文字列があります: {hits}'); rc = max(rc, 3)
        else:
            print('[scan] OK: 絶対パス・メールアドレスなし')

    print(f'wrote {out} ({out.stat().st_size / 1e6:.2f} MB) + {side.name}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
