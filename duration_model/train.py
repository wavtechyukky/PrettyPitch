"""子音長推定モデル 学習スクリプト（score-free）。

slab/ust/xml は使わない。PrettyPitch と同じ preprocess の F0 シャードを使い、
ピッチは dataloader が F0 の代表値（ビブラート除去後）で決める。CPU で数分。

Usage:
    python -m duration_model.train --data_dirs data/oniku data/natsume data/ritsu \
        --run_name cons_dur --config configs/ja_example.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import phoneset
from .dataset import ConsonantDurDataset
from .model import ConsonantDurModel
from prettypitch.config import merge_per_db_from_recipes



def _force_devib(pitch_cfg: dict) -> dict:
    """子音長モデルはピッチを常にビブラート除去後の代表値で採る。レシピが DB 別に
    devib_representative: false を書いていてもここでは効かせない（per_db の方が
    優先されるため、per_db 側も潰す必要がある）。"""
    pc = dict(pitch_cfg)
    pc['devib_representative'] = True
    pc['per_db'] = {db: {**(ov or {}), 'devib_representative': True}
                    for db, ov in (pc.get('per_db') or {}).items()}
    return pc

def _dict_relative(p) -> str | None:
    """音素表のパスを dict/ からの相対に畳む（ckpt へ焼き込む用）。

    dict/ の外にあるファイルはファイル名だけを残す。絶対パスは決して返さない:
    配布 ckpt に作者のホームディレクトリが載るのを防ぐため。
    """
    if not p:
        return None
    p = Path(p)
    try:
        return str(p.resolve().relative_to(phoneset._DICT_DIR))
    except ValueError:
        return p.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='子音長推定モデル 学習（score-free）')
    p.add_argument('--data_dirs', nargs='+', required=True,
                   help='PrettyPitch preprocess のシャードディレクトリ（複数可）')
    p.add_argument('--config', default='configs/ja_example.yaml',
                   help='pitch / decomp 設定を読む PrettyPitch config')
    p.add_argument('--run_name', default='cons_dur', help='TensorBoard / ckpt ログ名')
    p.add_argument('--min_sec', type=float, default=0.5, help='この長さ未満のフレーズは除外')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=30, help='early stopping patience (epochs)')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--wide_context', action='store_true',
                   help='音素の文脈を前後 2 音素まで広げる（既定は前後 1）')
    p.add_argument('--no_cluster_len', action='store_true',
                   help='子音連結の長さ特徴を使わない（従来と同一の入力に戻す）')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lang', default='ja',
                   help="言語パック名またはパス（dict/<lang>.yaml）")
    p.add_argument('--phonemes', default=None, help='音素表を上書き（言語パックの phonemes:）')
    p.add_argument('--nucleus', default=None, help='核音素表を上書き（言語パックの nucleus:）')
    p.add_argument('--ckpt_dir', default='logs', help='チェックポイント保存先')
    p.add_argument('--scatter_every', type=int, default=10,
                   help='何 epoch ごとに散布図を TensorBoard に書くか')
    return p.parse_args()


def make_scatter_figure(gt_ms: np.ndarray, pred_ms: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gt_ms, pred_ms, s=4, alpha=0.3, rasterized=True)
    lim = max(gt_ms.max(), pred_ms.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1)
    ax.set_xlabel('GT [ms]')
    ax.set_ylabel('Pred [ms]')
    ax.set_title('Consonant Duration: GT vs Pred')
    fig.tight_layout()
    return fig


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    gt_list, pred_list = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model.loss(batch)
        total_loss += loss.item() * len(batch['target_dur'])
        pred_list.append(model.predict(batch).cpu().numpy())
        gt_list.append(batch['target_dur'].cpu().numpy())
    n = sum(len(x) for x in gt_list)
    avg_loss = total_loss / n
    gt_ms = np.concatenate(gt_list) * 1000
    pred_ms = np.concatenate(pred_list) * 1000
    mae_ms = float(np.abs(gt_ms - pred_ms).mean())
    return avg_loss, mae_ms, gt_ms, pred_ms


def main():
    matplotlib.use('Agg')          # スクリプト実行時だけ（import 側の描画を壊さない）
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ps = phoneset.set_default(phoneset.load(args.lang, args.phonemes, args.nucleus))
    print(f'言語パック: {ps.lang}  音素{ps.n_phonemes} / 核{len(ps.nuclei)} / 子音{ps.n_consonants}')
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    for _k in ('pitch', 'decomp', 'data'):
        cfg.setdefault(_k, {})
    merge_per_db_from_recipes(cfg, args.data_dirs)  # DB別ハイパラはレシピが持つ
    pitch_cfg = _force_devib(cfg['pitch'])       # 子音長: ピッチは常にビブラート除去後の代表値
    decomp_cfg = dict(cfg.get('decomp', {}))
    eval_songs = int(cfg.get('data', {}).get('eval_songs', 2))
    eval_dbs = cfg.get('data', {}).get('eval_dbs')   # F0 モデル（train.py）と同じ: 指定した DB からだけ評価の曲を抜く
    nucleus_names = sorted(ps.nuclei)

    # 話者/スタイル（F0 モデルと同じ spk_map / style_map / n_styles を共用）
    spk_map = dict(cfg.get('data', {}).get('spk_map') or {})
    style_map = dict(cfg.get('data', {}).get('style_map') or {})
    n_speakers = (max(spk_map.values()) + 1) if spk_map else 1
    n_styles = int(cfg.get('model', {}).get('n_styles',
                                            (max(style_map.values()) + 1) if style_map else 0))

    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ckpt_dir / 'tb'

    common = dict(eval_songs=eval_songs, eval_dbs=eval_dbs, min_sec=args.min_sec, seed=args.seed,
                  nucleus_names=nucleus_names, pitch_cfg=pitch_cfg, decomp_cfg=decomp_cfg,
                  spk_map=spk_map, style_map=style_map)
    trn_dataset = ConsonantDurDataset(args.data_dirs, split='train', **common)
    val_dataset = ConsonantDurDataset(args.data_dirs, split='eval', **common)
    print(f'train samples: {len(trn_dataset)}  |  val samples: {len(val_dataset)}')

    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    use_cluster_len = not args.no_cluster_len
    model = ConsonantDurModel(hidden=args.hidden, dropout=args.dropout,
                              n_speakers=n_speakers, n_styles=n_styles,
                              use_cluster_len=use_cluster_len,
                              use_wide_context=args.wide_context).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'parameters: {n_params:,}  (speakers={n_speakers}, styles={n_styles}, '
          f'cluster_len={use_cluster_len}, wide_context={args.wide_context})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        trn_loss_sum = 0.0
        for batch in tqdm(trn_loader, desc=f'epoch {epoch:03d} [train]', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            trn_loss_sum += loss.item() * len(batch['target_dur'])

        trn_loss = trn_loss_sum / len(trn_dataset)
        scheduler.step()

        val_loss, val_mae, gt_ms, pred_ms = evaluate(model, val_loader, device)

        writer.add_scalar('loss/train', trn_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('mae_ms/val', val_mae, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        if epoch % args.scatter_every == 0:
            fig = make_scatter_figure(gt_ms, pred_ms)
            writer.add_figure('scatter/val', fig, epoch)
            plt.close(fig)

        print(f'epoch {epoch:03d}  trn={trn_loss:.4f}  val={val_loss:.4f}'
              f'  mae={val_mae:.1f}ms  lr={scheduler.get_last_lr()[0]:.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'config': {'hidden': args.hidden,
                           'n_speakers': n_speakers, 'n_styles': n_styles,
                           'use_cluster_len': use_cluster_len,
                           'use_wide_context': bool(args.wide_context),
                           'spk_map': spk_map, 'style_map': style_map,
                           # 推論で言語を取り違えないよう音素表を ckpt に焼き込む
                           'lang': ps.lang,
                           'n_phonemes': ps.n_phonemes,
                           'phonemes': list(ps.phonemes),
                           'nucleus': sorted(ps.nuclei),
                           # パスは dict/ からの相対で記録する。絶対パスを焼くと
                           # (1) 配布した ckpt に作者のホームディレクトリ（＝個人情報）が載り、
                           # (2) 他マシンでそのパスが存在せず言語パックの読み込みに失敗して
                           #     既定の日本語に化ける。相対なら配布先の dict/ を正しく指す。
                           'phonemes_path': _dict_relative(ps.phonemes_path),
                           'nucleus_path': _dict_relative(ps.nucleus_path)},
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae_ms': val_mae,
                'args': vars(args),
            }, ckpt_dir / 'best.ckpt')
            print(f'  → best checkpoint saved (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(no improvement for {args.patience} epochs)')
                break

    writer.close()
    print(f'\nDone. Best val_loss={best_val_loss:.4f}')
    print(f'Checkpoint: {ckpt_dir / "best.ckpt"}')


if __name__ == '__main__':
    main()
