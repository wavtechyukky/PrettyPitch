"""3ch F0 diffusion model (plain / vibrato / v-uv) — DiffWave-style DDPM.

Design notes — what shapes this model:

  1. **Continuous pitch input.** A single ``Linear(1, 32)`` over the note pitch — a
     normalized continuous value (typically ``(midi − 60) / 12``) forward-filled from each
     nucleus to every phoneme (a consonant keeps the previous nucleus's pitch; it changes
     only at the next nucleus). No has-flag, no nucleus/consonant split. The condition is
     the conditioner output alone → cond dim 256 (no per-frame pitch concat).
  2. **Speaker / style** (LeapSinger pattern): ``spk_bank Embedding(n_speakers, spk_dim=32)``
     + ``spk_proj Linear(spk_dim, 256)`` and ``style_emb Embedding(n_styles, 256)``, each
     added time-invariantly to the conditioner output. ``style_mix [B, n_styles]`` blends
     styles continuously at inference.
  3. Frame = hop/sr from the data (256/44100 by default).
  4. **Conditioner = ConvNeXt-1D** (dilations from ``model.convnext_dilations``, default
     [1,5,12,24] → RF ≈ 253 frames) over the InputEncoder's frame features; denoiser =
     DiffWave-standard [1,2,4,...,128]×2 (16 gated blocks). The denoiser's receptive
     field is ≈1021 frames ≈ 5.9 s @172.27 fps — ample for 4.5-7.5 Hz vibrato and
     note-scale intonation. Checkpoints trained with other stacks cannot load.
  5. **Encoder** (``model.encoder``): ``phonemes`` (default, InputEncoder) reads one pitch
     per phoneme. ``notes`` (NoteInputEncoder) takes the same inputs but rebuilds them into
     a phoneme stream and a note stream, so a vowel held over several notes (written by
     the host as the vowel + hold tokens) stays one phoneme. Same ONNX inputs and outputs.

Reverse process: DDPM (``arch: diffusion``, default 10 ancestral steps, cosine β schedule,
**x0 direct prediction**) or rectified flow (``arch: reflow``, 1 Euler step). Both share
the backbone; the diffusion channels (normalized x0, each std≈1):

  [0] plain   : low-frequency log2-F0 deviation from the baseline (intonation) ÷ plain_std
  [1] vibrato : 4.5-7.5 Hz vibrato / fine prosody ÷ vib_std

v/uv is NOT diffused — it is a separate deterministic classifier head (`uv_head`,
256-dim context → logit, BCE). sample() returns [B,3,T], channel 2 = sigmoid(logit) in [0,1].

Reconstruction (caller side):
  logf0(log2) = baseline + plain·plain_std + vibrato·vib_std;  f0[Hz] = 2**logf0 · (uv_prob>0.5)

Old hop-512 checkpoints can NOT warm-start this model (the pitch input representation
changed) — train from scratch.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from prettypitch.frontend import frames_for, length_regulate, position_in_token

OUT_CH = 2          # diffusion/reflow channels: plain, vibrato (v/uv is a separate head)
STEP_EMB = 64        # diffusion step embedding dim
COND_DIM = 256       # conditioner output (no frame-pitch concat)


def cosine_betas(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine schedule (Nichol & Dhariwal 2021)."""
    steps = torch.linspace(0.0, 1.0, n_steps + 1)
    f = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    ab = f / f[0]
    return (1.0 - ab[1:] / ab[:-1]).clamp(1e-4, 0.9999)


class DiffusionStepEmbedding(nn.Module):
    """DiffWave-standard sinusoidal table → MLP. Output STEP_EMB(64)-dim."""

    def __init__(self, n_steps: int, emb_dim: int = STEP_EMB, proj_dim: int = STEP_EMB):
        super().__init__()
        half = emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        steps = torch.arange(n_steps).unsqueeze(1) * freqs.unsqueeze(0) * 50.0
        table = torch.cat([torch.sin(steps), torch.cos(steps)], dim=1)  # [n_steps, emb_dim]
        self.register_buffer("table", table)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.PReLU(proj_dim),
                                 nn.Linear(proj_dim, proj_dim), nn.PReLU(proj_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B] long
        return self.mlp(self.table[t])  # [B, proj_dim]


class ContinuousTimeEmbedding(nn.Module):
    """Rectified-flow time embedding: continuous t in [0,1] → sinusoidal → MLP."""

    def __init__(self, emb_dim: int = STEP_EMB, proj_dim: int = STEP_EMB):
        super().__init__()
        half = emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.PReLU(proj_dim),
                                 nn.Linear(proj_dim, proj_dim), nn.PReLU(proj_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B] float in [0,1]
        args = t[:, None] * self.freqs[None, :] * 1000.0
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=1))


class GatedResBlock(nn.Module):
    """Bidirectional gated residual block: dilated conv → tanh×sigmoid gate → res/skip.
    cond (per frame) and step (broadcast) are added before the gate."""

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1,
                 cond_dim: int = 0, step_dim: int = 0):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # 'same'
        self.dconv = nn.Conv1d(ch, 2 * ch, kernel, padding=pad, dilation=dilation)
        self.cond_proj = nn.Conv1d(cond_dim, 2 * ch, 1) if cond_dim > 0 else None
        self.step_proj = nn.Linear(step_dim, 2 * ch) if step_dim > 0 else None
        self.res_conv = nn.Conv1d(ch, ch, 1)
        self.skip_conv = nn.Conv1d(ch, ch, 1)

    def forward(self, x, cond=None, step_emb=None):
        h = self.dconv(x)                                   # [B, 2C, T]
        if self.step_proj is not None and step_emb is not None:
            h = h + self.step_proj(step_emb).unsqueeze(-1)
        if self.cond_proj is not None and cond is not None:
            h = h + self.cond_proj(cond)
        a, b = h.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        res = (x + self.res_conv(h)) * (2.0 ** -0.5)
        skip = self.skip_conv(h)
        return res, skip


class InputEncoder(nn.Module):
    """Phoneme id / note pitch / duration → frame-level features (length regulation).

    Per-phoneme feature = [phon_emb 64 | pitch_proj(note pitch) 32 | log1p(dur_frames) 1]
    → expanded to frames → + within-phoneme position ramp → 1×1 conv proj → [B, 128, T].
    """

    def __init__(self, n_phonemes: int, phon_dim: int = 64, pitch_dim: int = 32,
                 out_ch: int = 128):
        super().__init__()
        self.phon_emb = nn.Embedding(n_phonemes, phon_dim)
        self.pitch_proj = nn.Linear(1, pitch_dim)           # forward-filled note pitch
        feat_dim = phon_dim + pitch_dim + 2                 # + position ramp, + log dur
        self.proj = nn.Sequential(nn.Conv1d(feat_dim, out_ch, 1), nn.PReLU(out_ch),
                                  nn.Conv1d(out_ch, out_ch, 1))

    @staticmethod
    def length_regulate(phon_feat, durations):
        # phon_feat: [B, L, D], durations: [B, L] → [B, T, D]
        outs, max_len = [], int(durations.sum(dim=1).max().item())
        for b in range(phon_feat.size(0)):
            segs = [phon_feat[b, i:i + 1].expand(int(d), -1)
                    for i, d in enumerate(durations[b]) if int(d) > 0]
            seq = torch.cat(segs, dim=0) if segs else phon_feat.new_zeros(0, phon_feat.size(2))
            if seq.size(0) < max_len:
                seq = torch.cat([seq, seq.new_zeros(max_len - seq.size(0), seq.size(1))], 0)
            outs.append(seq)
        return torch.stack(outs, dim=0)  # [B, T, D]

    @staticmethod
    def _frame_positions(durations):
        B, max_len = durations.size(0), int(durations.sum(dim=1).max().item())
        out = durations.new_zeros(B, max_len, dtype=torch.float)
        for b in range(B):
            idx = 0
            for d in durations[b]:
                d = int(d)
                if d > 0:
                    out[b, idx:idx + d] = torch.linspace(0, 1, d)
                    idx += d
        return out

    def forward(self, phoneme_ids, pitch, durations):
        pe = self.phon_emb(phoneme_ids)                                  # [B, L, 64]
        pp = self.pitch_proj(pitch.float().unsqueeze(-1))                # [B, L, 32]
        logdur = torch.log1p(durations.float()).unsqueeze(-1)
        phon_feat = torch.cat([pe, pp, logdur], dim=-1)
        frame_feat = self.length_regulate(phon_feat, durations)           # [B, T, D-1]
        pos = self._frame_positions(durations).unsqueeze(-1)
        frame_feat = torch.cat([frame_feat, pos.to(frame_feat)], dim=-1)
        return self.proj(frame_feat.transpose(1, 2))                      # [B, out_ch, T]


class NoteInputEncoder(nn.Module):
    """InputEncoder's inputs read as a phoneme stream and a note stream (``model.encoder: notes``).

    Hosts send what they send to InputEncoder: phoneme ids, a per-phoneme note pitch and
    per-phoneme durations. A vowel held over several notes is written as the vowel followed
    by hold tokens (``hold_id``, e.g. "-"), each carrying its own note's pitch. Before any
    embedding the tokens are rebuilt into

      * a phoneme stream — hold tokens merge into the phoneme before them, so the held
        vowel stays ONE phoneme (one embedding, one position ramp). The hold token never
        reaches an embedding, so the network never learns it;
      * a note stream — a note starts at every nucleus and hold token, at the first pau of a
        run (a rest note) and at the first token; a consonant belongs to the note in
        progress. Note pitch = the pitch of the note's first token, masked on rest notes.

    Each stream is expanded by its own durations and the two are concatenated per frame:
      phoneme [phon_emb 64 | log1p(dur) | position] + note [pitch_proj 32 · (1 − rest) | rest |
      log1p(note dur) | position] → 1×1 conv proj → [B, out_ch, T].
    """

    def __init__(self, n_phonemes: int, nucleus_ids, pau_id: int = 0, hold_id: int | None = None,
                 phon_dim: int = 64, pitch_dim: int = 32, out_ch: int = 128):
        super().__init__()
        self.phon_emb = nn.Embedding(n_phonemes, phon_dim)
        self.pitch_proj = nn.Linear(1, pitch_dim)
        is_nucleus = torch.zeros(n_phonemes, dtype=torch.bool)
        is_nucleus[torch.as_tensor(list(nucleus_ids), dtype=torch.long)] = True
        self.register_buffer("is_nucleus", is_nucleus, persistent=False)   # rebuilt from the ckpt config
        self.pau_id = int(pau_id)
        self.hold_id = -1 if hold_id is None else int(hold_id)
        feat_dim = (phon_dim + 2) + (pitch_dim + 3)
        self.proj = nn.Sequential(nn.Conv1d(feat_dim, out_ch, 1), nn.PReLU(out_ch),
                                  nn.Conv1d(out_ch, out_ch, 1))

    def streams(self, phoneme_ids, pitch, durations):
        """Token inputs [B, L] -> (phoneme ids, phoneme durations, note pitch, note rest, note
        durations), each [B, L] with unused trailing slots at duration 0."""
        ids, d, p = phoneme_ids.long(), durations.long(), pitch.float()
        hold = ids == self.hold_id
        group = (torch.cumsum((~hold).long(), dim=1) - 1).clamp(min=0)
        ph_dur = torch.zeros_like(d).scatter_add(1, group, d)
        ph_ids = torch.zeros_like(ids).scatter_add(1, group, ids * (~hold).long())

        is_pau = ids == self.pau_id
        prev_pau = torch.cat([torch.zeros_like(is_pau[:, :1]), is_pau[:, :-1]], dim=1)
        first = torch.cat([torch.ones_like(is_pau[:, :1]), torch.zeros_like(is_pau[:, 1:])], dim=1)
        nuc = self.is_nucleus[ids] & ~hold
        start = nuc | hold | (is_pau & ~prev_pau) | first
        rest = start & ~nuc & ~hold                            # a pau run, or consonants before any note
        note = torch.cumsum(start.long(), dim=1) - 1
        note_dur = torch.zeros_like(d).scatter_add(1, note, d)
        note_pitch = torch.zeros_like(p).scatter_add(1, note, p * start.float())
        note_rest = torch.zeros_like(p).scatter_add(1, note, rest.float())
        return ph_ids, ph_dur, note_pitch, note_rest, note_dur

    def forward(self, phoneme_ids, pitch, durations, frames=None):
        if frames is None:
            frames = frames_for(durations)
        ph_ids, ph_dur, n_pitch, n_rest, n_dur = self.streams(phoneme_ids, pitch, durations)
        ph_feat = torch.cat([self.phon_emb(ph_ids), torch.log1p(ph_dur.float()).unsqueeze(-1)], dim=-1)
        nt_feat = torch.cat([self.pitch_proj(n_pitch.unsqueeze(-1)) * (1.0 - n_rest).unsqueeze(-1),
                             n_rest.unsqueeze(-1), torch.log1p(n_dur.float()).unsqueeze(-1)], dim=-1)
        feat = torch.cat([length_regulate(ph_feat, ph_dur, frames),
                          position_in_token(ph_dur, frames).unsqueeze(-1),
                          length_regulate(nt_feat, n_dur, frames),
                          position_in_token(n_dur, frames).unsqueeze(-1)], dim=-1)
        return self.proj(feat.transpose(1, 2))                                # [B, out_ch, T]


class ConvNeXtBlock1d(nn.Module):
    """1D ConvNeXt block (Liu et al. 2022, own implementation): depthwise conv →
    channel LayerNorm → pointwise 4× → GELU → pointwise → layer-scaled residual.
    No gating, no 3^k dilation — a different lineage from the gated-conv stack."""

    def __init__(self, ch: int, kernel: int = 7, dilation: int = 1,
                 expand: int = 4, ls_init: float = 1e-6):
        super().__init__()
        pad = dilation * (kernel - 1) // 2                          # 'same'
        self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, expand * ch)
        self.pw2 = nn.Linear(expand * ch, ch)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(ls_init * torch.ones(ch))

    def forward(self, x):                                          # x: [B, C, T]
        h = self.dw(x).transpose(1, 2)                            # [B, T, C]
        h = self.pw2(self.act(self.pw1(self.norm(h))))
        return x + (self.gamma * h).transpose(1, 2)


class ConvNeXtConditioner(nn.Module):
    """Non-gated ConvNeXt-1D conditioner (``model.conditioner: convnext``):
    feat [B,128,T] → cond [B,256,T]. Receptive field ≈ 1+6·Σdil ≈ 253 frames."""
    DILATIONS = [1, 5, 12, 24]                       # default: 4 blocks, RF ≈ 253 f（配布 config と同じ）

    def __init__(self, in_ch: int = 128, ch: int = 128, cond_out: int = 256,
                 dilations=None):
        super().__init__()
        self.dilations = list(dilations) if dilations else list(self.DILATIONS)
        self.in_conv = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.ModuleList(ConvNeXtBlock1d(ch, dilation=d) for d in self.dilations)
        self.norm = nn.LayerNorm(ch)
        self.out = nn.Conv1d(ch, cond_out, 1)

    def forward(self, feat):                                       # [B, in_ch, T]
        x = self.in_conv(feat)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.out(x)                                        # [B, cond_out, T]


class Denoiser3ch(nn.Module):
    """16 gated blocks = [1,2,4,...,128]×2, 64ch. Predicts x0 directly (raw output)."""
    DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128] * 2

    def __init__(self, ch: int = 64, cond_dim: int = COND_DIM, step_dim: int = STEP_EMB):
        super().__init__()
        self.in_conv = nn.Conv1d(OUT_CH, ch, 1)
        self.blocks = nn.ModuleList(
            GatedResBlock(ch, kernel=3, dilation=d, cond_dim=cond_dim, step_dim=step_dim)
            for d in self.DILATIONS)
        self.out = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.PReLU(ch), nn.Conv1d(ch, OUT_CH, 1))

    def forward(self, x_t, cond, step_emb):       # x_t:[B,2,T], cond:[B,256,T], step:[B,64]
        x, skips = self.in_conv(x_t), 0
        for blk in self.blocks:
            x, s = blk(x, cond=cond, step_emb=step_emb)
            skips = skips + s
        return self.out(skips)                    # [B,2,T] x0 / velocity (plain, vibrato)


class _F0Base(nn.Module):
    """Shared backbone for the F0 models: input encoder + conditioner + speaker/style
    + 3ch denoiser + cond_from_inputs. The reverse process (DDPM vs rectified flow) and
    its time embedding live in the subclasses."""

    def __init__(self, n_phonemes: int = 50, n_speakers: int = 0, spk_dim: int = 32,
                 n_styles: int = 0, hidden: int = 256, conditioner: str = "convnext",
                 convnext_dilations=None, encoder: str = "phonemes", nucleus_ids=(),
                 pau_id: int = 0, hold_id: int | None = None):
        super().__init__()
        if conditioner != "convnext":
            raise ValueError(f"conditioner={conditioner!r}: only 'convnext' is supported "
                             f"(a checkpoint trained with another conditioner cannot load)")
        self.hidden = hidden
        self.conditioner_type = conditioner
        self.encoder_type = encoder
        if encoder == "phonemes":
            self.encoder = InputEncoder(n_phonemes, out_ch=128)
        elif encoder == "notes":
            self.encoder = NoteInputEncoder(n_phonemes, nucleus_ids, pau_id=pau_id, hold_id=hold_id,
                                            out_ch=128)
        else:
            raise ValueError(f"encoder={encoder!r}: 'phonemes' or 'notes'")
        self.conditioner = ConvNeXtConditioner(in_ch=128, ch=128, cond_out=hidden,
                                               dilations=convnext_dilations)
        self.denoiser = Denoiser3ch(ch=64, cond_dim=hidden, step_dim=STEP_EMB)
        # v/uv: separate deterministic classifier head. Takes the 256-dim context only
        # (phoneme identity decides voicing); the raw frame-pitch 2ch carries no v/uv cue.
        self.uv_head = nn.Sequential(nn.Conv1d(hidden, 64, 1), nn.PReLU(64),
                                     nn.Conv1d(64, 1, 1))

        # speaker: low-dim bank + projection, added time-invariantly to the condition
        self.spk_bank = self.spk_proj = None
        if n_speakers > 0 and spk_dim > 0:
            self.spk_bank = nn.Embedding(n_speakers, spk_dim)
            nn.init.normal_(self.spk_bank.weight, 0.0, spk_dim ** -0.5)
            self.spk_proj = nn.Linear(spk_dim, hidden)
        # style: full-width embedding, same additive injection; style_mix blends at inference
        self.style_emb = None
        if n_styles > 0:
            self.style_emb = nn.Embedding(n_styles, hidden)
            nn.init.normal_(self.style_emb.weight, 0.0, hidden ** -0.5)

    def cond_from_inputs(self, phoneme_ids, pitch, durations,
                         spk_id=None, style_id=None, style_mix=None):
        """Condition [B, 256, T] = conditioner 256 + speaker + style (time-invariant)."""
        feat = self.encoder(phoneme_ids, pitch, durations)
        c = self.conditioner(feat)                                    # [B, 256, T]
        if self.spk_bank is not None and spk_id is not None:
            c = c + self.spk_proj(self.spk_bank(spk_id)).unsqueeze(-1)
        if self.style_emb is not None:
            if style_mix is not None:                                 # continuous blend
                c = c + (style_mix @ self.style_emb.weight).unsqueeze(-1)
            elif style_id is not None:
                c = c + self.style_emb(style_id).unsqueeze(-1)
        return c

    def _append_uv(self, x2, cond):
        """[B,2,T] plain/vib + sigmoid(uv_head(cond)) → [B,3,T] (ch2 = uv prob in [0,1])."""
        return torch.cat([x2, torch.sigmoid(self.uv_head(cond))], dim=1)


class F0Diffusion3ch(_F0Base):
    """DDPM x0-prediction, cosine β, ancestral sampling."""

    def __init__(self, n_steps: int = 10, n_phonemes: int = 50,
                 n_speakers: int = 0, spk_dim: int = 32, n_styles: int = 0,
                 hidden: int = 256, conditioner: str = "convnext", convnext_dilations=None,
                 **encoder_kw):
        super().__init__(n_phonemes, n_speakers, spk_dim, n_styles, hidden, conditioner,
                         convnext_dilations=convnext_dilations, **encoder_kw)
        self.n_steps = n_steps
        self.step_emb = DiffusionStepEmbedding(n_steps, emb_dim=STEP_EMB, proj_dim=STEP_EMB)
        betas = cosine_betas(n_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def predict_x0(self, x_t, cond, t):
        return self.denoiser(x_t, cond, self.step_emb(t))

    @torch.no_grad()
    def sample(self, phoneme_ids, pitch, durations,
               spk_id=None, style_id=None, style_mix=None):
        """Phoneme-level inputs → normalized 3ch [B,3,T]. DDPM x0-prediction ancestral."""
        cond = self.cond_from_inputs(phoneme_ids, pitch, durations,
                                     spk_id=spk_id, style_id=style_id, style_mix=style_mix)
        B, T = cond.size(0), cond.size(2)
        x = torch.randn(B, OUT_CH, T, device=cond.device)
        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=cond.device, dtype=torch.long)
            x0 = self.predict_x0(x, cond, t)
            ab = self.alpha_bar[i]
            ab_prev = self.alpha_bar[i - 1] if i > 0 else torch.ones((), device=cond.device)
            a, b = self.alphas[i], self.betas[i]
            mu = (torch.sqrt(ab_prev) * b / (1 - ab)) * x0 + \
                 (torch.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                sigma = torch.sqrt(b * (1 - ab_prev) / (1 - ab))
                x = mu + sigma * torch.randn_like(x)
            else:
                x = x0
        return self._append_uv(x, cond)                   # [B,3,T]: plain, vib, uv_prob


class F0Reflow3ch(_F0Base):
    """Rectified flow: velocity field v = x1 − x0 over a straight noise→data path,
    Euler integration (num_steps=1 by default). Same 3ch normalized target and cond
    contract as F0Diffusion3ch — only the reverse process differs."""

    def __init__(self, n_phonemes: int = 50, n_speakers: int = 0, spk_dim: int = 32,
                 n_styles: int = 0, hidden: int = 256, conditioner: str = "convnext",
                 convnext_dilations=None, num_steps: int = 1, **encoder_kw):
        super().__init__(n_phonemes, n_speakers, spk_dim, n_styles, hidden, conditioner,
                         convnext_dilations=convnext_dilations, **encoder_kw)
        self.num_steps = num_steps
        self.step_emb = ContinuousTimeEmbedding(emb_dim=STEP_EMB, proj_dim=STEP_EMB)

    def predict_velocity(self, x_t, cond, t):     # t: [B] float in [0,1]
        return self.denoiser(x_t, cond, self.step_emb(t))

    @torch.no_grad()
    def sample(self, phoneme_ids, pitch, durations,
               spk_id=None, style_id=None, style_mix=None, num_steps=None):
        """Phoneme-level inputs → normalized 3ch [B,3,T]. Euler from Gaussian noise."""
        cond = self.cond_from_inputs(phoneme_ids, pitch, durations,
                                     spk_id=spk_id, style_id=style_id, style_mix=style_mix)
        B, T = cond.size(0), cond.size(2)
        steps = int(num_steps or self.num_steps)
        x = torch.randn(B, OUT_CH, T, device=cond.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=cond.device)
            x = x + self.predict_velocity(x, cond, t) * dt
        return self._append_uv(x, cond)                   # [B,3,T]: plain, vib, uv_prob
