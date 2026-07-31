"""Two-tower fracture-edge matcher.

Each fracture edge -- the silhouette profile plus the texture strip immediately
inside it -- is embedded into a shared space in which the two sides of one
break land close together.  Scoring a corpus is then a single matrix product
over the embeddings, which is what makes 5,000-fragment instances tractable:
the alternative, a cross-encoder over all ordered pairs, is 25 million forward
passes.

The matcher deliberately never sees slip width, binding-notch heights, scribal
hand or excavation unit.  Those are the global layer's evidence; withholding
them here keeps the constraint ablations in the assembly stage free of leakage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth import CANON_W, PATCH_H

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class EdgeEncoder(nn.Module):
    """Encode one fracture edge to a unit vector.

    Height is collapsed by strided convolution while the width axis is kept
    until late: the evidence for a join is *column-registered* (does the fibre
    at column 17 continue, does the tongue at column 30 line up), so pooling
    across the width early would throw away exactly the signal that matters.
    """

    def __init__(self, dim=128, width=64):
        super().__init__()
        c = width

        def blk(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride=(s, 1), padding=1, bias=False),
                nn.BatchNorm2d(o), nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.GELU(),
            )

        self.b1 = blk(2, c, 2)        # 32 -> 16
        self.b2 = blk(c, c * 2, 2)    # 16 -> 8
        self.b3 = blk(c * 2, c * 2, 2)  # 8 -> 4
        self.b4 = blk(c * 2, c * 2, 4)  # 4 -> 1
        self.wpool = nn.Conv1d(c * 2, c * 2, 3, stride=2, padding=1)  # 48 -> 24
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c * 2 * 24, 512), nn.GELU(), nn.Linear(512, dim)
        )

    def forward(self, x):
        x = self.b4(self.b3(self.b2(self.b1(x))))  # (B, C, 1, W)
        x = x.squeeze(2)
        x = F.gelu(self.wpool(x))
        return F.normalize(self.head(x), dim=-1)


def edge_tensor(prof, patch):
    """Pack (profile, patch) into the 2-channel input the encoder expects.

    The profile is standardised per edge and broadcast down the height so that
    it is registered column-by-column with the texture it belongs to.
    """
    prof = np.asarray(prof, np.float32)
    patch = np.asarray(patch, np.float32)
    if prof.ndim == 1:
        prof = prof[None]
        patch = patch[None]
    p = prof / (prof.std(axis=1, keepdims=True) + 1e-3)
    p = np.clip(p, -4, 4)
    pc = np.repeat(p[:, None, None, :], PATCH_H, axis=2)
    tx = patch[:, None] - patch.reshape(patch.shape[0], -1).mean(1)[:, None, None, None]
    return np.concatenate([tx, pc], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# training data
# --------------------------------------------------------------------------

def _gen_pairs(args):
    """Worker: one instance -> the descriptors of its true conjoining pairs."""
    import synth
    seed, n_slips, dmg = args
    spec = synth.InstanceSpec(n_slips=n_slips, seed=seed,
                              damage=synth.DamageSpec(**dmg))
    inst = synth.generate_instance(spec)
    f = {x["frag_id"]: x for x in inst["frags"]}
    bp, bt, tp, tt = [], [], [], []
    for a, b in inst["joins"]:
        bp.append(f[a]["bot_prof"]); bt.append(f[a]["bot_patch"])
        tp.append(f[b]["top_prof"]); tt.append(f[b]["top_patch"])
    if not bp:
        z = np.zeros((0, CANON_W), np.float16)
        zz = np.zeros((0, PATCH_H, CANON_W), np.float16)
        return z, zz, z, zz
    return (np.array(bp, np.float16), np.array(bt, np.float16),
            np.array(tp, np.float16), np.array(tt, np.float16))


def build_pairs(out, n_instances=60, n_slips=300, workers=12, seed0=100000):
    """Training corpus.

    Training instances come from a seed range disjoint from every evaluation
    instance.  Preservation state is randomised across instances -- surface
    loss at the fracture above all, which is what governs how much of the
    mating relief survives -- so that the matcher is trained for the whole
    range of conditions it is later evaluated on.  Without that randomisation
    the matcher would be out of distribution on the poorly preserved corpora
    and the global layer would be credited with beating a straw man.
    """
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(seed0)
    jobs = []
    for i in range(n_instances):
        dmg = dict(
            p_gap=0.0, p_fragment_lost=0.0, breaks_per_slip=3.0,
            erosion_mm=float(np.exp(rng.uniform(np.log(0.12), np.log(1.5)))),
            pullout_mean_mm=float(rng.uniform(0.30, 2.00)),
            face_loss_mm=float(rng.uniform(0.20, 2.00)),
            face_texture_noise=float(rng.uniform(0.05, 0.70)),
            warp_deg=float(rng.uniform(0.4, 2.0)),
            stain_rate=float(rng.uniform(0.0, 1.5)),
            ink_fade=float(rng.uniform(0.1, 0.6)),
        )
        jobs.append((seed0 + i, n_slips, dmg))
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_gen_pairs, jobs)):
            parts.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_instances} instances, {time.time()-t0:.0f}s", flush=True)
    bp = np.concatenate([p[0] for p in parts])
    bt = np.concatenate([p[1] for p in parts])
    tp = np.concatenate([p[2] for p in parts])
    tt = np.concatenate([p[3] for p in parts])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, bot_prof=bp, bot_patch=bt, top_prof=tp, top_patch=tt)
    print(f"{len(bp)} positive pairs -> {out} ({time.time()-t0:.0f}s)")
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _augment(x, rng):
    """Photometric and sub-pixel jitter applied independently to the two sides
    of a pair, so the encoder cannot key on a shared nuisance."""
    b = x.shape[0]
    x = x.clone()
    x[:, 0] *= torch.as_tensor(rng.uniform(0.85, 1.15, (b, 1, 1)),
                               dtype=x.dtype, device=x.device)
    x[:, 0] += torch.as_tensor(rng.normal(0, 0.02, (b, 1, 1)),
                               dtype=x.dtype, device=x.device)
    x = x + torch.randn_like(x) * 0.02
    if rng.random() < 0.5:  # small horizontal shift: imperfect width registration
        s = int(rng.integers(-2, 3))
        if s:
            x = torch.roll(x, s, dims=-1)
    return x


def train(pairs, out, epochs=30, bs=512, dim=128, lr=3e-4, tau=0.07, seed=0,
          val_frac=0.05):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    z = np.load(pairs)
    n = len(z["bot_prof"])
    idx = rng.permutation(n)
    nval = int(n * val_frac)
    val, tr = idx[:nval], idx[nval:]

    B = torch.from_numpy(edge_tensor(z["bot_prof"], z["bot_patch"]))
    T = torch.from_numpy(edge_tensor(z["top_prof"], z["top_patch"]))
    print(f"{n} pairs ({len(tr)} train / {len(val)} val), tensor {tuple(B.shape)}")

    model = EdgeEncoder(dim=dim).to(DEV)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / tau), dtype=torch.float32,
                                            device=DEV))
    opt = torch.optim.AdamW(
        [{"params": list(model.parameters()), "weight_decay": 1e-4},
         {"params": [logit_scale], "weight_decay": 0.0}], lr=lr)
    steps = max(1, len(tr) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr],
                                                total_steps=steps)
    scaler = torch.amp.GradScaler(DEV)

    t0 = time.time()
    step = 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for k in range(0, len(perm) - bs + 1, bs):
            sel = tr[perm[k:k + bs]]
            xb = _augment(B[sel].to(DEV, non_blocking=True), rng)
            xt = _augment(T[sel].to(DEV, non_blocking=True), rng)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                eb, et = model(xb), model(xt)
                logits = logit_scale.exp().clamp(max=100) * eb @ et.T
                lbl = torch.arange(len(sel), device=DEV)
                loss = 0.5 * (F.cross_entropy(logits, lbl) +
                              F.cross_entropy(logits.T, lbl))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            if step < steps - 1:
                sched.step()
            step += 1
            tot += loss.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            r1, mrr = _val(model, B, T, val, bs=1024)
            print(f"  ep{ep+1:3d} loss {tot/max(nb,1):.4f}  val R@1 {r1:.3f}  MRR {mrr:.3f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "dim": dim}, out)
    print(f"saved {out}")
    return model


@torch.no_grad()
def _val(model, B, T, val, bs=1024):
    """Retrieval within a held-out pool: for each bottom edge, is its true
    mating top edge ranked first among the whole pool?"""
    model.eval()
    sel = val[:bs]
    eb = model(B[sel].to(DEV)).float()
    et = model(T[sel].to(DEV)).float()
    s = eb @ et.T
    rank = (s > s.diag()[:, None]).sum(1) + 1
    return float((rank == 1).float().mean()), float((1.0 / rank.float()).mean())


def load_model(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = EdgeEncoder(dim=ck.get("dim", 128)).to(DEV)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# --------------------------------------------------------------------------
# scoring a corpus
# --------------------------------------------------------------------------

@torch.no_grad()
def embed_edges(model, prof, patch, bs=4096):
    out = []
    X = edge_tensor(prof, patch)
    for k in range(0, len(X), bs):
        xb = torch.from_numpy(X[k:k + bs]).to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def score_matrix(model, inst):
    """S[i, j] = compatibility of fragment i's *bottom* with j's *top*.

    Self-scores are removed; a fragment cannot join itself.
    """
    eb = embed_edges(model, inst["bot_prof"], inst["bot_patch"])
    et = embed_edges(model, inst["top_prof"], inst["top_patch"])
    S = eb @ et.T
    np.fill_diagonal(S, -np.inf)
    return S.astype(np.float32)


def ncc_score_matrix(inst):
    """Classical baseline: normalised cross-correlation of the two silhouette
    profiles, which is in essence what edge-shape matching in the conservation
    literature does before any learning is involved."""
    A = np.asarray(inst["bot_prof"], np.float32)
    Bp = np.asarray(inst["top_prof"], np.float32)
    A = A - A.mean(1, keepdims=True)
    Bp = Bp - Bp.mean(1, keepdims=True)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)
    Bp /= (np.linalg.norm(Bp, axis=1, keepdims=True) + 1e-6)
    S = A @ Bp.T
    np.fill_diagonal(S, -np.inf)
    return S.astype(np.float32)


def topk_accuracy(S, joins, ks=(1, 5, 10, 50)):
    """Fraction of true joins whose partner is in the top-k of that fragment's
    ranked candidate list -- the metric the rejoining literature reports."""
    out = {}
    if len(joins) == 0:
        return {f"top{k}": float("nan") for k in ks}
    order = np.argsort(-S, axis=1)
    rank_of = {}
    for a, b in joins:
        pos = int(np.where(order[a] == b)[0][0]) + 1
        rank_of[(a, b)] = pos
    r = np.array(list(rank_of.values()))
    for k in ks:
        out[f"top{k}"] = float((r <= k).mean())
    out["mrr"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "train", "eval"])
    ap.add_argument("--pairs", default="data/train_pairs.npz")
    ap.add_argument("--model", default="data/matcher.pt")
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--slips", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()

    if a.cmd == "pairs":
        build_pairs(a.pairs, a.instances, a.slips, a.workers)
    elif a.cmd == "train":
        train(a.pairs, a.model, epochs=a.epochs, bs=a.bs)
    else:
        import synth
        m = load_model(a.model)
        inst = synth.generate_instance(synth.InstanceSpec(n_slips=300, seed=0))
        d = dict(bot_prof=np.array([f["bot_prof"] for f in inst["frags"]]),
                 bot_patch=np.array([f["bot_patch"] for f in inst["frags"]]),
                 top_prof=np.array([f["top_prof"] for f in inst["frags"]]),
                 top_patch=np.array([f["top_patch"] for f in inst["frags"]]))
        S = score_matrix(m, d)
        print("learned :", topk_accuracy(S, inst["joins"]))
        print("NCC     :", topk_accuracy(ncc_score_matrix(d), inst["joins"]))
