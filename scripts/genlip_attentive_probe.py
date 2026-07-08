""" Attentive-probe (AttentionPoolLatent) ImageNet-1k evaluation of a frozen GenLIP image encoder.

GenLIP has no [CLS] token, so we follow the AIM/DINOv2-style frozen-backbone protocol: freeze the trunk,
extract last-layer image patch features (post-ln_post), and train a small attention-pooling head
(``timm.AttentionPoolLatent``: a learnable latent query that cross-attends the patch tokens, padding-masked)
+ a linear classifier. The backbone is frozen, so features are extracted ONCE and cached; only the head trains
-> fast, many epochs. No train-time augmentation (cached features are deterministic) -- a plain frozen probe.

Example:
    python scripts/genlip_attentive_probe.py \
        --model naflexgenlip_b16 --checkpoint /path/epoch_32.pt \
        --imagenet-train /data/f/imagenet/train --imagenet-val /data/f/imagenet/val \
        --seq-len 256 --train-per-class 100 --epochs 20 --lr 1e-3 \
        --device cuda --precision amp_bf16
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from open_clip import create_model_and_transforms
from open_clip.naflex_config import NaFlexDataConfig
from open_clip.naflex_genlip_model import build_image_attn_mask, build_image_position_ids
from open_clip_train.naflex_data import collate_naflex_tuples, create_naflex_eval_transform
from timm.layers import AttentionPoolLatent


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "_orig_mod.", "trainable_module."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def load_torch_checkpoint(path: str, allow_unsafe: bool = False):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe:
            raise RuntimeError(
                f"failed to load checkpoint with weights_only=True: {path}. "
                "If this is a trusted legacy checkpoint that requires pickle loading, "
                "retry with --allow-unsafe-checkpoint-load."
            ) from exc
        print(f"WARNING: falling back to unsafe pickle checkpoint loading for trusted file: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)


def load_weights(model, path: str, use_ema: bool = False, allow_unsafe: bool = False) -> None:
    obj = load_torch_checkpoint(path, allow_unsafe=allow_unsafe)
    while isinstance(obj, dict):
        if use_ema and isinstance(obj.get("state_dict_ema"), dict):
            obj = obj["state_dict_ema"]
            continue
        if use_ema:
            raise RuntimeError(f"--use-ema was requested, but checkpoint has no state_dict_ema: {path}")
        if isinstance(obj.get("state_dict"), dict):
            obj = obj["state_dict"]
            continue
        break
    if not isinstance(obj, dict):
        raise RuntimeError(f"checkpoint payload is not a state dict or checkpoint dict: {type(obj).__name__}")
    state_dict = {strip_prefix(k): v for k, v in obj.items() if torch.is_tensor(v)}
    if not state_dict:
        raise RuntimeError(f"checkpoint has no tensor weights: {path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded_keys = set(state_dict).intersection(model.state_dict())
    visual_loaded = sum(k.startswith("visual.") for k in loaded_keys)
    if not visual_loaded:
        raise RuntimeError(
            f"loaded checkpoint has no matching visual.* tensors for {path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    print(
        f"Loaded {len(state_dict)} tensors from {path} "
        f"(matched={len(loaded_keys)}, visual_matched={visual_loaded}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})."
    )
    if missing:
        print("  missing sample:", ", ".join(missing[:8]))
    if unexpected:
        print("  unexpected sample:", ", ".join(unexpected[:8]))


@torch.no_grad()
def extract_patch_features(visual, image, device, autocast):
    """Frozen GenLIP -> last-layer image patch hidden ``[B, Ni, width]`` (post-ln_post) + patch_valid."""
    patches = image["patches"].to(device, non_blocking=True)
    coord = image["patch_coord"].to(device, non_blocking=True)
    valid = image["patch_valid"].to(device, non_blocking=True)
    with autocast():
        x = visual.patch_embed(patches)
        cos, sin = visual.rotary(x, build_image_position_ids(coord, valid))
        x = visual.trunk(x, build_image_attn_mask(valid), cos, sin)  # [B, Ni, width], ln_post inside
    return x, valid


class ProbeHead(nn.Module):
    """AttentionPoolLatent (padding-masked) -> BN(affine=False) -> linear classifier."""

    def __init__(self, dim, num_classes, num_heads=12, q_proj=False, mlp_ratio=0.0, use_bn=True, bn_affine=False):
        super().__init__()
        import inspect

        pool_kwargs = dict(
            in_features=dim,
            embed_dim=dim,
            num_heads=num_heads,
            latent_len=1,
            mlp_ratio=mlp_ratio,
            out_features=0,  # out_features=0 -> pooled dim-vector (no proj/mlp)
        )
        if "q_proj" in inspect.signature(AttentionPoolLatent.__init__).parameters:
            pool_kwargs["q_proj"] = q_proj
        elif not q_proj:
            print(
                "note: installed timm AttentionPoolLatent has no q_proj arg; using the default query "
                "projection (expressivity-equivalent for a learnable latent query)."
            )
        self.pool = AttentionPoolLatent(**pool_kwargs)
        # affine=False = AIM-style pure standardizer (no learnable scale/shift); affine=True adds learnable gamma/beta.
        self.bn = nn.BatchNorm1d(dim, affine=bn_affine) if use_bn else nn.Identity()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, feats, valid):
        # additive mask so the latent query ignores padding patches: 0 for valid keys, -inf for padding
        attn_mask = torch.zeros(feats.shape[0], 1, 1, feats.shape[1], device=feats.device, dtype=feats.dtype)
        attn_mask = attn_mask.masked_fill(~valid[:, None, None, :], float("-inf"))
        pooled = self.pool(feats, attn_mask=attn_mask)  # [B, dim]
        return self.fc(self.bn(pooled))


def build_loader(root, eval_tf, max_seq_len, per_class, batch_size, workers, seed):
    import torchvision

    if not os.path.isdir(root):
        raise FileNotFoundError(f"ImageNet split directory not found: {root}")
    dataset = torchvision.datasets.ImageFolder(root, transform=eval_tf)
    if len(dataset.classes) != 1000:
        raise ValueError(f"expected ImageNet-1k with 1000 classes at {root}, found {len(dataset.classes)}")
    if per_class:
        by_class = defaultdict(list)
        for idx, (_, cls) in enumerate(dataset.samples):
            by_class[cls].append(idx)
        too_small = {cls: len(idxs) for cls, idxs in by_class.items() if len(idxs) < per_class}
        if too_small:
            few = ", ".join(f"{cls}:{n}" for cls, n in list(too_small.items())[:8])
            raise ValueError(f"--train-per-class={per_class} exceeds samples for some classes: {few}")
        rng = random.Random(seed)
        keep = []
        for cls, idxs in by_class.items():
            rng.shuffle(idxs)
            keep.extend(idxs[:per_class])
        dataset = Subset(dataset, keep)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=partial(collate_naflex_tuples, max_seq_len=max_seq_len),
    )
    return loader, len(dataset)


@torch.no_grad()
def cache_features(visual, loader, n, seq_len, dim, device, cache_device, autocast, tag):
    feats = torch.empty(n, seq_len, dim, dtype=torch.bfloat16, device=cache_device)
    valid = torch.empty(n, seq_len, dtype=torch.bool, device=cache_device)
    labels = torch.empty(n, dtype=torch.long, device=cache_device)
    i, t0 = 0, time.time()
    for image, y in loader:
        x, v = extract_patch_features(visual, image, device, autocast)
        b = x.shape[0]
        feats[i:i + b] = x.to(cache_device, torch.bfloat16)
        valid[i:i + b] = v.to(cache_device)
        labels[i:i + b] = y.to(cache_device)
        i += b
        if (i // b) % 50 == 0:
            print(f"  [{tag}] cached {i}/{n}  ({i / (time.time() - t0):.0f} img/s)", flush=True)
    return feats[:i], valid[:i], labels[:i]


def cache_meta(args, tag, n, seq_len, dim):
    ckpt_stat = os.stat(args.checkpoint)
    return {
        "tag": tag,
        "model": args.model,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_size": ckpt_stat.st_size,
        "checkpoint_mtime_ns": ckpt_stat.st_mtime_ns,
        "use_ema": args.use_ema,
        "imagenet_root": os.path.abspath(args.imagenet_train if tag == "train" else args.imagenet_val),
        "seq_len": seq_len,
        "patch_size": args.patch_size,
        "dim": dim,
        "n": n,
        "train_per_class": args.train_per_class if tag == "train" else 0,
        "seed": args.seed,
        "dtype": "bfloat16",
    }


def cache_path(cache_dir, tag):
    return Path(cache_dir) / f"{tag}_features.pt"


def load_feature_cache(cache_dir, tag, expected_meta, cache_device):
    path = cache_path(cache_dir, tag)
    if not path.is_file():
        return None
    obj = torch.load(path, map_location=cache_device, weights_only=True)
    meta = obj.get("meta", {})
    mismatches = {k: (meta.get(k), v) for k, v in expected_meta.items() if meta.get(k) != v}
    if mismatches:
        details = ", ".join(f"{k}: cached={a!r} expected={b!r}" for k, (a, b) in list(mismatches.items())[:8])
        raise RuntimeError(f"cache metadata mismatch for {path}: {details}. Use --force-recache to rebuild.")
    print(f"Loaded {tag} feature cache from {path}")
    return obj["feats"].to(cache_device), obj["valid"].to(cache_device), obj["labels"].to(cache_device)


def save_feature_cache(cache_dir, tag, meta, feats, valid, labels):
    path = cache_path(cache_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"meta": meta, "feats": feats.cpu(), "valid": valid.cpu(), "labels": labels.cpu()},
        tmp,
    )
    os.replace(tmp, path)
    print(f"Saved {tag} feature cache to {path}")


def bytes_to_gib(nbytes):
    return nbytes / (1024 ** 3)


def estimate_cache_bytes(n, seq_len, dim):
    return n * seq_len * dim * 2 + n * seq_len + n * 8


def choose_cache_device(requested, feature_bytes, head_batch, seq_len, dim):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA cache requested but CUDA is unavailable; using CPU cache.")
        return torch.device("cpu")
    cache_device = torch.device(requested)
    if cache_device.type != "cuda":
        return cache_device

    free, total = torch.cuda.mem_get_info(cache_device)
    batch_working = head_batch * seq_len * dim * 4
    needed = feature_bytes + batch_working
    if needed > free * 0.85:
        print(
            "WARNING: estimated cache footprint "
            f"{bytes_to_gib(needed):.1f} GiB exceeds safe free CUDA memory "
            f"{bytes_to_gib(free * 0.85):.1f} GiB; using CPU cache."
        )
        return torch.device("cpu")
    print(
        f"Estimated cache footprint: {bytes_to_gib(feature_bytes):.1f} GiB "
        f"on {cache_device} (free {bytes_to_gib(free):.1f} / total {bytes_to_gib(total):.1f} GiB)."
    )
    return cache_device


def append_jsonl(path, record):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def validate_args(args):
    positive_ints = ("seq_len", "patch_size", "epochs", "head_batch", "extract_batch")
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.train_per_class < 0:
        raise ValueError("--train-per-class must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.wd < 0:
        raise ValueError("--wd must be >= 0")
    if args.pool_num_heads <= 0:
        raise ValueError("--pool-num-heads must be > 0")
    if args.mlp_ratio < 0:
        raise ValueError("--mlp-ratio must be >= 0")
    if args.output_dir and not args.results_file:
        args.results_file = str(Path(args.output_dir) / "results.jsonl")


def accuracy(logits, target, topk=(1, 5)):
    pred = logits.topk(max(topk), 1, True, True).indices.t()
    correct = pred.eq(target.view(1, -1))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def evaluate(head, feats, valid, labels, batch_size, device):
    head.eval()
    top1 = top5 = 0.0
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):
            x = feats[i:i + batch_size].to(device, torch.float32)
            v = valid[i:i + batch_size].to(device)
            logits = head(x, v)
            a1, a5 = accuracy(logits, labels[i:i + batch_size].to(device))
            top1 += a1
            top5 += a5
    n = feats.shape[0]
    return 100 * top1 / n, 100 * top5 / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="naflexgenlip_b16")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help="Allow pickle checkpoint loading if weights_only=True fails. Only use for trusted files.",
    )
    p.add_argument("--imagenet-train", required=True)
    p.add_argument("--imagenet-val", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--train-per-class", type=int, default=100, help="Images/class to cache for training (0=all).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pool-num-heads", type=int, default=12)
    p.add_argument("--no-q-proj", dest="q_proj", action="store_false", help="AIM-style: latent used directly as Q.")
    p.add_argument("--mlp-ratio", type=float, default=0.0, help=">0 adds the MAP-head residual MLP.")
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--head-batch", type=int, default=512, help="Batch size for head train/eval on cached features.")
    p.add_argument("--extract-batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-device", default="cuda", help="Where to keep cached features (cuda|cpu).")
    p.add_argument("--cache-dir", default=None, help="Optional directory for reusable train/val feature caches.")
    p.add_argument("--force-recache", action="store_true", help="Rebuild feature caches even if --cache-dir exists.")
    p.add_argument("--output-dir", default=None, help="Optional directory for metrics and best head checkpoint.")
    p.add_argument("--results-file", default=None, help="Optional JSONL metrics path. Defaults to output_dir/results.jsonl.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="amp_bf16", choices=("amp_bf16", "amp_bfloat16", "amp_fp16", "fp32"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA device requested but CUDA is unavailable; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    use_amp = args.precision.startswith("amp")
    amp_dtype = torch.bfloat16 if "bf16" in args.precision or args.precision == "amp_bfloat16" else torch.float16
    if use_amp and device.type == "cpu":
        print("WARNING: AMP requested on CPU; using fp32.")
        use_amp = False
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext

    print(f"Building {args.model} (frozen backbone) ...")
    model, _, preprocess_val = create_model_and_transforms(args.model, aug_cfg={"use_timm": True, "naflex": True})
    load_weights(
        model,
        args.checkpoint,
        use_ema=args.use_ema,
        allow_unsafe=args.allow_unsafe_checkpoint_load,
    )
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    visual = model.visual
    dim = model.trunk_cfg.width

    ndc = NaFlexDataConfig.resolve(
        patch_sizes=[args.patch_size],
        seq_lens=[args.seq_len],
        eval_patch_size=args.patch_size,
        eval_seq_len=args.seq_len,
    )
    eval_tf, max_seq_len, _ = create_naflex_eval_transform(preprocess_val, ndc)

    tr_loader, n_tr = build_loader(
        args.imagenet_train,
        eval_tf,
        max_seq_len,
        args.train_per_class,
        args.extract_batch,
        args.workers,
        args.seed,
    )
    va_loader, n_va = build_loader(
        args.imagenet_val,
        eval_tf,
        max_seq_len,
        0,
        args.extract_batch,
        args.workers,
        args.seed,
    )
    print(f"  train images: {n_tr} ({args.train_per_class}/class) | val images: {n_va}")
    feature_bytes = estimate_cache_bytes(n_tr + n_va, max_seq_len, dim)
    cache_device = choose_cache_device(args.cache_device, feature_bytes, args.head_batch, max_seq_len, dim)
    print(f"Caching features (dim={dim}, seq_len={max_seq_len}, cache_device={cache_device}) ...")

    tr_meta = cache_meta(args, "train", n_tr, max_seq_len, dim)
    va_meta = cache_meta(args, "val", n_va, max_seq_len, dim)
    tr_cache = None if args.force_recache or not args.cache_dir else load_feature_cache(args.cache_dir, "train", tr_meta, cache_device)
    va_cache = None if args.force_recache or not args.cache_dir else load_feature_cache(args.cache_dir, "val", va_meta, cache_device)
    if tr_cache is None:
        tr_cache = cache_features(visual, tr_loader, n_tr, max_seq_len, dim, device, cache_device, autocast, "train")
        if args.cache_dir:
            save_feature_cache(args.cache_dir, "train", tr_meta, *tr_cache)
    if va_cache is None:
        va_cache = cache_features(visual, va_loader, n_va, max_seq_len, dim, device, cache_device, autocast, "val")
        if args.cache_dir:
            save_feature_cache(args.cache_dir, "val", va_meta, *va_cache)
    tr_feats, tr_valid, tr_labels = tr_cache
    va_feats, va_valid, va_labels = va_cache

    head = ProbeHead(
        dim,
        num_classes=1000,
        num_heads=args.pool_num_heads,
        q_proj=args.q_proj,
        mlp_ratio=args.mlp_ratio,
        use_bn=args.use_bn,
    ).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(
        f"Head: AttentionPoolLatent(q_proj={args.q_proj}, mlp_ratio={args.mlp_ratio}) + "
        f"{'BN' if args.use_bn else 'noBN'} + Linear  ({n_head / 1e6:.2f}M params)"
    )

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best1 = 0.0
    best_record = None
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.output_dir) / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(tr_feats.shape[0], device=cache_device)
        t0 = time.time()
        loss_sum = 0.0
        seen = 0
        for i in range(0, perm.shape[0], args.head_batch):
            idx = perm[i:i + args.head_batch]
            x = tr_feats[idx].to(device, torch.float32)
            v = tr_valid[idx].to(device)
            y = tr_labels[idx].to(device)
            logits = head(x, v)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.shape[0]
            seen += y.shape[0]
        sched.step()
        top1, top5 = evaluate(head, va_feats, va_valid, va_labels, args.head_batch, device)
        epoch_loss = loss_sum / max(1, seen)
        lr = sched.get_last_lr()[0]
        is_best = top1 > best1
        best1 = max(best1, top1)
        record = {
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "loss": epoch_loss,
            "lr": lr,
            "val_top1": top1,
            "val_top5": top5,
            "best_val_top1": best1,
            "seconds": time.time() - t0,
            "train_images": int(tr_feats.shape[0]),
            "val_images": int(va_feats.shape[0]),
            "cache_device": str(cache_device),
        }
        append_jsonl(args.results_file, record)
        if is_best:
            best_record = record
            if args.output_dir:
                torch.save(
                    {"epoch": epoch + 1, "state_dict": head.state_dict(), "metrics": record, "args": vars(args)},
                    Path(args.output_dir) / "best_head.pt",
                )
        print(
            f"epoch {epoch + 1:2d}/{args.epochs} | loss {epoch_loss:.3f} | val top1 {top1:.2f}% top5 {top5:.2f}% "
            f"| {time.time() - t0:.1f}s",
            flush=True,
        )

    print(f"\n=== {args.model} attentive probe (epochs={args.epochs}, {args.train_per_class}/class) ===")
    if best_record:
        print(f"  best val top-1: {best1:.2f}% (epoch {best_record['epoch']})")
    else:
        print(f"  best val top-1: {best1:.2f}%")


if __name__ == "__main__":
    main()
