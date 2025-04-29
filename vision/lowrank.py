#!/usr/bin/env python
# vision/lowrank.py
#
# Low-Rank factorisation of every spatial Conv2d in a fine-tuned
# ResNet-50 (CIFAR-100).  We:
#   1. load the FP32 checkpoint you already trained;
#   2. replace each k×k Conv (k>1) with   Conv(Cin→r,k×k) + Conv(r→Cout,1×1)
#      obtained via truncated SVD;
#   3. run a short FP32 recovery fine-tune (AdamW);
#   4. (optional) torch.compile;
#   5. evaluate & log   size_MB | params_M | latency_ms | accuracy.
# -------------------------------------------------------------------------

import sys, os, argparse, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb, benchmark, accuracy_classification, log_row
)

# ──────────────────────── dataset helpers ───────────────────────────────────
def cifar100_loader(batch_size=128, train=False):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.8, 1.0)) if train else transforms.Resize(224),
        transforms.RandomHorizontalFlip()              if train else transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.CIFAR100("data", train=train,
                              download=True, transform=tfm)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=min(8, os.cpu_count() // 2),
                      pin_memory=True)

# ──────────────────────── model helpers ────────────────────────────────────
def load_resnet50_ckpt(path: str):
    m = tv.models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 100)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m

def svd_factorise_conv(conv: nn.Conv2d, rank: int) -> nn.Sequential:
    """
    Decompose a Conv2d (Cin→Cout, k×k) into
        Conv2d(Cin→rank, k×k)  +  Conv2d(rank→Cout, 1×1)
    via truncated SVD of the weight.
    """
    W = conv.weight.data           # (Cout, Cin, kH, kW)
    Cout, Cin, kH, kW = W.shape
    Wmat = W.reshape(Cout, -1)     # (Cout, Cin·kH·kW)

    # ── truncated SVD ──────────────────────────────────────────────────────
    U, S, Vh = torch.linalg.svd(Wmat, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank]

    W1 = Vh_r.reshape(rank, Cin, kH, kW)          # first conv weight
    W2 = (U_r * S_r).reshape(Cout, rank, 1, 1)    # second conv weight

    conv1 = nn.Conv2d(Cin,  rank, (kH, kW),
                      stride=conv.stride,
                      padding=conv.padding,
                      dilation=conv.dilation,
                      groups=conv.groups,
                      bias=False)
    conv2 = nn.Conv2d(rank, Cout, 1, bias=True)

    with torch.no_grad():
        conv1.weight.copy_(W1)
        conv2.weight.copy_(W2)
        if conv.bias is not None:
            conv2.bias.copy_(conv.bias)

    return nn.Sequential(conv1, conv2)

def factorise_module(m: nn.Module, rank_frac: float) -> int:
    """
    Recursively replace every spatial Conv2d by its low-rank pair.
    Returns the number of layers replaced.
    """
    n = 0
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            max_rank = min(child.out_channels,
                           child.in_channels * child.kernel_size[0] * child.kernel_size[1])
            rank = max(1, int(round(rank_frac * max_rank)))
            setattr(m, name, svd_factorise_conv(child, rank))
            n += 1
        else:
            n += factorise_module(child, rank_frac)
    return n

# ─────────────────── recovery fine-tune (FP32) ──────────────────────────────
def recovery_finetune(model, device, epochs, lr, batch):
    if epochs <= 0:
        return
    loader = cifar100_loader(batch_size=batch, train=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    model.train().to(device)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"recovery-ft {ep+1}/{epochs}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()

# ───────────────────────────── CLI ──────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser("Low-Rank factorisation for ResNet-50")
    p.add_argument("--ckpt",      default="results/resnet50_cifar100_ft.pth",
                   help="path to fine-tuned FP32 checkpoint")
    p.add_argument("--rank_frac", type=float, default=0.5,
                   help="fraction of maximum rank to keep (0<r≤1)")
    p.add_argument("--ft_epochs", type=int,   default=1,
                   help="recovery fine-tune epochs")
    p.add_argument("--ft_lr",     type=float, default=1e-4)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--no_compile",action="store_true")
    p.add_argument("--device",    default=None)
    p.add_argument("--note",      default="")
    return p.parse_args()

# ───────────────────────────── main ─────────────────────────────────────────
def main():
    args   = _cli()
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available()      else
        "mps"  if torch.backends.mps.is_available() else "cpu"))

    # 1) load model ----------------------------------------------------------
    model = load_resnet50_ckpt(args.ckpt)
    model.eval()

    # 2) factorise -----------------------------------------------------------
    n_replaced = factorise_module(model, args.rank_frac)
    print(f"► factorised {n_replaced} conv layers (rank_frac={args.rank_frac})")

    # 3) recovery fine-tune --------------------------------------------------
    recovery_finetune(model, device,
                      epochs=args.ft_epochs,
                      lr=args.ft_lr,
                      batch=args.batch)

    # 4) optional torch.compile ---------------------------------------------
    if (not args.no_compile) and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # 5) evaluation ----------------------------------------------------------
    val_loader = cifar100_loader(batch_size=args.batch, train=False)
    acc  = accuracy_classification(model, val_loader, device)
    lat  = benchmark(model, val_loader, device)
    size = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters())/1e6, 2)

    # 6) save & log ----------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    ckpt = f"results/resnet50_lrf{args.rank_frac:.2f}_ft{args.ft_epochs}.pth"
    torch.save(model.state_dict(), ckpt)

    log_row("results/metrics.csv",
            model="resnet50",
            technique=f"lowrank{args.rank_frac:.2f}+ft{args.ft_epochs}",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note=args.note or f"AdamW {args.ft_lr}")

    print(f"✅  {acc:.2f}% acc | {lat} ms/batch | {size} MB | "
          f"{params} M params | saved → {ckpt}")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()