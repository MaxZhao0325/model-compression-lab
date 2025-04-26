#!/usr/bin/env python
# vision/quantize.py
#
# Structured exactly like prune.py, but implements FP16 quantisation:
# 1) load & fuse Conv-BN-ReLU
# 2) 1-epoch recovery fine-tune (AdamW 1e-4, grad-clip 1.0, AMP)
# 3) cast → FP16 (BatchNorm stays FP32), wrap with autocast
# 4) optional torch.compile
# 5) recalibrate BN (32 batches, no gradients)
# 6) evaluate + log   model | technique | size_MB | params_M | latency_ms | accuracy | note
# -----------------------------------------------------------------------------------------

import sys, os, warnings, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, torchvision as tv
from torch.ao.quantization import fuse_modules
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb,
    benchmark,
    accuracy_classification,
    log_row
)

# --------------------------- data helpers ------------------------------------
def cifar100_loader(batch_size: int = 128, train: bool = False):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if train else transforms.Resize(224),
        transforms.RandomHorizontalFlip()                    if train else transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.CIFAR100(root="data", train=train,
                              download=True, transform=tfm)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=8 if train else 2,
                      pin_memory=True)
# -----------------------------------------------------------------------------

# --------------------------- model helpers -----------------------------------
def load_resnet50_ckpt(path: str):
    m = tv.models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 100)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m

def fuse_resnet(m: nn.Module):
    fuse_modules(m, [["conv1", "bn1", "relu"]], inplace=True)
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        blk = getattr(m, lname)
        for _, b in blk.named_children():
            fuse_modules(b, [["conv1", "bn1", "relu"],
                             ["conv2", "bn2"],
                             ["conv3", "bn3"]], inplace=True)
            if hasattr(b, "downsample") and b.downsample:
                fuse_modules(b.downsample, [["0", "1"]], inplace=True)
    return m

def cast_half_preserve_bn(m: nn.Module):
    m.half()
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.float()
    return m

class AMPWrapper(nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, x):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.core(x)
    # proxy state-dict I/O
    def state_dict(self,*a,**k): return self.core.state_dict(*a,**k)
    def load_state_dict(self,*a,**k): return self.core.load_state_dict(*a,**k)
# -----------------------------------------------------------------------------

# --------------------- stabilised recovery fine-tune -------------------------
def recovery_finetune(model: nn.Module, device: torch.device,
                      epochs: int, lr: float, batch: int):
    if epochs <= 0: return
    loader = cifar100_loader(batch_size=batch, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss()
    model.train().to(device)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"FT {ep+1}", unit="batch", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(xb)
                loss = crit(logits, yb)
            if torch.isnan(loss):             # skip rare bad batch
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    model.eval()
# -----------------------------------------------------------------------------

@torch.inference_mode()
def recalibrate_bn(model: nn.Module, device: torch.device,
                   num_batches: int = 32, batch: int = 128):
    loader = cifar100_loader(batch_size=batch, train=True)
    model.train()
    for i, (xb, _) in enumerate(loader):
        if i == num_batches: break
        model(xb.to(device))
    model.eval()
# -----------------------------------------------------------------------------

# ---------------------------------- main -------------------------------------
def cli():
    p = argparse.ArgumentParser("FP16 quantisation w/ BN recalibration")
    p.add_argument("--ckpt", default="results/resnet50_cifar100_ft.pth")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ft_epochs", type=int, default=1)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--note", default="")
    return p.parse_args()

def main():
    args   = cli()
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"))
    model = load_resnet50_ckpt(args.ckpt)
    model.eval()
    model = fuse_resnet(model)
    recovery_finetune(model, device,
                      epochs=args.ft_epochs,
                      lr=args.ft_lr,
                      batch=args.batch)

    model = cast_half_preserve_bn(model)
    model = AMPWrapper(model).to(device)

    if int(torch.__version__.split(".")[0]) >= 2:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    recalibrate_bn(model, device, num_batches=32, batch=args.batch)

    val_loader = cifar100_loader(batch_size=args.batch, train=False)
    acc  = accuracy_classification(model, val_loader, device)
    lat  = benchmark(model, val_loader, device)
    size = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    log_row("results/metrics.csv",
            model="resnet50",
            technique=f"fused_fp16+ft{args.ft_epochs}",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note=args.note or
                 f"BN-recal, AdamW {args.ft_lr}, clip 1.0")

    print(f"✅  {acc:.2f}% acc | {lat} ms/batch | {size} MB | {params} M params")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()