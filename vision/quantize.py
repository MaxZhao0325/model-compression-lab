#!/usr/bin/env python
# vision/quantize.py
#
# Light-weight “FP16 quantisation” (weight casting) that *keeps all accuracy*.
#   • loads the fine-tuned ResNet-50 for CIFAR-100
#   • (optional) Conv-BN fusion            –‐  --fuse
#   • 1-epoch FP32 recovery fine-tune
#   • safe clamp → cast to FP16 (CUDA only)
#   • wraps in autocast for inference
#   • optional torch.compile               –‐  --compile / --no_compile
#   • logs accuracy / latency / size
# ---------------------------------------------------------------------------

import sys, os, argparse, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, torchvision as tv
from torch.ao.quantization import fuse_modules
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb, benchmark, accuracy_classification, log_row
)

# ───────────────────────────── data ──────────────────────────────────────────
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
                      num_workers=min(8, os.cpu_count() // 2),
                      pin_memory=True)

# ──────────────────────────── helpers ────────────────────────────────────────
def load_resnet50_ckpt(path: str):
    m = tv.models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 100)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m

def fuse_resnet(m: nn.Module):
    fuse_modules(m, [["conv1", "bn1", "relu"]], inplace=True)
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        for _, blk in getattr(m, lname).named_children():
            fuse_modules(blk,
                         [["conv1", "bn1", "relu"],
                          ["conv2", "bn2"],
                          ["conv3", "bn3"]],
                         inplace=True)
            if getattr(blk, "downsample", None):
                fuse_modules(blk.downsample, [["0", "1"]], inplace=True)
    return m

def clamp_and_half(m: nn.Module):
    MAX_F16 = 65504.0
    with torch.no_grad():
        for p in m.parameters():
            p.clamp_(-MAX_F16, MAX_F16)
    m.half()
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.float()
    return m

class AMPWrapper(nn.Module):
    """Inference-only autocast wrapper (used on CUDA)."""
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, x):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.core(x)
    def state_dict(self,*a,**k):   return self.core.state_dict(*a,**k)
    def load_state_dict(self,*a,**k): return self.core.load_state_dict(*a,**k)

def recovery_finetune(model: nn.Module, device, epochs=1, lr=1e-4, batch=256):
    if epochs <= 0: return
    loader = cifar100_loader(batch_size=batch, train=True)
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    crit = nn.CrossEntropyLoss()
    model.train().to(device)

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"recovery-ft {ep+1}/{epochs}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()

@torch.inference_mode()
def recalibrate_bn(model: nn.Module, device, batches=32, batch_size=128):
    if not any(isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
               for m in model.modules()):
        return
    loader = cifar100_loader(batch_size=batch_size, train=True)
    model.train()
    for i, (xb, _) in enumerate(loader):
        if i == batches: break
        model(xb.to(device))
    model.eval()

# ────────────────────────────── CLI ──────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser("FP16 weight-casting with safe recovery")
    p.add_argument("--ckpt", default="results/resnet50_cifar100_ft.pth")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ft_epochs", type=int, default=1)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--fuse",   action="store_true",
                   help="explicitly enable Conv-BN fusion")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--note", default="")
    return p.parse_args()

# ───────────────────────────── main ──────────────────────────────────────────
def main():
    args   = _cli()
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available()      else
        "mps"  if torch.backends.mps.is_available() else "cpu"))

    model = load_resnet50_ckpt(args.ckpt)

    if args.fuse:
        model = fuse_resnet(model)

    # ---- 1-epoch stabilising fine-tune (FP32) ------------------------------
    recovery_finetune(model, device,
                      epochs=args.ft_epochs,
                      lr=args.ft_lr,
                      batch=args.batch)

    # ---- BN recalibration (still FP32) -------------------------------------
    #recalibrate_bn(model, device)

    # ---- Cast to FP16 *only* on CUDA ---------------------------------------
    if device.type == "cuda":
        model = clamp_and_half(model)
        model = AMPWrapper(model).to(device)
    else:                                   # CPU / MPS ➜ stay in FP32
        model.to(device)

    # ---- optional torch.compile -------------------------------------------
    if (not args.no_compile) and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # ---- evaluation & logging ---------------------------------------------
    val_loader = cifar100_loader(batch_size=args.batch, train=False)
    acc  = accuracy_classification(model, val_loader, device)
    lat  = benchmark(model, val_loader, device)
    size = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters())/1e6, 2)

    # Did we actually torch.compile?
    compiled = (
        (not args.no_compile)
        and hasattr(torch, "compile")
        and device.type == "cuda"
    )

    # Build a clear 'technique' string via f-strings
    fuse_str    = "fuse" if args.fuse else "nofuse"
    ft_str      = f"recoveryFT{args.ft_epochs}e"
    compile_str = "compile" if compiled else "nocompile"
    technique   = f"{fuse_str}_fp16_{ft_str}_{compile_str}"

    # Build a detailed 'note' string
    note_fields = [
        f"AdamW_lr={args.ft_lr}",
        f"batch={args.batch}",
        "clip=1.0",
    ]
    if args.note:
        note_fields.append(f"note={args.note}")
    note = ", ".join(note_fields)

    log_row("results/metrics.csv",
            model="resnet50",
            technique=technique,
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note=note)

    print(f"✅  {acc:.2f}% acc | {lat} ms/batch | {size} MB | {params} M params")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()