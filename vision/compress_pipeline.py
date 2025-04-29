#!/usr/bin/env python
# vision/compress_pipeline.py
#
# One-shot pipeline that applies, in order:
#   1. knowledge-distillation (ResNet-50 → ResNet-18 student)
#   2. structured L1 channel-pruning (torch-pruning)
#   3. low-rank factorisation (truncated SVD, rank_frac)
#   4. FP16 weight casting (BN kept FP32)  + autocast wrapper
#
# After each structural change we fine-tune 1 epoch on CIFAR-100.
# Finally we evaluate and log size_MB | params_M | latency_ms | accuracy.
# ------------------------------------------------------------------------

import sys, os, argparse, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import torch_pruning as tp

from mc_utils.measure import (
    model_size_mb, benchmark, accuracy_classification, log_row
)

# ─────────────────────────── data helpers ───────────────────────────────────
def cifar_loader(batch=256, train=False):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, (0.8, 1.0)) if train else transforms.Resize(224),
        transforms.RandomHorizontalFlip()              if train else transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.CIFAR100("data", train=train, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch, shuffle=train,
                      num_workers=min(8, os.cpu_count() // 2), pin_memory=True)

# ───────────────────── 1) knowledge-distillation ────────────────────────────
class DistillLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T, self.alpha = T, alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
    def forward(self, s, t, y):
        T, a = self.T, self.alpha
        return (a * self.ce(s, y) +
                (1-a) * (T*T) * self.kl(
                    nn.functional.log_softmax(s / T, 1),
                    nn.functional.softmax(t.detach() / T, 1)))

def distil_resnet18(teacher_ckpt, device,
                    epochs=10, batch=256, lr=5e-4,
                    T=4.0, alpha=0.5):
    # teacher
    t = tv.models.resnet50(weights=None)
    t.fc = nn.Linear(t.fc.in_features, 100)
    t.load_state_dict(torch.load(teacher_ckpt, map_location="cpu"))
    t.eval().to(device)
    for p in t.parameters(): p.requires_grad = False
    # student
    s = tv.models.resnet18(weights=None)
    s.fc = nn.Linear(s.fc.in_features, 100)
    s.to(device)

    loss_fn = DistillLoss(T, alpha)
    opt = torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_loader = cifar_loader(batch, train=True)
    s.train()
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"KD {ep+1}/{epochs}", unit="batch", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = loss_fn(s(xb), t(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    s.eval()
    return s

# ───────────────────── 2) structured pruning ────────────────────────────────
def prune_structured(model, sparsity, device):
    dg = tp.DependencyGraph().build_dependency(model,
                                               example_inputs=(torch.randn(1,3,224,224)
                                                               .to(device),))
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach()
            n_prune = int(w.size(0) * sparsity)
            if n_prune == 0: continue
            idx = torch.argsort(w.abs().sum((1,2,3)))[:n_prune].tolist()
            try:
                grp = dg.get_pruning_group(m, tp.prune_conv_out_channels, idxs=idx)
                grp.prune()
            except ValueError:
                continue
    return model

# ───────────────────── 3) low-rank factorisation ────────────────────────────
def svd_factorise_conv(conv, rank):
    W = conv.weight.data
    Cout, Cin, kH, kW = W.shape
    Wm = W.reshape(Cout, -1)
    U, S, Vh = torch.linalg.svd(Wm, full_matrices=False)
    Ur, Sr, Vhr = U[:, :rank], S[:rank], Vh[:rank]
    W1 = Vhr.reshape(rank, Cin, kH, kW)
    W2 = (Ur*Sr).reshape(Cout, rank, 1, 1)
    c1 = nn.Conv2d(Cin, rank, (kH,kW), stride=conv.stride,
                   padding=conv.padding, dilation=conv.dilation,
                   groups=conv.groups, bias=False)
    c2 = nn.Conv2d(rank, Cout, 1, bias=True)
    with torch.no_grad():
        c1.weight.copy_(W1); c2.weight.copy_(W2)
        if conv.bias is not None: c2.bias.copy_(conv.bias)
    return nn.Sequential(c1, c2)

def lowrank_model(m, rank_frac):
    n=0
    for name, mod in list(m.named_children()):
        if isinstance(mod, nn.Conv2d) and mod.kernel_size != (1,1):
            max_rank = min(mod.out_channels,
                           mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1])
            r = max(1, int(round(rank_frac*max_rank)))
            setattr(m, name, svd_factorise_conv(mod, r)); n+=1
        else:
            n += lowrank_model(mod, rank_frac)
    return n

# ───────────────────── 4) weight-only FP16 “quantisation” ───────────────────
def clamp_and_half(model):
    LIM = 65504.
    with torch.no_grad():
        for p in model.parameters(): p.clamp_(-LIM, LIM)
    model.half()
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
            mod.float()
    return model

class AMPWrapper(nn.Module):
    def __init__(self, core): super().__init__(); self.core=core
    def forward(self,x):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.core(x)
    def state_dict(self,*a,**k): return self.core.state_dict(*a,**k)

# ───────────────────── recovery fine-tune ───────────────────────────────────
def quick_ft(model, device, epochs, lr, batch):
    if epochs<=0: return
    train_loader = cifar_loader(batch, train=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    model.train().to(device)
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"FT {ep+1}/{epochs}", unit="batch", leave=False)
        for xb,yb in pbar:
            xb,yb=xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    model.eval()

# ───────────────────────── CLI ──────────────────────────────────────────────
def _cli():
    p=argparse.ArgumentParser("4-stage compression pipeline")
    p.add_argument("--teacher_ckpt", default="results/resnet50_cifar100_ft.pth")
    p.add_argument("--kd_epochs", type=int, default=15)
    p.add_argument("--prune_frac", type=float, default=0.1)
    p.add_argument("--rank_frac",  type=float, default=0.6)
    p.add_argument("--ft_each",    type=int,   default=2,
                   help="recovery epochs after prune & lowrank")
    p.add_argument("--batch",      type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--device",     default=None)
    p.add_argument("--note",       default="")
    return p.parse_args()

# ───────────────────────── main ─────────────────────────────────────────────
def main():
    args=_cli()
    device=torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"))

    # 1) knowledge distillation ---------------------------------------------
    student = distil_resnet18(args.teacher_ckpt, device,
                              epochs=args.kd_epochs,
                              batch=args.batch, lr=args.lr)
    print("✓ distillation done")

    # 2) structured pruning --------------------------------------------------
    student.to(device)
    student = prune_structured(student, args.prune_frac, device)
    quick_ft(student, device, args.ft_each, args.lr, args.batch)
    print("✓ pruning + FT done")

    # 3) low-rank factorisation ---------------------------------------------
    n_lr = lowrank_model(student, args.rank_frac)
    print(f"✓ low-rank: factorised {n_lr} convs  (rank_frac={args.rank_frac})")
    quick_ft(student, device, args.ft_each, args.lr, args.batch)

    # 4) FP16 cast -----------------------------------------------------------
    if device.type=="cuda":
        student = clamp_and_half(student)
        student = AMPWrapper(student).to(device)

    # optional compile
    if (not args.no_compile) and hasattr(torch,"compile") and device.type=="cuda":
        student = torch.compile(student, mode="reduce-overhead", fullgraph=False)

    # ─── evaluation ────────────────────────────────────────────────────────
    val_loader = cifar_loader(args.batch, train=False)
    acc  = accuracy_classification(student, val_loader, device)
    lat  = benchmark(student, val_loader, device)
    size = model_size_mb(student)
    params = round(sum(p.numel() for p in student.parameters())/1e6,2)

    # save and log
    os.makedirs("results", exist_ok=True)
    ckpt=f"results/resnet18_combo.pth"
    torch.save(student.state_dict(), ckpt)

    log_row("results/metrics.csv",
            model="resnet18",
            technique=(f"KD{args.kd_epochs}e→prune{args.prune_frac}→LR{args.rank_frac}"
                       f"→FP16+ft{args.ft_each}"),
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note=args.note)

    print(f"\n✅  final: {acc:.2f}% acc | {lat} ms/batch | "
          f"{size} MB | {params} M params | saved → {ckpt}")

# ────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    torch.backends.cuda.matmul.allow_tf32=True
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()