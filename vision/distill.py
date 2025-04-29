#!/usr/bin/env python
# vision/distill.py
#
# Knowledge-distillation for CIFAR-100
# ─────────────────────────────────────────────────────────────────────────────
#   • Teacher = the fine-tuned ResNet-50 you trained earlier
#   • Student = choose from {resnet18, resnet34, mobilenet_v2}
#   • Loss    = α·CE(student, y)  +  (1-α)·T²·KL(softmax_t/T, softmax_s/T)
#   • Mixed-precision (AMP) support on CUDA
#   • Same evaluation / logging utilities as quantize.py
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, argparse, warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb, benchmark, accuracy_classification, log_row
)

# ───────────────────────────── data ──────────────────────────────────────────
def cifar100_loader(batch_size=128, train=False):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if train else transforms.Resize(224),
        transforms.RandomHorizontalFlip()                    if train else transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.CIFAR100("data", train=train, download=True, transform=tfm)
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=train,
                      num_workers=min(8, os.cpu_count() // 2),
                      pin_memory=True)

# ──────────────────────────── models ─────────────────────────────────────────
def get_student(name: str, num_classes: int = 100) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = tv.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet34":
        m = tv.models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name in {"mobilenet_v2", "mobilenetv2"}:
        m = tv.models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown student architecture '{name}'")
    return m

def load_teacher(path: str) -> nn.Module:
    t = tv.models.resnet50(weights=None)
    t.fc = nn.Linear(t.fc.in_features, 100)
    t.load_state_dict(torch.load(path, map_location="cpu"))
    t.eval()
    for p in t.parameters(): p.requires_grad = False
    return t

# ─────────────────────────── KD loss ─────────────────────────────────────────
class DistillLoss(nn.Module):
    """
    α·CE + (1-α)·T²·KL  (matches Hinton et al. 2015)
    """
    def __init__(self, T: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T     = T
        self.alpha = alpha
        self.ce    = nn.CrossEntropyLoss()
        self.kl    = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits_s, logits_t, targets):
        T, α = self.T, self.alpha
        # hard-label branch
        loss_ce = self.ce(logits_s, targets)
        # soft-label branch
        p_soft = nn.functional.log_softmax(logits_s / T, dim=1)
        q_soft = nn.functional.softmax   (logits_t.detach() / T, dim=1)
        loss_kl = self.kl(p_soft, q_soft) * (T * T)
        return α * loss_ce + (1 - α) * loss_kl

# ───────────────────────── training loop ─────────────────────────────────────
def distil(student, teacher, device,
           epochs=10, lr=5e-4, batch=256, T=4.0, alpha=0.5):
    loader = cifar100_loader(batch_size=batch, train=True)
    opt    = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn= DistillLoss(T=T, alpha=alpha)
    student.train().to(device)
    teacher.to(device).eval()

    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"KD {ep+1}/{epochs}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits_s = student(xb)
                logits_t = teacher(xb)
                loss = loss_fn(logits_s, logits_t, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    student.eval()

# ───────────────────────────── CLI ───────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser("Knowledge distillation on CIFAR-100")
    p.add_argument("--teacher_ckpt", default="results/resnet50_cifar100_ft.pth")
    p.add_argument("--student",      default="resnet18",
                   choices=["resnet18", "resnet34", "mobilenet_v2"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=5e-4)
    p.add_argument("--T",      type=float, default=4.0, help="temperature")
    p.add_argument("--alpha",  type=float, default=0.5,
                   help="hard-label weight (0=only soft, 1=only hard)")
    p.add_argument("--device", default=None)
    p.add_argument("--note",   default="")
    return p.parse_args()

# ───────────────────────────── main ──────────────────────────────────────────
def main():
    args = _cli()
    device = torch.device(
        args.device or (
            "cuda" if torch.cuda.is_available()      else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    )

    teacher = load_teacher(args.teacher_ckpt)
    student = get_student(args.student).to(device)

    distil(student, teacher, device,
           epochs=args.epochs, lr=args.lr,
           batch=args.batch, T=args.T, alpha=args.alpha)

    # ── evaluation & logging ────────────────────────────────────────────────
    val_loader = cifar100_loader(batch_size=args.batch, train=False)
    acc  = accuracy_classification(student, val_loader, device)
    lat  = benchmark(student, val_loader, device)
    size = model_size_mb(student)
    params = round(sum(p.numel() for p in student.parameters())/1e6, 2)

    ckpt_name = f"results/{args.student}_cifar100_kd_{args.epochs}e.pth"
    os.makedirs("results", exist_ok=True)
    torch.save(student.state_dict(), ckpt_name)

    log_row("results/metrics.csv",
            model=args.student,
            technique=f"distill_T{args.T}_α{args.alpha}",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note=args.note or
                 f"KDist, epochs={args.epochs}, lr={args.lr}")

    print(f"✅  {acc:.2f}% acc | {lat} ms/batch | "
          f"{size} MB | {params} M params | saved → {ckpt_name}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()