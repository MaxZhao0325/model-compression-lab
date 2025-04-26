#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random, numpy as np, torch
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb,
    benchmark,
    accuracy_classification,
    log_row
)

# ─── Deterministic training for the one-time fine-tune ────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ──────────────────────────────────────────────────────────────────

def cifar_loader(batch=256, train=False):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = tv.datasets.CIFAR100(
        root="data", train=train, download=True, transform=tfm
    )
    return DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=2
    )

def quick_finetune(model, loader, epochs=3, lr=1e-3, device="cpu"):
    # freeze backbone, train only the head
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    model.to(device).train()
    for ep in range(epochs):
        pbar = tqdm(loader,
                    desc=f"FT epoch {ep+1}/{epochs}",
                    unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
    model.eval()

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Baseline FT on", device)

    train_loader = cifar_loader(batch=256, train=True)
    val_loader   = cifar_loader(batch=256, train=False)

    # load Imagenet-pretrained ResNet-50 & swap head
    model = tv.models.resnet50(
        weights=tv.models.ResNet50_Weights.IMAGENET1K_V2
    )
    model.fc = nn.Linear(model.fc.in_features, 100)

    quick_finetune(model, train_loader,
                   epochs=3, lr=1e-3, device=device)

    # save checkpoint
    os.makedirs("results", exist_ok=True)
    ckpt = "results/resnet50_cifar100_ft.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved fine-tuned model to {ckpt}")

    # log baseline metrics
    acc    = accuracy_classification(model, val_loader, device)
    lat    = benchmark(model, val_loader, device)
    size   = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters())/1e6,2)

    log_row("results/metrics.csv",
            model="resnet50", technique="baseline",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note="CIFAR-100 head-tuned (3 epochs)")

if __name__ == "__main__":
    main()