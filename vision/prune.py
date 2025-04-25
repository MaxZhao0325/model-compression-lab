#!/usr/bin/env python
# Prune ResNet-50 (global unstructured L1) and log metrics
import argparse, torch, torchvision as tv, torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from mc_utils.measure import model_size_mb, benchmark, log_row
from pathlib import Path

def get_val_loader(root: str, batch_size: int = 64):
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    ds = tv.datasets.ImageNet(root=root, split="val", transform=weights.transforms())
    return DataLoader(ds, batch_size=batch_size, num_workers=8), len(ds), weights

@torch.inference_mode()
def accuracy(model, loader, total, device="cuda"):
    model.eval().to(device)
    correct = 0
    for x, y in loader:
        preds = model(x.to(device)).argmax(-1).cpu()
        correct += (preds == y).sum().item()
    return round(100 * correct / total, 2)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_loader, n_items, _ = get_val_loader(args.data_dir, args.batch_size)

    # ----- load pretrained -----
    model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)

    # ----- global pruning -----
    params = [(m, "weight") for m in model.modules()
              if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                              amount=args.sparsity)

    # remove masks (so we can `torch.save`)
    for m, _ in params:
        prune.remove(m, "weight")

    # (optional) quick 1-epoch fine-tune could go here

    # ----- metrics -----
    top1 = accuracy(model, val_loader, n_items, device)
    lat  = benchmark(model, val_loader, device)
    size = model_size_mb(model)
    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    log_row("results/metrics.csv",
            model="resnet50", technique=f"prune-{args.sparsity}",
            size_MB=size, params_M=params_m, latency_ms=lat,
            accuracy=top1, note="unstructured L1")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/imagenet", help="ImageNet val root")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sparsity",  type=float, default=0.5,
                   help="fraction of weights to prune")
    main(p.parse_args())