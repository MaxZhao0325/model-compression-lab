#!/usr/bin/env python
# INT8 PTQ for ResNet-50 (CPU) and log metrics
import argparse, torch, torchvision as tv
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader
from mc_utils.measure import model_size_mb, benchmark, log_row

def get_val_loader(root: str, batch_size: int = 32):
    weights = tv.models.ResNet50_Weights.IMAGENET1K_V2
    ds = tv.datasets.ImageNet(root=root, split="val", transform=weights.transforms())
    return DataLoader(ds, batch_size=batch_size, num_workers=8), len(ds), weights

@torch.inference_mode()
def accuracy(model, loader, total):
    model.eval()
    correct = 0
    for x, y in loader:
        preds = model(x).argmax(-1)
        correct += (preds == y).sum().item()
    return round(100 * correct / total, 2)

def main(args):
    val_loader, n_items, _ = get_val_loader(args.data_dir)
    calib_loader = DataLoader(val_loader.dataset, batch_size=32, shuffle=True)

    # ---- load & fuse ----
    model = tv.models.quantization.resnet50(weights="DEFAULT")
    model.eval()

    # ---- prepare ----
    model.qconfig = get_default_qconfig("fbgemm")
    model_prepared = prepare(model, inplace=False)

    # ---- calibration (≈512 images) ----
    with torch.inference_mode():
        for i, (x, _) in enumerate(calib_loader):
            model_prepared(x)
            if (i + 1) * calib_loader.batch_size >= 512:
                break

    # ---- convert to int8 ----
    model_int8 = convert(model_prepared)

    # ---- metrics ----
    top1 = accuracy(model_int8, val_loader, n_items)
    lat  = benchmark(model_int8, val_loader, device="cpu")
    size = model_size_mb(model_int8)
    params_m = round(sum(p.numel() for p in model_int8.parameters()) / 1e6, 2)

    log_row("results/metrics.csv",
            model="resnet50", technique="int8-PTQ",
            size_MB=size, params_M=params_m, latency_ms=lat,
            accuracy=top1, note="static qconfig=fbgemm, CPU")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/imagenet")
    main(ap.parse_args())