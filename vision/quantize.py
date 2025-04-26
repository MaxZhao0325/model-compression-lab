#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, torchvision as tv
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules
from torch.utils.data import DataLoader
from torchvision import transforms
from mc_utils.measure import (model_size_mb, benchmark,
                              accuracy_classification, log_row)
from tqdm.auto import tqdm

def fuse_resnet_manual(model):
    # fuse the very first stem
    fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)

    # fuse each residual block’s conv/BN/ReLU
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for block_name, block in layer.named_children():
            # in each BasicBlock: conv1+bn1+relu, and conv2+bn2
            fuse_modules(block, 
                         [["conv1", "bn1", "relu"], ["conv2", "bn2"]], 
                         inplace=True)
    return model

def cifar100_loader(batch_size=64):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    ds = tv.datasets.CIFAR100(root="data", train=False,
                              download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, num_workers=8)

def main():
    calib_bs = 32
    calib_batches = 16      # 16×32 = 512 images
    val_loader = cifar100_loader(batch_size=128)
    calib_loader = cifar100_loader(batch_size=calib_bs)

    # load + fuse
    model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model = fuse_resnet_manual(model)

    engine = 'qnnpack' if 'qnnpack' in torch.backends.quantized.supported_engines else 'fbgemm'
    torch.backends.quantized.engine = engine
    model.qconfig = get_default_qconfig(engine)

    model_prepared = prepare(model, inplace=False)

    # calibration with tqdm
    with torch.inference_mode():
        for i, (x, _) in tqdm(enumerate(calib_loader),
                              total=calib_batches, desc="Calibrate"):
            model_prepared(x)
            if i + 1 == calib_batches:
                break

    model_int8 = convert(model_prepared)

    acc = accuracy_classification(model_int8, val_loader, device="cpu")
    lat = benchmark(model_int8, val_loader, device="cpu")
    size = model_size_mb(model_int8)
    params = round(sum(p.numel() for p in model_int8.parameters())/1e6, 2)

    note = (
        f"CIFAR-100 static quantization, "
        f"{calib_bs}bs×{calib_batches}batches (≈{calib_bs * calib_batches} images) for calibration, "
        f"engine={engine.upper()}, "
        f"eval=CPU"
    )

    log_row("results/metrics.csv",
            model="resnet50", technique="int8-PTQ",
            size_MB=size, params_M=params, latency_ms=lat,
            accuracy=acc, note=note)

if __name__ == "__main__":
    main()