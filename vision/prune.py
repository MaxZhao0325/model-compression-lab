#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch_pruning as tp
from tqdm.auto import tqdm

from mc_utils.measure import (
    model_size_mb,
    benchmark,
    accuracy_classification,
    log_row
)

def cifar100_loader(batch_size=128):
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    ds = tv.datasets.CIFAR100(
        root="data", train=False, download=True, transform=tfm
    )
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=2)

def prune_structured_tp(model, example_inputs, amount: float):
    DG = tp.DependencyGraph().build_dependency(
        model, example_inputs=example_inputs
    )
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            w = layer.weight.detach()
            chn_norm = w.abs().sum((1,2,3))
            n_prune = int(chn_norm.numel() * amount)
            if n_prune == 0:
                continue
            idxs = torch.argsort(chn_norm)[:n_prune].tolist()
            try:
                grp = DG.get_pruning_group(
                    layer, tp.prune_conv_out_channels, idxs=idxs
                )
                grp.prune()
            except ValueError:
                continue
    return model

@torch.inference_mode()
def recalibrate_bn(model, loader, num_batches: int, device: torch.device):
    model.train()
    for i, (xb, _) in enumerate(loader):
        model(xb.to(device))
        if i + 1 == num_batches:
            break
    model.eval()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sparsity", type=float, default=0.125,
                   help="fraction of channels to prune")
    p.add_argument("--batch",    type=int,   default=256)
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # 1) Load the checkpointed baseline
    ckpt  = "results/resnet50_cifar100_ft.pth"
    model = tv.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # 2) Structured channel pruning
    example = torch.randn(1,3,224,224).to(device)
    model.to(device)
    model = prune_structured_tp(model, (example,), args.sparsity)


    # fine-tune the pruned model for one epoch
    
    # Prepare training loader
    train_loader = DataLoader(
        tv.datasets.CIFAR100(root="data", train=True, download=True, transform=transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])),
        batch_size=args.batch, shuffle=True, num_workers=8
    )
    
    # Set up optimizer & loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    # One epoch of fine-tuning
    model.train()
    for x_train, y_train in tqdm(train_loader, desc="Fine-tuning", unit="batch"):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"{loss.item():.3f}")
    
    # Back to eval mode
    model.eval()

    val_loader = cifar100_loader(batch_size=args.batch)

    # recalibrate batch norm will increase the acc if we do not retrain model
    # recalibrate_bn(model, train_loader, num_batches=32, device=device)

    x0, y0 = next(iter(val_loader))
    preds0 = model(x0.to(device)).argmax(-1).cpu()
    print("preds:", preds0[:10].tolist())
    print("labels:", y0[:10].tolist())

    # 4) Measure
    acc    = accuracy_classification(model, val_loader, device)
    lat    = benchmark(model, val_loader, device)
    size   = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters())/1e6,2)

    log_row("results/metrics.csv",
            model="resnet50",
            technique=f"prune-struct-{args.sparsity}",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note="CIFAR-100 structured L1 + head-FT + retraining")

if __name__ == "__main__":
    main()