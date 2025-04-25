import torch, torchvision as tv
from mc_utils.measure import model_size_mb, benchmark, log_row

device = "cuda"
model  = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
transform = tv.models.ResNet50_Weights.IMAGENET1K_V2.transforms()

val_set = tv.datasets.ImageNet(root="data/imagenet", split="val", transform=transform)  # 50 k imgs
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, num_workers=8)

# Top-1 accuracy
@torch.inference_mode()
def accuracy():
    correct = total = 0
    for x,y in val_loader:
        logits = model(x.to(device))
        preds  = logits.argmax(-1).cpu()
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return round(100*correct/total, 2)

acc = accuracy()
lat = benchmark(model, val_loader, device)
sz  = model_size_mb(model)

log_row("results/metrics.csv",
        model="resnet50", technique="baseline",
        size_MB=sz, params_M=round(sum(p.numel() for p in model.parameters())/1e6,2),
        latency_ms=lat, accuracy=acc, note="ImageNet-val")