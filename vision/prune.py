import torch, torchvision as tv, torch.nn.utils.prune as prune
from mc_utils.measure import model_size_mb, benchmark, log_row
device="cuda"; sparsity=0.5

model = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
# global unstructured L1 pruning on all Conv + FC weights
parameters_to_prune = [(m, "weight") for m in model.modules()
                       if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
prune.global_unstructured(parameters_to_prune,
                           pruning_method=prune.L1Unstructured, amount=sparsity)
prune.remove(model.fc, "weight")  # remove mask so we can serialize dense weights+zeros

# (optional) short fine-tune pass could be added here.

# reuse val_loader from baseline
# … (copy transform + loader setup)

acc = accuracy()
lat = benchmark(model, val_loader, device)
sz  = model_size_mb(model)

log_row("results/metrics.csv",
        model="resnet50", technique=f"prune-{sparsity}",
        size_MB=sz, params_M=round(sum(p.numel() for p in model.parameters())/1e6,2),
        latency_ms=lat, accuracy=acc, note="unstructured L1")