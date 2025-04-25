import torch, torchvision as tv
from torch.ao.quantization import get_default_qconfig, prepare, convert
from mc_utils.measure import model_size_mb, benchmark, log_row
device="cpu"      # static int8 shines on CPUs

model_fp32 = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
model_fp32.eval()
model_fp32.fuse_model()  # fuse Conv+BN+ReLU

model_fp32.qconfig = get_default_qconfig("fbgemm")
model_prepared = prepare(model_fp32, inplace=False)

# calibration (few batches)
calib_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)
for x,_ in calib_loader:
    model_prepared(x)
    if calib_loader.batch_sampler.batch_size * calib_loader.batch_sampler.num_batches >= 512: break

model_int8 = convert(model_prepared)

acc = accuracy(model_int8, val_loader)
lat = benchmark(model_int8, val_loader, device="cpu")
sz  = model_size_mb(model_int8)

log_row("results/metrics.csv",
        model="resnet50", technique="int8-PTQ",
        size_MB=sz, params_M=round(sum(p.numel() for p in model_int8.parameters())/1e6,2),
        latency_ms=lat, accuracy=acc, note="post-train static (CPU)")