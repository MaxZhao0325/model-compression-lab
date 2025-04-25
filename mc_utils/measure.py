# mc_utils/measure.py
import os, time, torch, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

def model_size_mb(model):
    tmp = "/tmp/tmp.pt"
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return round(mb, 2)

@torch.inference_mode()
def benchmark(model, dataloader, device="cuda", reps=50):
    model.to(device).eval()
    it = iter(dataloader)
    imgs, _ = next(it)
    imgs = imgs.to(device)
    # warm-up
    for _ in range(10):
        _ = model(imgs)
    # measure latency
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        _ = model(imgs)
    end.record(); torch.cuda.synchronize()
    return round(start.elapsed_time(end) / reps, 2)  # ms per batch

def log_row(csv_path, **kwargs):
    df = pd.DataFrame([kwargs])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)