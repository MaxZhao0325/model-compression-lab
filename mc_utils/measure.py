# mc_utils/measure.py
import os, time, torch, pandas as pd
from tqdm.auto import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# ── Determinism setup ───────────────────────────────────────────────
import random, numpy as np, torch
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # enforce deterministic algorithms (may slow some ops)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def model_size_mb(model):
    tmp = "/tmp/_tmp_weights.pt"
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return round(mb, 2)

@torch.inference_mode()
def benchmark(model, dataloader, device="cuda", reps=50):
    """
    Returns avg ms per batch.
    Uses CUDA events if device=='cuda' and CUDA is available,
    otherwise falls back to a simple time.time() loop on CPU.
    """
    model.eval().to(device)
    batch = next(iter(dataloader))
    # unpack images or input_ids
    x = batch[0] if isinstance(batch, (list,tuple)) else batch.get("input_ids", batch)

    x = x.to(device)
    # warm-up
    for _ in range(10):
        _ = model(x)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        return round(start.elapsed_time(end) / reps, 2)
    else:
        t0 = time.time()
        for _ in range(reps):
            _ = model(x)
        return round(1e3 * (time.time() - t0) / reps, 2)

def accuracy_classification(model, dataloader, device="cuda"):
    """Top-1 acc for image classification, with a tqdm loop."""
    model.eval().to(device)
    correct = total = 0
    with torch.inference_mode():
        for x, y in tqdm(dataloader, desc="Eval", unit="batch"):
            preds = model(x.to(device)).argmax(-1).cpu()
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return round(100 * correct / total, 2)

def log_row(csv_path, **row):
    df = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)