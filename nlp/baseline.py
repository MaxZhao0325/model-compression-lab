# nlp/baseline.py
#!/usr/bin/env python
import time, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm
from mc_utils.measure import model_size_mb, log_row

def build_dataset(tokenizer, max_len=128):
    # downloads GLUE SST-2
    ds = load_dataset("glue", "sst2")
    ds = ds.map(lambda b: tokenizer(
                     b["sentence"],
                     truncation=True,
                     padding="max_length",
                     max_length=max_len),
                batched=True, desc="Tokenize")
    ds.set_format(type="torch",
                  columns=["input_ids","attention_mask","label"])
    return ds

def accuracy_and_latency(model, dataloader, device, reps=100):
    model.eval().to(device)
    correct = total = 0

    # accuracy with progress bar
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Eval", unit="batch"):
            inputs = {k: batch[k].to(device)
                      for k in ["input_ids","attention_mask"]}
            logits = model(**inputs).logits
            preds  = logits.argmax(-1).cpu()
            correct += (preds == batch["label"]).sum().item()
            total   += batch["label"].size(0)
    acc = round(100 * correct / total, 2)

    # warm-up
    first = next(iter(dataloader))
    inputs = {k: first[k].to(device) for k in ["input_ids","attention_mask"]}
    with torch.inference_mode():
        for _ in range(10):
            _ = model(**inputs)

    # latency reps with progress bar
    if device.type != "cpu":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for _ in tqdm(range(reps), desc="Latency reps"):
            _ = model(**inputs)
    if device.type != "cpu":
        torch.cuda.synchronize()
    lat = round(1e3 * (time.time() - t0) / reps, 2)

    return acc, lat

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = build_dataset(tokenizer, max_len=128)
    val_loader = DataLoader(ds["validation"],
                            batch_size=32, num_workers=4)

    model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2)

    acc, lat = accuracy_and_latency(model, val_loader, device)
    size   = model_size_mb(model)
    params = round(sum(p.numel() for p in model.parameters())/1e6,2)

    log_row("results/metrics.csv",
            model="bert-base", technique="baseline",
            size_MB=size, params_M=params,
            latency_ms=lat, accuracy=acc,
            note="SST-2")

if __name__ == "__main__":
    main()