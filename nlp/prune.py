#!/usr/bin/env python
import argparse, torch, torch.nn.utils.prune as prune, time
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import load_dataset
from mc_utils.measure import model_size_mb, log_row
from tqdm.auto import tqdm

def build_dataset(tok, max_len=128):
    ds = load_dataset("glue", "sst2")
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True,
                              padding="max_length", max_length=max_len),
                batched=True, desc="Tokenize")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def latency(model, batch, reps=100):
    torch.cuda.synchronize(); t0=time.time()
    for _ in tqdm(range(reps), desc="Latency reps"):
        _ = model(**batch)
    torch.cuda.synchronize(); return round(1e3*(time.time()-t0)/reps, 2)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds   = build_dataset(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2).to(device)

    # global linear pruning
    mods = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]
    prune.global_unstructured(mods, prune.L1Unstructured, amount=args.sparsity)
    for m, _ in mods: prune.remove(m, "weight")

    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir="/tmp/out",
                                             per_device_eval_batch_size=32,
                                             logging_steps=500))
    metrics = trainer.evaluate(ds["validation"])

    batch = {k: v[:32].to(device) for k,v in ds["validation"][:32].items()
             if k!="label"}
    lat = latency(model, batch)

    log_row("results/metrics.csv",
            model="bert-base", technique=f"prune-{args.sparsity}",
            size_MB=model_size_mb(model),
            params_M=round(sum(p.numel() for p in model.parameters())/1e6, 2),
            latency_ms=lat, accuracy=round(100*metrics["eval_accuracy"],2),
            note="SST-2 linear L1")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.4)
    main(ap.parse_args())