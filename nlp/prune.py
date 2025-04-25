#!/usr/bin/env python
# Global linear-layer pruning for BERT-base on SST-2
import argparse, torch, torch.nn.utils.prune as prune, time
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import load_dataset
from mc_utils.measure import model_size_mb, log_row

def build_dataset(tokenizer, max_len=128):
    ds = load_dataset("glue", "sst2")
    ds = ds.map(lambda b: tokenizer(b["sentence"],
                                    truncation=True, padding="max_length",
                                    max_length=max_len), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def latency_ms(model, batch, reps=100):
    device = next(model.parameters()).device
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(**batch)
    torch.cuda.synchronize(); return round(1e3*(time.time()-t0)/reps, 2)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds  = build_dataset(tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2).to(device)

    # ---- global prune over all Linear weights ----
    modules = [(m, "weight") for m in model.modules()
               if isinstance(m, torch.nn.Linear)]
    prune.global_unstructured(modules, prune.L1Unstructured, amount=args.sparsity)
    for m, _ in modules:
        prune.remove(m, "weight")

    # ---- optional quick tune (skipped) ----

    trainer = Trainer(model=model,
                      args=TrainingArguments(output_dir="/tmp/out",
                                             per_device_eval_batch_size=32))
    metrics = trainer.evaluate(ds["validation"])

    # ---- latency ----
    batch = {k: v[:32].to(device) for k, v in ds["validation"][:32].items()
             if k != "label"}
    lat = latency_ms(model, batch)

    log_row("results/metrics.csv",
            model="bert-base", technique=f"prune-{args.sparsity}",
            size_MB=model_size_mb(model),
            params_M=round(sum(p.numel() for p in model.parameters())/1e6, 2),
            latency_ms=lat, accuracy=round(100*metrics["eval_accuracy"], 2),
            note="global linear L1")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sparsity", type=float, default=0.4)
    main(p.parse_args())