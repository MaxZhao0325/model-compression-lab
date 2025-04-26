#!/usr/bin/env python
import torch, time
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import load_dataset
from mc_utils.measure import model_size_mb, log_row
from torch.ao.quantization import quantize_dynamic
from tqdm.auto import tqdm

def build_dataset(tok, max_len=128):
    ds = load_dataset("glue", "sst2")
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True,
                              padding="max_length", max_length=max_len),
                batched=True, desc="Tokenize")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def latency(model, batch, reps=100):
    t0=time.time()
    for _ in tqdm(range(reps), desc="Latency reps"):
        _=model(**batch)
    return round(1e3*(time.time()-t0)/reps,2)

def main():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds  = build_dataset(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    trainer = Trainer(model=qmodel,
                      args=TrainingArguments(output_dir="/tmp/out",
                                             per_device_eval_batch_size=32,
                                             no_cuda=True))
    metrics = trainer.evaluate(ds["validation"])

    batch = {k: v[:32] for k,v in ds["validation"][:32].items() if k!="label"}
    lat = latency(qmodel, batch)

    log_row("results/metrics.csv",
            model="bert-base", technique="int8-dynamic",
            size_MB=model_size_mb(qmodel),
            params_M=round(sum(p.numel() for p in qmodel.parameters())/1e6, 2),
            latency_ms=lat, accuracy=round(100*metrics["eval_accuracy"],2),
            note="SST-2 dynamic, CPU")

if __name__ == "__main__":
    main()