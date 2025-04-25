from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from mc_utils.measure import model_size_mb, log_row
import torch, time, numpy as np

device="cuda"
model_name="bert-base-uncased"
ds = load_dataset("glue", "sst2")
tok = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tok(batch["sentence"], truncation=True, padding="max_length", max_length=128)
ds = ds.map(tokenize, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
trainer = Trainer(model=model,
                  args=TrainingArguments(output_dir="/tmp/out", per_device_eval_batch_size=32))
metrics = trainer.evaluate(ds["validation"])

# latency (single-batch forward pass repeated)
inputs = {k:v.to(device) for k,v in ds["validation"][:32].items() if k!="label"}
torch.cuda.synchronize(); t0=time.time()
with torch.no_grad():
    for _ in range(100):
        _=model(**inputs)
torch.cuda.synchronize(); latency = 1e3*(time.time()-t0)/100  # ms

log_row("results/metrics.csv",
        model="bert-base", technique="baseline",
        size_MB=model_size_mb(model),
        params_M=round(sum(p.numel() for p in model.parameters())/1e6,2),
        latency_ms=round(latency,2),
        accuracy=round(100*metrics["eval_accuracy"],2),
        note="SST-2")