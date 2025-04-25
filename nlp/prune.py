import torch, torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification
from mc_utils.measure import model_size_mb, log_row
# load tokenizer + dataset as in baseline …

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
modules = [(m, "weight") for m in model.modules() if isinstance(m, torch.nn.Linear)]
prune.global_unstructured(modules, pruning_method=prune.L1Unstructured, amount=0.4)
for m, _ in modules: prune.remove(m, "weight")

# quick 1-epoch tune to recover accuracy (optional)

metrics = trainer.evaluate(ds["validation"])
latency  # reuse snippet above

log_row("results/metrics.csv", model="bert-base", technique="prune-0.4",
        size_MB=model_size_mb(model),
        params_M=…, latency_ms=…, accuracy=…, note="global linear L1")