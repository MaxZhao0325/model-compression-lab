import torch
from transformers import AutoModelForSequenceClassification
from torch.ao.quantization import quantize_dynamic
from mc_utils.measure import model_size_mb, log_row

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# eval
# … (same dataset + trainer but pass qmodel)
metrics = trainer.evaluate(ds["validation"])
latency  # CPU latency similar to ResNet snippet (run on cpu)

log_row("results/metrics.csv",
        model="bert-base", technique="int8-dynamic",
        size_MB=model_size_mb(qmodel),
        params_M=round(sum(p.numel() for p in qmodel.parameters())/1e6,2),
        latency_ms=…, accuracy=round(100*metrics["eval_accuracy"],2),
        note="HF dynamic quant")