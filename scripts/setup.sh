# scripts/setup.sh
conda create -n mc24 python=3.10 -y
conda activate mc24

# core libs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets evaluate accelerate
pip install torchmetrics pandas tqdm rich

# (optional) speed tools
pip install onnx psutil