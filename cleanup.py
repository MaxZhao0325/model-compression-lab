# cleanup.py
import shutil
import os

def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted folder: {path}")
    else:
        print(f"Folder not found: {path}")

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted file: {path}")
    else:
        print(f"File not found: {path}")

def main():
    # for vision models
    # Delete CIFAR-100 dataset
    delete_folder("data")

    # Delete benchmark results
    # delete_folder("results")

    # Delete torchvision model cache
    torch_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    delete_folder(torch_cache)


    # for BERT transformers
    # Delete Huggingface datasets cache (e.g., SST-2)
    delete_folder(os.path.expanduser("~/.cache/huggingface/datasets"))

    # Delete Huggingface transformers model cache (e.g., bert-base)
    delete_folder(os.path.expanduser("~/.cache/huggingface/transformers"))

    # Delete Huggingface Trainer output folder
    delete_folder("/tmp/out")

    # Delete results CSV
    # delete_file("results/metrics.csv")

if __name__ == "__main__":
    main()