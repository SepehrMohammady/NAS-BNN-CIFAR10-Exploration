# Small script to check OPs (e.g., check_ops.py)
import torch
import models # Assuming models/__init__.py is set up

# Load the CIFAR-10 model structure (no need for trained weights here)
model_arch = models.__dict__['superbnn_cifar10']()

smallest_ops = model_arch.get_ops(model_arch.smallest_cand)[2] # Index 2 is total_ops
biggest_ops = model_arch.get_ops(model_arch.biggest_cand)[2]

print(f"Smallest candidate OPs: {smallest_ops:.2f} M")
print(f"Biggest candidate OPs: {biggest_ops:.2f} M")