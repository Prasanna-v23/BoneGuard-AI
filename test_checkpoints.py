import os
import sys
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath("05_Code"))
from models.densenet121_dgcbam import build_densenet121_dgcbam, get_model_parameter_counts

ckpt_exp_e = os.path.abspath("06_Model_Results/DGCBAM_DomainRobustness/Exp_E_ClassWeight/best_model.pth")
ckpt_final = os.path.abspath("06_Model_Results/DGCBAM_DomainRobustness/Final_Model/best_model.pth")
ckpt_dgcbam_orig = os.path.abspath("06_Model_Results/DenseNet121_DGCBAM/best_model.pth")

print("Checking Checkpoints:")
for name, p in [("Exp_E_ClassWeight", ckpt_exp_e), ("Final_Model", ckpt_final), ("DenseNet121_DGCBAM", ckpt_dgcbam_orig)]:
    if os.path.exists(p):
        data = torch.load(p, map_location="cpu")
        print(f"[{name}] exists, size: {os.path.getsize(p)/(1024*1024):.2f} MB")
        if isinstance(data, dict):
            print(f"  keys: {list(data.keys())}")
            if "best_epoch" in data:
                print(f"  best_epoch: {data.get('best_epoch')}")
            if "epoch" in data:
                print(f"  epoch: {data.get('epoch')}")
            if "experiment_name" in data:
                print(f"  experiment_name: {data.get('experiment_name')}")
    else:
        print(f"[{name}] NOT FOUND")

# Test loading Exp E model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTesting model load on device: {device}")
model = build_densenet121_dgcbam(num_classes=2, pretrained=False, initial_gamma=0.1)
exp_e_data = torch.load(ckpt_exp_e, map_location=device)
state_dict = exp_e_data["model_state_dict"] if "model_state_dict" in exp_e_data else exp_e_data
model.load_state_dict(state_dict)
model.to(device)
model.eval()

counts = get_model_parameter_counts(model)
gammas = model.get_gamma_values()
print(f"Model parameters: {counts}")
print(f"Learned gammas: {gammas}")

# Test forward pass with dummy tensor
dummy = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    out = model(dummy)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    print(f"Dummy output: {out.cpu().numpy()}, Softmax probs: {probs}, Sum: {probs.sum():.6f}")

print("\nAll checks passed successfully!")
