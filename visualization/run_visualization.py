"""
Generate visualization images for thesis:
1. Grad-CAM visualization (clean vs adversarial samples)
2. Adversarial examples comparison
"""

import os

# Solve OpenMP library conflict problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------- Paths ----------------
# ROOT points to project_root (parent directory of script)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(ROOT, "models", "resnet18_chestxray.pth")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Dataset ----------------
class NIHBinaryDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["Image Index"])).convert("RGB")
        y = int(row["binary_label"])
        if self.transform:
            img = self.transform(img)
        return img, y, row["Image Index"]


# ---------------- Model ----------------
class NormalizeWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_model():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    model = NormalizeWrapper(backbone).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------- FGSM Attack ----------------
def fgsm_attack(model, x, y, eps):
    if eps == 0.0:
        return x
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + eps * torch.sign(grad)
    return torch.clamp(x_adv, 0.0, 1.0).detach()


# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device).requires_grad_(True)

        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(x.size(2), x.size(3)),
                                               mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return tensor.detach().permute(1, 2, 0).cpu().numpy()


# ============================================
# Visualization 1: Grad-CAM Comparison (Clean vs Adversarial)
# ============================================
def visualize_gradcam_comparison(model, dataset, n_samples=4, eps=2/255):
    """Generate Grad-CAM comparison: clean images vs adversarial images"""
    print("\n=== Generating Grad-CAM Visualization ===")
    
    gradcam = GradCAM(model, model.backbone.layer4)
    
    # Randomly select samples
    random.seed(42)
    indices = random.sample(range(len(dataset)), n_samples)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.2 * n_samples))
    
    for i, idx in enumerate(indices):
        x, y, img_name = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y_tensor = torch.tensor([y]).to(device)
        
        # Generate adversarial sample
        with torch.enable_grad():
            x_adv = fgsm_attack(model, x, y_tensor, eps)
        
        # Prediction
        with torch.no_grad():
            pred_clean = model(x).argmax(1).item()
            pred_adv = model(x_adv).argmax(1).item()
        
        # Grad-CAM
        cam_clean = gradcam(x, pred_clean)
        cam_adv = gradcam(x_adv, pred_adv)
        
        # Plot
        img_clean = tensor_to_numpy(x)
        img_adv = tensor_to_numpy(x_adv)
        
        axes[i, 0].imshow(np.clip(img_clean, 0, 1))
        axes[i, 0].set_title(f"Clean (True: {y}, Pred: {pred_clean})", fontsize=10, pad=4)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam_clean, cmap='jet')
        axes[i, 1].set_title("Grad-CAM (Clean)", fontsize=10, pad=4)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(np.clip(img_adv, 0, 1))
        color = 'red' if pred_adv != y else 'green'
        axes[i, 2].set_title(f"Adversarial (True: {y}, Pred: {pred_adv})", color=color, fontsize=10, pad=4)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(cam_adv, cmap='jet')
        axes[i, 3].set_title("Grad-CAM (Adv)", fontsize=10, pad=4)
        axes[i, 3].axis('off')
    
    fig.suptitle(f"Grad-CAM Comparison: Clean vs Adversarial (ε={eps:.4f})", 
                 fontsize=14, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(RESULTS_DIR, "gradcam_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================
# Visualization 2: Adversarial Examples Comparison
# ============================================
def visualize_adversarial_examples(model, dataset, n_samples=8, eps_list=None):
    """Generate adversarial examples comparison"""
    print("\n=== Generating Adversarial Examples Visualization ===")
    
    if eps_list is None:
        eps_list = [0, 1/255, 2/255, 4/255]
    
    random.seed(42)
    indices = random.sample(range(len(dataset)), n_samples)
    
    fig, axes = plt.subplots(n_samples, len(eps_list) + 1, 
                              figsize=(2.8 * (len(eps_list) + 1), 2.8 * n_samples))
    
    for i, idx in enumerate(indices):
        x, y, img_name = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y_tensor = torch.tensor([y]).to(device)
        
        for j, eps in enumerate(eps_list):
            with torch.enable_grad():
                x_adv = fgsm_attack(model, x, y_tensor, eps)
            
            with torch.no_grad():
                pred = model(x_adv).argmax(1).item()
            
            img = tensor_to_numpy(x_adv)
            axes[i, j].imshow(np.clip(img, 0, 1))
            
            if eps == 0:
                title = f"Clean (Pred: {pred})"
            else:
                title = f"ε={eps:.4f} (Pred: {pred})"
            
            color = 'green' if pred == y else 'red'
            axes[i, j].set_title(title, color=color, fontsize=9, pad=4)
            axes[i, j].axis('off')
        
        # Last column shows perturbation
        perturbation = tensor_to_numpy(x_adv) - tensor_to_numpy(x)
        pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        axes[i, -1].imshow(pert_vis)
        axes[i, -1].set_title("Perturbation (amplified)", fontsize=9, pad=4)
        axes[i, -1].axis('off')
    
    fig.suptitle("Adversarial Examples with Increasing Perturbation", fontsize=14, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(RESULTS_DIR, "adversarial_examples.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================
# Visualization 3: Successful Attack Examples
# ============================================
def visualize_successful_attacks(model, dataset, n_samples=6, eps=2/255):
    """Display successful attack examples"""
    print("\n=== Generating Successful Attack Examples Visualization ===")
    
    successful = []
    
    for idx in range(min(500, len(dataset))):
        x, y, img_name = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y_tensor = torch.tensor([y]).to(device)
        
        with torch.no_grad():
            pred_clean = model(x).argmax(1).item()
        
        if pred_clean != y:
            continue
        
        with torch.enable_grad():
            x_adv = fgsm_attack(model, x, y_tensor, eps)
        
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1).item()
        
        if pred_adv != y:
            successful.append({
                'x_clean': x,
                'x_adv': x_adv,
                'y': y,
                'pred_clean': pred_clean,
                'pred_adv': pred_adv,
                'img_name': img_name
            })
        
        if len(successful) >= n_samples:
            break
    
    if len(successful) == 0:
        print("No successful attacks found!")
        return
    
    fig, axes = plt.subplots(len(successful), 3, figsize=(12, 3.5 * len(successful)))
    
    if len(successful) == 1:
        axes = axes.reshape(1, -1)
    
    for i, ex in enumerate(successful):
        img_clean = tensor_to_numpy(ex['x_clean'])
        img_adv = tensor_to_numpy(ex['x_adv'])
        perturbation = img_adv - img_clean
        
        axes[i, 0].imshow(np.clip(img_clean, 0, 1))
        axes[i, 0].set_title(f"Clean Image\nTrue: {ex['y']}, Pred: {ex['pred_clean']}", 
                            color='green', fontsize=10, pad=5)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(img_adv, 0, 1))
        axes[i, 1].set_title(f"Adversarial Image\nTrue: {ex['y']}, Pred: {ex['pred_adv']}", 
                            color='red', fontsize=10, pad=5)
        axes[i, 1].axis('off')
        
        pert_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        axes[i, 2].imshow(pert_vis)
        axes[i, 2].set_title(f"Perturbation (amplified)\nL∞: {np.abs(perturbation).max():.4f}", 
                            fontsize=10, pad=5)
        axes[i, 2].axis('off')
    
    fig.suptitle(f"Successful Attack Examples (ε={eps:.4f})", fontsize=14, fontweight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(RESULTS_DIR, "successful_attacks.png")
    fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ============================================
# Main Function
# ============================================
def main():
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    
    # Load model
    model = build_model()
    
    # Load dataset
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    dataset = NIHBinaryDataset(TEST_CSV, IMAGE_DIR, tf)
    print(f"Dataset size: {len(dataset)}")
    
    # Generate visualizations
    visualize_gradcam_comparison(model, dataset, n_samples=4, eps=2/255)
    visualize_adversarial_examples(model, dataset, n_samples=6)
    visualize_successful_attacks(model, dataset, n_samples=6, eps=2/255)
    
    print("\n=== All visualizations complete! ===")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
