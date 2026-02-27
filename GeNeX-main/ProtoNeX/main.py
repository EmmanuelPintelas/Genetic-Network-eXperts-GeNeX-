############################################################################################################
 
############################################ ProtoNeX ######################################################
 
############################################################################################################
 
 
 
 
import os
import glob
import random
import shutil
import numpy as np
import timm
import torch.nn as nn
import torch
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchvision.transforms import functional as FF
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageEnhance
from sklearn.metrics import confusion_matrix
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
sys.path.append(os.path.abspath(".."))
from GenE.CONFIGS import *


# --- clean up ---
folder = "ProtoNeX_output"

for name in os.listdir(folder):
    path = os.path.join(folder, name)

    if os.path.isdir(path):
        shutil.rmtree(path)     # delete directory
    else:
        os.remove(path)         # delete file
# --- clean up ---

 
class ProtoNeX:
    def __init__(self, model_pool_path, train_loader, val_loader, num_experts=5):
        self.model_pool_path = model_pool_path
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.num_experts = num_experts
        self.prototype_models = []
 
    def _get_prediction_vectors(self, model):
        """Extract prediction vectors on validation set"""
        model.eval()
        pred_vectors = []
        with torch.no_grad():
            for inputs, _ in self.val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = FF.softmax(outputs, dim=1)
                pred_vectors.append(probs.cpu().numpy())
        return np.concatenate(pred_vectors, axis=0)
 
    def _behavioral_clustering(self, models):
        """Cluster models by prediction behavior using GMM"""
        print("\nClustering process...")
        pred_signatures = []
        for model in tqdm(models, desc="Extracting prediction signatures"):
            pred_matrix = self._get_prediction_vectors(model)[:,1]
            pred_signatures.append(pred_matrix)
        signatures = np.array([p.flatten() for p in pred_signatures])
 
        max_clusters = min(10, len(models)-1)
        gmm_models = []
        for k in range(3, max_clusters+1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(signatures)
            gmm_models.append(gmm)
        best_gmm = gmm_models[np.argmin([m.bic(signatures) for m in gmm_models])]
        # Assign models to clusters
        clusters = best_gmm.predict(signatures)
        return clusters, best_gmm.means_, pred_signatures
 
 
    def _select_experts(self, models, cluster_labels, cluster_centers, pred_matrices):
        print("\nExperts Election process...")
        expert_indices = []
        ###pred_matrices = [self._get_prediction_vectors(m) for m in models]
 
        for k in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            cluster_preds = [pred_matrices[i] for i in cluster_indices]
 
            ###pred_arrays = [p.cpu().numpy() for p in cluster_preds]
            centroid = cluster_centers[k]
 
            # 1. Top Validation Performer
            val_scores = [self.scoring(pred_matrices[i], self.val_loader) for i in cluster_indices]#[self.Validation(models[i], self.val_loader) for i in cluster_indices]
            top_val_idx = cluster_indices[np.argmax(val_scores)]
 
            # 2. Most Robust (via noise perturbation)
            robustness_scores = []
            for idx in cluster_indices:
                orig_acc = self.scoring(pred_matrices[idx], self.val_loader)#self.Validation(models[idx], self.val_loader)
                noisy_acc = self._evaluate_robustness(models[idx])
                robustness_scores.append(orig_acc - noisy_acc)
            robust_idx = cluster_indices[np.argmin(robustness_scores)]
 
            # 3. Cluster Representative (nearest to centroid)
            centroid_dists = [pairwise_distances([p], [centroid], metric='cosine')[0,0] for p in cluster_preds]
            centroid_idx = cluster_indices[np.argmin(centroid_dists)]
 
            # 4. Intra-Anomalous (furthest from centroid)
            anomalous_idx = cluster_indices[np.argmax(centroid_dists)]
 
            # 5. Inter-Anomalous (furthest from other clusters)
            other_centers = [c for i,c in enumerate(cluster_centers) if i != k]
            aver_inter_dist = [np.mean([pairwise_distances([p], [oc])[0,0] for oc in other_centers]) for p in cluster_preds]
            inter_anomalous_idx = cluster_indices[np.argmax(aver_inter_dist)]
 
            # Store unique expert indices
            selected = list({top_val_idx, robust_idx, centroid_idx, 
                            anomalous_idx, inter_anomalous_idx})
            expert_indices.append(selected[:self.num_experts])
 
        return expert_indices
 
    def _evaluate_robustness(self, model, noise_std=0.1):
        """Evaluate model under input perturbations"""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                noise = torch.randn_like(inputs) * noise_std
                noisy_inputs = torch.clamp(inputs + noise, 0, 1)
                outputs = model(noisy_inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total
 
    def _fuse_prototypes(self, expert_models):
        """Fuse experts via weight averaging"""
        prototype_state = OrderedDict()
        expert_states = [m.state_dict() for m in expert_models]
 
        for key in expert_states[0].keys():
            if 'num_batches_tracked' in key:
                continue
 
            # Average all expert weights
            weights = torch.stack([s[key] for s in expert_states])
            prototype_state[key] = weights.mean(dim=0)
 
        return prototype_state
 
    def _optimize_ensemble_weights(self, prototypes, val_loader):
        """Optimize ensemble weights via SQP"""
        # Get predictions on validation set
        all_preds = []
        for model in prototypes:
            model.eval()
            preds = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds.append(FF.softmax(outputs, dim=1).cpu().numpy())
            all_preds.append(np.concatenate(preds, axis=0))
 
        # Define objective function
        def objective(w):
            w = np.clip(w, 0, 1)
            w = w / w.sum()
            ensemble_pred = sum(w[i]*all_preds[i] for i in range(len(w)))
            ensemble_acc = Gm_Score(
                np.argmax(ensemble_pred, axis=1),
                val_loader.dataset.targets)
            return -ensemble_acc  # Minimize negative accuracy
 
        # SQP optimization
        n_prototypes = len(prototypes)
        w0 = np.ones(n_prototypes) / n_prototypes
        bounds = [(0,1) for _ in range(n_prototypes)]
        cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        res = minimize(objective, w0, method='SLSQP', 
                      bounds=bounds, constraints=cons)
 
        return res.x
 
    def Validation (self, M, loader):
            M.eval()
            PREDS, PROBS, LABELS = [], [], []
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = M(inputs)
                    _, preds = torch.max(outputs, 1)
                    PREDS.append(preds.detach().cpu().numpy())
                    PROBS.append(outputs[:, 1].detach().cpu().numpy())
                    LABELS.append(targets.detach().cpu().numpy())
            PREDS, PROBS, LABELS = np.concatenate(PREDS), np.concatenate(PROBS), np.concatenate(LABELS)
            score = Gm_Score(PREDS, LABELS)
            return score
    
    def scoring(self, outputs, loader):
        PREDS = (outputs + 0.5).astype(int)
        PROBS = outputs
        LABELS = np.array(loader.dataset.targets)
        score = Gm_Score(PREDS, LABELS)
        return score

 
    def Train (self, M, loader, epochs=1):
        M.train()
 
        optimizer = optim.Adam(M.parameters(), lr = 5e-5, weight_decay = 1e-4)
        loss_function = nn.CrossEntropyLoss()
 
        PREDS, PROBS, LABELS = [], [], []
        for ep in range(epochs):
            for inputs, targets in loader:
                ##visualize(inputs, num_samples=9)
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = M(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
 
                PREDS.append(preds.detach().cpu().numpy())
                PROBS.append(outputs[:, 1].detach().cpu().numpy())
                LABELS.append(targets.detach().cpu().numpy())
            PREDS, PROBS, LABELS = np.concatenate(PREDS), np.concatenate(PROBS), np.concatenate(LABELS)
            score = Gm_Score(PREDS, LABELS)
            return score
 
 
    def run(self):
        """Execute full ProtoNeX pipeline"""
        # Load all models from pool
        model_files = glob.glob(os.path.join(self.model_pool_path, "*.pt"))
        models = []
        for mfile in tqdm(model_files, desc="Loading models"):
            model = MobileNet('mobilenetv3_small_100', pretrained=False).to(device)
            model.load_state_dict(torch.load(mfile))
            models.append(model)
 
        # 1. Behavioral Clustering
        clusters, centers, pred_signatures = self._behavioral_clustering(models)
 
        # 2. Experts Election
        cluster_expert_indices = self._select_experts(models, clusters, centers, pred_signatures)
 
        # 3. Prototype Synthesis (per cluster)
        print("\nPrototypes fusion...")
        for cei in cluster_expert_indices:
            cluster_experts = [models[i] for i in cei]
 
            proto_state = self._fuse_prototypes(cluster_experts)
 
 
            prototype = MobileNet('mobilenetv3_small_100', pretrained=False).to(device)
            prototype.load_state_dict(proto_state)
 
 
            for param in prototype.parameters():
                param.requires_grad = False
            for param in prototype.head.parameters():
                param.requires_grad = True
 
 
            self.Train(prototype, self.train_loader)
            _val_acc = self.Validation(prototype, self.val_loader)
 
            self.prototype_models.append(prototype)
 
 
 
        optimal_weights = self._optimize_ensemble_weights(
            self.prototype_models, self.val_loader)
 
        return self.prototype_models, optimal_weights
 
class RandomRotation:
    def __init__(self, angles):
        self.angles = angles
 
    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)
class Sharpen:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Example sharpening
        #img = TF.to_tensor(img).to(device)
        return img
 
class RandomSharpen:
    def __init__(self, p=0.5, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.p = p
        self.sharp = Sharpen()
 
    def __call__(self, img):
        if random.random() < self.p:
            img = self.sharp(img)
        return img
 
 
# ----------------------------------------------------------------------------------------------------------------------------------------------
 
# ------> LINK for Dataset:
#   https://drive.google.com/file/d/1RcZsiCOuFFWXuEoiVptNvP7lDTGWlhvh/view?usp=sharing
 
# ------> LINK for Storage_Folder models from GenE for this dataset:
#   https://drive.google.com/file/d/1UTcYYejBnhaeyNa0MEX5Ok38I0SQtB43/view?usp=sharing
 
SIZE = GenE_CONFIG["SIZE"]
root =  GENERAL_CONFIG["ROOT_PATH"]
storage = GENERAL_CONFIG["STORAGE_PATH"]
train_dir = GENERAL_CONFIG["train_dir"]
val_dir = GENERAL_CONFIG["val_dir"]
test_dir = GENERAL_CONFIG["test_dir"]
 
# ----------------------------------------------------------------------------------------------------------------------------------------------
 
transform_train = transforms.Compose([
    transforms.Resize(SIZE, antialias=True),
    RandomRotation([0, 90, 180, 270]),
    RandomSharpen(),
    transforms.ToTensor(),
])
transform_val = transforms.Compose([
        transforms.Resize(SIZE, antialias=True),
        transforms.ToTensor(),
    ])
 
train_dataset = ImageFolder(os.path.join(root, train_dir), transform = transform_train)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
val_dataset = ImageFolder(os.path.join(root, val_dir), transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataset = ImageFolder(os.path.join(root, test_dir), transform=transform_val)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
 
 
 
 
 
# Run ProtoNeX
protonex = ProtoNeX(
        model_pool_path=os.path.join(storage, "M"),
        train_loader=train_loader,
        val_loader=val_loader,
        num_experts=5)
 
prototypes, weights = protonex.run()

# Create folder
save_dir = "ProtoNeX_output"
os.makedirs(save_dir, exist_ok=True)

# Save each prototype
for i, model in enumerate(prototypes):
    torch.save(model.state_dict(), f"{save_dir}/prototype_{i}.pt")

# Save weights
np.save(f"{save_dir}/ensemble_weights.npy", weights)

print("Models and weights saved.")
