import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import torch
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torchvision.transforms import functional as F
from copy import deepcopy
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import ImageEnhance, ImageOps
import cv2
import glob
import random
import os
import warnings
import torch.nn.functional as FF
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import shutil
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = torch.device("cuda")





def Gm_Score(predictions, labels_query):
    # Confusion matrix
    cm = confusion_matrix(labels_query, predictions)
    # --- multiclass -----
    # # GM: Geometric Mean metric, generalize for multiclass
    # # We extract the diagonal of the confusion matrix and divide by the sum of each row
    # recalls_per_class = np.diag(cm) / np.sum(cm, axis=1)
    # # Calculate the geometric mean of the recall values
    # gm = np.prod(recalls_per_class) ** (1.0 / len(recalls_per_class))
 
    # --- binary -----
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    gm = np.sqrt(sensitivity * specificity)
 
    return round(gm, 5)
 
 
 
 
def downsample_dataset(dataset, target_class_0=0, target_class_1=1, ratio=1):
    """
    Randomly down-samples the dominant class to match the number of samples in the minority class.
 
    Returns:
        Subset: A subset of the dataset.
    """
    def majority_down_sample (class_minority_indices, class_majority_indices):
        minority_size = len(class_minority_indices)
        mazority_size = len(class_majority_indices)
        majority_size_sample = int(min([minority_size*ratio, mazority_size]))
        downsampled_class_majority_indices = random.sample(class_majority_indices, majority_size_sample)
        total_indices = downsampled_class_majority_indices + class_minority_indices
        return total_indices
 
    # Find indices of each class
    class_0_indices = [i for i, (img, label) in enumerate(dataset) if label == target_class_0]
    class_1_indices = [i for i, (img, label) in enumerate(dataset) if label == target_class_1]
 
    if len(class_0_indices) > len(class_1_indices):
        indices = majority_down_sample (class_1_indices, class_0_indices)
    else:
        indices = majority_down_sample (class_0_indices, class_1_indices)
 
    # Create a subset of the dataset with the down-sampled indices
    dataset = Subset(dataset, indices)
 
    return dataset
 
 
 
 
 
 
 
 
class RandomRotation:
    def __init__(self, angles):
        self.angles = angles
 
    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)
class RandomGaussianBlur:
    def __init__(self, p=0.5, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.p = p
        self.blur = transforms.GaussianBlur(kernel_size, sigma)
 
    def __call__(self, img):
        if random.random() < self.p:
            img = self.blur(img)
        return img
class RandomSharpen:
    def __init__(self, p=0.5, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.p = p
        self.sharp = Sharpen()
 
    def __call__(self, img):
        if random.random() < self.p:
            img = self.sharp(img)
        return img
class RandomCLAHE:
    def __init__(self, p=0.5, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.p = p
        self.clahe = CLAHE()
 
    def __call__(self, img):
        if random.random() < self.p:
            img = self.clahe(img)
        return img
class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
 
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        img = np.array(img)
 
        # Convert image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
 
        # Split LAB image to different channels
        l, a, b = cv2.split(lab)
 
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
 
        # Merge the CLAHE enhanced L-channel with A and B channels
        limg = cv2.merge((cl, a, b))
 
        # Convert image back to RGB color space
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
 
        # final = TF.to_tensor(final).to(device)
        final = Image.fromarray(final)
        return final
class Sharpen:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Example sharpening
        #img = TF.to_tensor(img).to(device)
        return img
 
 
def flatten_conv_filters(weight_tensor):
    return weight_tensor.view(weight_tensor.shape[0], -1)
 
def cosine_similarity_matrix(base, peer):
    base_norm = FF.normalize(base, dim=1)
    peer_norm = FF.normalize(peer, dim=1)
    return torch.matmul(base_norm, peer_norm.T) 
 
def align_and_weight_filters(base_filters, peer_filters):
    base_flat = flatten_conv_filters(base_filters)
    peer_flat = flatten_conv_filters(peer_filters)
    sim_matrix = cosine_similarity_matrix(base_flat, peer_flat).cpu().numpy()
 
    # Hungarian matching: optimal filter pairing
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # maximize sim → minimize negative sim
 
    aligned = peer_filters[col_ind]
    similarities = sim_matrix[row_ind, col_ind]
    weights = torch.tensor(similarities, dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(3)
 
    # Normalize weights
    weights = weights / weights.sum()
 
    return aligned, weights
 
def average_aligned_weights_weighted(state_dicts):
    avg_state_dict = OrderedDict()
    base = state_dicts[0]
 
    for key in base.keys():
        if 'num_batches_tracked' in key:
            continue
 
        # Conv weight layer
        if 'conv' in key and 'weight' in key and len(base[key].shape) == 4:
            base_tensor = base[key]
            out_channels = base_tensor.shape[0]
 
            weighted_sum = base_tensor.clone()
            total_weight = torch.ones(out_channels, 1, 1, 1)
 
            for peer_sd in state_dicts[1:]:
                aligned, weights = align_and_weight_filters(base_tensor, peer_sd[key])
                weighted_sum += aligned * weights
                total_weight += weights
 
            avg_state_dict[key] = weighted_sum / total_weight
 
        else:
            # Default average for everything else (e.g., BN, Linear)
            tensors = [sd[key] for sd in state_dicts]
            avg_state_dict[key] = torch.mean(torch.stack(tensors, dim=0), dim=0)
 
    return avg_state_dict
 
 
class MobileNet(nn.Module):
    def __init__(self, model_name=None, num_classes=0, pretrained = True):
        super(MobileNet, self).__init__()
 
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.head = nn.Linear(self.model.num_features, 2)
 
    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
 
 
def Train (M, loader, optimizer, loss_function):
        M.train()
        PREDS, PROBS, LABELS = [], [], []
 
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
 
 
 
def uniform_crossover(state_dict_A, state_dict_B):
    child_dict = OrderedDict()
    for key in state_dict_A.keys():
        if 'num_batches_tracked' in key:
            continue
        tensor_A = state_dict_A[key]
        tensor_B = state_dict_B[key]
        mask = torch.rand_like(tensor_A) > 0.5
        child_tensor = torch.where(mask, tensor_A, tensor_B)
        child_dict[key] = child_tensor
    return child_dict
 
def average_crossover(state_dict_A, state_dict_B):
    child_dict = OrderedDict()
    for key in state_dict_A.keys():
        if 'num_batches_tracked' in key:
            continue
        child_dict[key] = 0.5 * state_dict_A[key] + 0.5 * state_dict_B[key]
    return child_dict
 
def mutate(child_dict, mutation_rate=0.05, sigma=0.01):
    for key in child_dict.keys():
        if 'weight' in key and len(child_dict[key].shape) >= 2:  # Skip batch norm stats etc.
            if torch.rand(1).item() < mutation_rate:
                noise = torch.randn_like(child_dict[key]) * sigma
                child_dict[key] += noise
    return child_dict
 
 
def crossover_mutation(pathA, pathB, strategy="aligned", mutate_flag=True):
    state_dict_A = torch.load(pathA, map_location='cpu')
    state_dict_B = torch.load(pathB, map_location='cpu')
 
    if strategy == "aligned":
        child_dict = average_aligned_weights_weighted([state_dict_A, state_dict_B])
    elif strategy == "uniform":
        child_dict = uniform_crossover(state_dict_A, state_dict_B)
    elif strategy == "avg":
        child_dict = average_crossover(state_dict_A, state_dict_B)
    else:
        raise ValueError("Unknown crossover strategy")
 
    if mutate_flag:
        child_dict = mutate(child_dict)
 
    for param in child_dict:
        child_dict[param].requires_grad = False
 
    child = MobileNet('mobilenetv3_small_100', pretrained=True).to(device)
    child.load_state_dict(child_dict, strict=False)
    return child
 
 
def Validation (M, loader):
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
