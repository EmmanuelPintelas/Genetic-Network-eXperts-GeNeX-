
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



def load_protonex_models(path="ProtoNeX_output"):
    weights = np.load(f"{path}/ensemble_weights.npy")

    prototypes = []
    i = 0
    while True:
        model_path = f"{path}/prototype_{i}.pt"
        try:
            model = MobileNet('mobilenetv3_small_100', pretrained=False).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            prototypes.append(model)
            i += 1
        except FileNotFoundError:
            break

    return prototypes, weights


def inference_ensemble(prototypes, weights, dataloader):
    weights = torch.tensor(weights).to(device)

    all_preds = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Inferencing"):
            x = x.to(device)
            ensemble_probs = 0

            for model, w in zip(prototypes, weights):
                probs = FF.softmax(model(x), dim=1)
                ensemble_probs += w * probs

            preds = ensemble_probs.argmax(dim=1).cpu()
            all_preds.append(preds)

    return torch.cat(all_preds)


SIZE = GenE_CONFIG["SIZE"]
root =  GENERAL_CONFIG["ROOT_PATH"]
storage = GENERAL_CONFIG["STORAGE_PATH"]
test_dir = GENERAL_CONFIG["test_dir"]

transform_val = transforms.Compose([
        transforms.Resize(SIZE, antialias=True),
        transforms.ToTensor(),
    ])
test_dataset = ImageFolder(os.path.join(root, test_dir), transform=transform_val)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)



prototypes, weights = load_protonex_models("ProtoNeX_output")
preds = inference_ensemble(prototypes, weights, test_loader)
gm = Gm_Score(preds.numpy(), test_dataset.targets)
print("GM:", gm)
