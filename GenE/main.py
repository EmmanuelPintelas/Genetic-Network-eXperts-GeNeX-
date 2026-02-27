############################################################################################################
 
############################################ GenE ##########################################################
 
############################################################################################################
 
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

from utils import*
from CONFIGS import*

# --- clean up ---
folder = GENERAL_CONFIG["STORAGE_PATH"]

for name in os.listdir(folder):
    path = os.path.join(folder, name)

    if os.path.isdir(path):
        shutil.rmtree(path)     # delete directory
    else:
        os.remove(path)         # delete file
# --- clean up ---




# ----- random augmentations for diversity ---->
set0 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    transforms.ToTensor(),
])
set1 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    transforms.ToTensor(),
])
# set 2: heavy-geometric-based
set2 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    transforms.ToTensor(),
])
# set 3: blurring-based
set3 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    RandomGaussianBlur(),
    transforms.ToTensor(),
])
# set 4: sharpness-based
set4 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    RandomSharpen(),
    transforms.ToTensor(),
])
# set 5: contrast-based
set5 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    RandomCLAHE(),
    transforms.ToTensor(),
])
 
set6 = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    RandomRotation([0, 90, 180, 270]),
    RandomSharpen(),
    RandomCLAHE(),
    transforms.ToTensor(),
])
# <----- random augmentations for diversity ----

transform_val = transforms.Compose([
    transforms.Resize(GenE_CONFIG["SIZE"], antialias=True),
    transforms.ToTensor(),
])
val_dataset = ImageFolder(os.path.join(GENERAL_CONFIG["ROOT_PATH"], GENERAL_CONFIG["val_dir"]), transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
### test_dataset = ImageFolder(os.path.join(GENERAL_CONFIG["ROOT_PATH"], GENERAL_CONFIG["test_dir"]), transform=transform_val)
### test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

os.makedirs(os.path.join(GENERAL_CONFIG["STORAGE_PATH"],"M"), exist_ok=True) # Initialize output model pool M 
 



# --- Ranged-Hyperparamters --------->
augm_setups = [set4]
learning_rates_inits = [1e-4, 5e-5, 1e-5]
weight_decays = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0]
ratios = [0.9, 1, 1.1]
batch_sizes = [32, 64, 128]
num_epochs = [1, 3, 5]
# <----------------------------------
 


 
# <----- Gradient-based Initialization (Generation 0) ---------------------------------
M_evol = os.path.join(GENERAL_CONFIG["STORAGE_PATH"],"M_evol")# Initialize evolutionary model pool
os.makedirs(M_evol, exist_ok=True)
 
print("\n")
for it in tqdm(range(GenE_CONFIG["N"]), desc="Gradient-based Initialization (Generation 0)"):
    M = MobileNet('mobilenetv3_small_100', pretrained=True).to(device)
    augm_stp = set4     
    lr_init = learning_rates_inits[random.randint(0,len(learning_rates_inits)-1)]
    wei_decay = weight_decays[random.randint(0,len(weight_decays)-1)]
    rat = ratios[random.randint(0,len(ratios)-1)]
    bs = batch_sizes[random.randint(0,len(batch_sizes)-1)]
    n_ep = num_epochs[random.randint(0,len(num_epochs)-1)]
    train_dataset = ImageFolder(os.path.join(GENERAL_CONFIG["ROOT_PATH"], GENERAL_CONFIG["train_dir"]), transform = augm_stp)
    train_dataset = downsample_dataset(train_dataset, ratio = rat)
    train_loader = DataLoader(train_dataset, batch_size = bs, shuffle=True)
    optimizer = optim.Adam(M.parameters(), lr = lr_init, weight_decay = wei_decay)
    loss_function = nn.CrossEntropyLoss()
 
    for ep in range(n_ep):
        tr_sc = Train(M, train_loader, optimizer, loss_function)
        ###_ts_sc = Validation(M, test_loader)
        ###print (np.round(tr_sc,4),'_',np.round(_ts_sc,4))
 
    torch.save(M.state_dict(), os.path.join(M_evol, f"M_gen0_{it}.pt"))
# <----- Gradient-based Initialization (Generation 0) ---------------------------------
 
# <---------- Evolution Loop ------------------------------------------------------------------------------------ 
for g in range(GenE_CONFIG["G"]):
    # <----- Genetic Child Generation path and Light Fine-tuning ----------------------
    M_gen = os.path.join(GENERAL_CONFIG["STORAGE_PATH"],"M_gen") # Initialize temporary set M_gen
    os.makedirs(M_gen, exist_ok=True)
    for j in tqdm(range(N_g), desc="Genetic Childs Generation "+ str(g)+ ' - Crossover & Mutation'):
        # random sample from M_evol
        model_paths = glob.glob(os.path.join(M_evol, "*.pt"))
        pathA, pathB = random.sample(model_paths, 2)
 
        child_1 = crossover_mutation(pathA, pathB, strategy="uniform")
        child_2 = crossover_mutation(pathA, pathB, strategy="uniform") # for faster we can disable 2nd child
 
        for child in [child_1, child_2]:# Lightly fine-tune (Output head) for 1 epoch
            augm_stp = set4     
            lr_init = 1e-5
            wei_decay = weight_decays[random.randint(0,len(weight_decays)-1)]
            rat = ratios[random.randint(0,len(ratios)-1)]
            bs = 64
            train_dataset = ImageFolder(os.path.join(GENERAL_CONFIG["ROOT_PATH"], GENERAL_CONFIG["train_dir"]), transform = augm_stp)
            train_dataset = downsample_dataset(train_dataset, ratio = rat)
            train_loader = DataLoader(train_dataset, batch_size = bs, shuffle=True)
            optimizer = optim.Adam(child.parameters(), lr = lr_init, weight_decay = wei_decay)
            loss_function = nn.CrossEntropyLoss()
 
            tr_sc = Train (child, train_loader, optimizer, loss_function) 
            val_sc = Validation (child, val_loader)
            TO = tr_sc - val_sc
            ###_ts_sc = Validation(child, test_loader)
            ###print (np.round(tr_sc,4),'_',np.round(_ts_sc,4))
            torch.save(child.state_dict(), os.path.join(M_gen, f"C_{np.round(TO,7)}.pt"))
    # <----- Genetic Child Generation path and Light Fine-tuning ----------------------
 
 
 
    # <---- Select `sel` models from M_gen and move them to output M -------------
    selection_mode = "random"  # Options: "TO" or "random"
    model_files = glob.glob(os.path.join(M_gen, "C_*.pt"))
 
    if selection_mode == "TO":
        models_with_TO = []
        for path in model_files:
            filename = os.path.basename(path)
            try:
                TO = float(filename.split("_")[1].replace(".pt", ""))
                models_with_TO.append((TO, path))
            except:
                continue
        # Sort by minimum TO (smallest overfitting)
        models_with_TO.sort(key=lambda x: x[0])
        sel_models = models_with_TO[:sel]
 
    elif selection_mode == "random":
        selected_paths = random.sample(model_files, sel)
        sel_models = [(0.0, path) for path in selected_paths]  # TO = 0.0 (placeholder)
 
    else:
        raise ValueError("Invalid selection_mode. Use 'TO' or 'random'.")
 
    # Copy to folder M
    for i, (TO, path) in enumerate(sel_models):
        shutil.copy2(path, os.path.join(GENERAL_CONFIG["STORAGE_PATH"], "M", f"M_{g}_{i}_TO_{round(TO, 7)}.pt"))
    print(str(sel)+ " genetic networks stored to output folder M\n")
    # <---- Select `sel` models from M_gen and move them to output M -------------
 
 
    # <---- termination check
    if g == GenE_CONFIG["G"]-1:
        try:
            shutil.rmtree(M_gen)
        except Exception:
            pass
        try:
            shutil.rmtree(M_evol)
        except Exception:
            pass
        try:
            shutil.rmtree(M_rest)
        except Exception:
            pass
        print("\n GenE Terminated!!! -> Produced ", str(GenE_CONFIG["N"]), 'diverse networks')
        break
    # <---- termination check
 
    # <--- Create M_rest = M_gen \ sel_models + Fresh new initialized models ---
    sel_paths = set(p for _, p in sel_models)
    all_paths = set(glob.glob(os.path.join(M_gen, "C_*.pt")))
    rest_paths = list(all_paths - sel_paths)
    M_rest = os.path.join(GENERAL_CONFIG["STORAGE_PATH"],"M_rest")
    os.makedirs(M_rest, exist_ok=True)
    # Copy unselected children to M_rest
    for path in rest_paths:
        shutil.copy2(path, os.path.join(M_rest, os.path.basename(path)))
 
    # Add (N - sel) new fresh random initializations to M_rest
    for idx in range(GenE_CONFIG["N"] - sel):
        M = MobileNet('mobilenetv3_small_100', pretrained=True).to(device)
        torch.save(M.state_dict(), os.path.join(M_rest, f"M_New_{g}_{idx}.pt"))
 
    # Remove the old evolutionary and temporal gen pool
    shutil.rmtree(M_evol)
    shutil.rmtree(M_gen)
    os.makedirs(M_evol, exist_ok=True)
    # <--- Create M_rest = M_gen \ sel_models + Fresh new initialized models ---
 
 
    # <--- Gradient-based Training path on rest + fresh ---------------------------------------
    model_paths = glob.glob(os.path.join(M_rest, "*.pt"))
    for i in tqdm(range(len(model_paths)), desc="Gradient-based Training on rest + fresh"):
        path = model_paths[i]
        M = MobileNet('mobilenetv3_small_100', pretrained=False).to(device)
        M.load_state_dict(torch.load(path))
        for param in M.parameters():
            param.requires_grad = True
        augm_stp = set4     
        lr_init = learning_rates_inits[random.randint(0,len(learning_rates_inits)-1)]
        wei_decay = weight_decays[random.randint(0,len(weight_decays)-1)]
        rat = ratios[random.randint(0,len(ratios)-1)]
        bs = batch_sizes[random.randint(0,len(batch_sizes)-1)]
        n_ep = 1
        train_dataset = ImageFolder(os.path.join(GENERAL_CONFIG["ROOT_PATH"], GENERAL_CONFIG["train_dir"]), transform = augm_stp)
        train_dataset = downsample_dataset(train_dataset, ratio = rat)
        train_loader = DataLoader(train_dataset, batch_size = bs, shuffle=True)
        optimizer = optim.Adam(M.parameters(), lr = lr_init, weight_decay = wei_decay)
        loss_function = nn.CrossEntropyLoss()
 
        for ep in range(n_ep):
            tr_sc = Train(M, train_loader, optimizer, loss_function)
            ###_ts_sc = Validation(M, test_loader)
            ###print (np.round(tr_sc,4),'_',np.round(_ts_sc,4))
 
        torch.save(M.state_dict(), os.path.join(M_evol, f"M_{g}_{i}.pt"))
    shutil.rmtree(M_rest)
    # <--- Gradient-based Training path on rest + fresh ---------------------------------------
 

# <---------- Evolution Loop ------------------------------------------------------------------------------------ 
