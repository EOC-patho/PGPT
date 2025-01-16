import os
from sklearn.metrics import balanced_accuracy_score
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

import torchvision.transforms as transforms

from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "yourpath"

from patho_vit.vit_luci2 import ViT as ViT
from patho_vit.airport2 import train_features_extractor_gpvit2 as Extractor
from patho_vit.airport2 import valid_features_extractor_gpvit2 as Extractor2
from collections import OrderedDict

vit = ViT(
    image_size = 384,
    patch_size = 16,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 768 * 4,
    num_classes = 3
)

weights = torch.load("up13.pt", map_location=torch.device('cpu'))
new_dict = OrderedDict()
for k, v in weights.items():
    if "module.target_encoder" in k:
        new_key = k[22:]
        new_dict[new_key] = v

vit.to(device)
vit.load_state_dict(new_dict, strict = False)
vit = nn.DataParallel(vit)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])

train_lib = torch.load("train.db")
valid_lib = torch.load("valid.db")

extractor = Extractor(path = path, train_lib = train_lib, transform = transform, batch_size = 200, model = vit, device = device)
extractor2 = Extractor(path = path, train_lib = valid_lib, transform = transform, batch_size = 200, model = vit, device = device)





