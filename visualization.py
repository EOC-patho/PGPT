import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from vit2.CLS2IDX import CLS2IDX


from vit2.vit_explanation_try import CamModel
from vit2.vit_explanation_try import print_top_classes
from vit2.vit_explanation_try import generate_heatmap_any
from vit2.vit_explanation_try import generate_heatmap_224

import torchvision.transforms as transforms
from vit2.vit_new import ViT
device = torch.device("cpu")

import os
import math
import random
from sklearn.metrics import balanced_accuracy_score

import copy
from functools import wraps, partial

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

from einops import repeat, rearrange 
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

model = ViT(
    image_size = 96,
    patch_size = 4, 
    channels = 768,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 768 * 4,
    num_classes = 3
)

from collections import OrderedDict

weights = torch.load("8.pth", map_location=torch.device('cpu'))
from collections import OrderedDict
new_dict = OrderedDict()
for k, v in weights.items():
    if "module" in k:
        new_key = k[7:]
        new_dict[new_key] = v
model.eval()
#model.to(device)
#model = nn.DataParallel(model)
model.load_state_dict(new_dict, strict = False)

cam_model = CamModel(model)

image = torch.load("yourimage.svs.db")  #image.db from store_5

transform = transforms.ToTensor()

image = np.array(image)
image = transform(image)
image = image.unsqueeze(0)
image = F.interpolate(image, size =(96, 96))
image = image.squeeze(0)
image = image.to(torch.float32)

output = model(image.unsqueeze(0))
print_top_classes(output)

fig, axs = plt.subplots(1, 2)

axs[0].imshow(image0);
axs[0].axis("off");

cat, patch_index = generate_heatmap_224(model = cam_model, image = image0,
                                    transformed_image = image, img_size = 384,
                                    index = 2, n=3)
axs[1].imshow(cat);
axs[1].axis("off");

