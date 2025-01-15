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

import torchvision.transforms as transforms

from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/data2/patho-vit_5_23_oc"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import os

def Extractor(path, train_lib, transform, batch_size, model, device):
    model.eval()

    for i, image in enumerate(train_lib["images"]):
        cols = []
        rows=[]

        for patch in train_lib["images"][i]:
            col, row = patch.split("/")[-1].split(".")[0].split("_")
            cols.append(int(col))
            rows.append(int(row))
        col_last = sorted(np.array(cols))[-1]
        row_last = sorted(np.array(rows))[-1]
        image = torch.zeros((row_last+1), (col_last+1), 768)
    
        patches_batch = [] 
        patches_pos = []  
        img_features =[]
        for patch in train_lib["images"][i]:
            col, row = patch.split("/")[-1].split(".")[0].split("_")
            patches_pos.append([int(col), int(row)])
    
        for j, patch in enumerate(train_lib["images"][i]):
            torch.cuda.empty_cache() 
            patch = Image.open(patch)

            patch = transform(patch).unsqueeze(0) 
            patches_batch.append(patch)
        
            if ((j+1) % batch_size == 0) or ((j+1) == len(train_lib["images"][i])):
                torch.cuda.empty_cache()  
                patches_batch = torch.cat(patches_batch, 0).to(device)
                _, features = model(patches_batch)
                features = features.cpu().tolist()
                for k in range(len(features)):
                    img_features.append(features[k])
                patches_batch = []  
    
        for index in range(len(patches_pos)):
            batch_range = torch.arange(patches_pos[index][1], patches_pos[index][1] + 1)[:, None]
            indices = torch.arange(patches_pos[index][0], patches_pos[index][0] + 1)
            image[batch_range, indices] = torch.Tensor(img_features[index]).reshape(1, 1, 768)
    
        torch.save(image, "store_5_96/{}.db".format(train_lib["pathid"][i]))

    images = []
    for i, pathid in enumerate(train_lib["pathid"]):
        images.append(os.path.join(path, "store_5_96", pathid+".db"))
    c = {"pathid": train_lib["pathid"], "images": images, "labels": train_lib["labels"]}
    df = pd.DataFrame(c)
    df.to_csv("files_label/store_5_96_exva.csv", index=False, encoding="utf_8_sig")
    torch.save(c, "files_label/store_5_96_exva.db")
    return "store_5_96_exva.db"

import ViT as ViT
from collections import OrderedDict

vit = ViT(
    image_size = 384,
    patch_size = 16,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 768 * 4,
    num_classes = 2
)

weights = torch.load("/data2/patho-vit_5_23_oc/tile_level.pt")

vit.to(device)
vit = nn.DataParallel(vit)
vit.load_state_dict(weights, strict = True)

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])


train_lib = torch.load("/data2/patho-vit_5_23_oc/files_label/store_4_train.db")
valid_lib = torch.load("/data2/patho-vit_5_23_oc/files_label/store_4_valid.db")
exva_lib = torch.load("/data2/patho-vit_5_23_oc/files_label/store_4_exvalid.db")

extractor = Extractor(path = path, train_lib = train_lib, transform = transform, batch_size = 200, model = vit, device = device)
extractor2 = Extractor(path = path, train_lib = valid_lib, transform = transform, batch_size = 200, model = vit, device = device)
extractor3 = Extractor(path = path, train_lib = exva_lib, transform = transform, batch_size = 200, model = vit, device = device)





