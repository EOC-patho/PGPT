import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import torchvision  
import torchvision.transforms as transforms
import torchvision.models as models 
import timm

import sys
import os
import shutil
import glob
import pyvips
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import random
import cv2
from PIL.Image import Image
import matplotlib.pyplot as plt

path = "yourpath"

if not os.path.exists(os.path.join(path, "store_1")):
        os.makedirs(os.path.join(path, "store_1"))
if not os.path.exists(os.path.join(path, "store_2")):
        os.makedirs(os.path.join(path, "store_2"))
if not os.path.exists(os.path.join(path, "store_3")):
        os.makedirs(os.path.join(path, "store_3"))
if not os.path.exists(os.path.join(path, "store_4")):
        os.makedirs(os.path.join(path, "store_4"))

if not os.path.exists(os.path.join(path, "broken")):
        os.makedirs(os.path.join(path, "broken"))

if not os.path.exists(os.path.join(path, "files")):
        os.makedirs(os.path.join(path, "files"))

if not os.path.exists("output"):
        os.makedirs("output")

pathid = []
pathids = []
pathid_total = [f for f in os.listdir(os.path.join(path + "/store_1")) if not f.startswith(".")]
pathid_tif = [f for f in pathid_total if f.endswith("tif")]
pathid_svs = [f for f in pathid_total if f.endswith("svs")]
pathid.extend(pathid_tif)
pathid.extend(pathid_svs)

pathids_tif = glob.glob(os.path.join(path, "store_1", "*.tif"))
pathids_svs = glob.glob(os.path.join(path, "store_1", "*.svs"))
pathids.extend(pathids_tif)
pathids.extend(pathids_svs)

c = {"pathid": pathid, "pathids": pathids}
df = pd.DataFrame(c)
df.to_csv("files/store_1.csv", encoding="utf_8_sig") 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import torchvision  
import torchvision.transforms as transforms
import torchvision.models as models 
import timm
import sys
import os
import shutil
import glob
import pyvips
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import random
import cv2
from PIL.Image import Image
import matplotlib.pyplot as plt

device = torch.device("cpu")

df = pd.read_csv("files/store_1.csv")
pathid = df["pathid"]
pathids = df["pathids"]
broken = []
for i, image in enumerate(pathids):
    try:
        img = pyvips.Image.new_from_file(image, access="sequential")
    except:
        print("{} is not allowed".format(image))
        broken.append(image)
        continue
    try:
        img.dzsave(os.path.join(path,"store_2", str(pathid[i])), depth="one", tile_size=1024, overlap=0)
    except:
        print("{} cannot be save".format(image))
        broken.append(image)
        continue

c = {"broken": broken}
df = pd.DataFrame(c)
df.to_csv("broken/broken.txt", sep="\t")

dzi_file = glob.glob(os.path.join(path, "store_2", "*.dzi"))
for dzi in dzi_file:    
    os.remove(dzi)

pathid = [f for f in os.listdir(os.path.join(path, "store_2")) if not f.startswith(".")]
empty = []
white = []
irregular = []
examined_ids = []
for i, id in enumerate(pathid):
    print("{} has now been examined".format(id))
    patches = glob.glob(os.path.join(path, "store_2", str(id), "0", "*.jpeg"))
    if len(patches) == 0:
        empty.append(id)
        shutil.rmtree(os.path.join(path, "store_2", str(id))) 
    else:
        for j, patch in enumerate(patches):
            image = cv2.imread(patch)
            if image.var() < 500:
                white.append(patch)
                os.remove(patch)
            elif image.shape[0] < 1024:
                irregular.append(patch)
                os.remove(patch)
        
            if (j + 1) % 10000 == 0:
                print("{} patches have been examined".format(j+1))
                
    examined_ids.append(id)
    if (i + 1) % 20 == 0:
        print("{} patients have been examined".format(i+1))

c = {"examined_ids": examined_ids}
df = pd.DataFrame(c)
df.to_csv("broken/examined_ids.txt", sep="\t")
        
c = {"empty": empty}
df = pd.DataFrame(c)
df.to_csv("broken/empty.txt", sep = "\t")
                
c = {"white": white}
df = pd.DataFrame(c)
df.to_csv("broken/white.txt", sep = "\t")

c = {"irregular": irregular}
df = pd.DataFrame(c)
df.to_csv("broken/irregular.txt", sep = "\t")

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchstain
import cv2

target = cv2.cvtColor(cv2.imread("target.jpeg"), cv2.COLOR_BGR2RGB)
T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])
torch_normalizer = torchstain.MacenkoNormalizer(backend = "torch")
torch_normalizer.fit(T(target))
pathid = [f for f in os.listdir("store_2") if not f.startswith(".")]
for i, id in enumerate(pathid):
    print("{} slide has now been stained".format(id))
    path_new = os.path.join(path, "store_2", str(id), "0")
    pathes = [i for i in os.listdir(path_new) if not i.startswith(".")]
    for m, patch in enumerate(pathes):
        img = cv2.imread(os.path.join(path, "store_2", str(id), "0", "*.jpeg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = T(img)
        try:
            img, _, _ = torch_normalizer.normalize(I=img, stains = True)
            if not os.path.exists(os.path.join(path, "store_3", str(id), "0")):
                os.makedirs(os.path.join(path, "store_3", str(id), "0"))
            plt.imsave(os.path.join(path, "store_3", str(id), "0", "*.jpeg"), img.type(dtype = torch.uint8).numpy())
        except:
            print("{} patch cannot be stained".format(patch))



label_csv = None
def exists(val):
    return val is not None

pathid_orig = [f[:-6] for f in os.listdir(os.path.join(path, "store_3")) if not f.startswith(".")]
    
if exists(label_csv):
    df_label = pd.read_excel(label_csv, index_col = "pathid")
pathid = []
images = []
labels = []
for i, id in enumerate(pathid_orig):
    if len(glob.glob(os.path.join(path, "store_3", str(id) + "_files","0", "*.jpeg"))) != 0:
        pathid.append(id)
        images.append(glob.glob(os.path.join(path, "store_3", str(id) + "_files","0", "*.jpeg")))
        if exists(label_csv):
            labels.append(df_label.loc[id]["labels"])
        else:
            labels.append(1e-6)
    else:
        pathid_orig_2 = [f for f in os.listdir(os.path.join(path, "store_3", str(id) + "_files")) if not f.startswith(".")]
        for j, id_2 in enumerate(pathid_orig_2):
            pathid.append(id_2)
            images.append(glob.glob(os.path.join(path, "store_3", str(id) + "_files", str(id_2), "0", "*.jpeg")))
            if exists(label_csv):
                labels.append(df_label.loc[id]["labels"])
            else:
                labels.append(1e-6)
c = {"pathid": pathid, "images": images, "labels": labels}
df = pd.DataFrame(c)

df.to_csv("files/store_3.csv", index=False, encoding="utf_8_sig")
torch.save(df.to_dict(orient='list'), "files/store_3.db")

file = torch.load("files/store_3.db")
for i in range(len(file["images"]):
    for j in range(len(file["images"][i])):
        file_name = file["images"][i][j]
        img = cv2.imread(file_name)
        h = img.shape[0]
        while h // 2 > 384:
            img = cv2.resize(img, (h//2, h//2))
            h = h//2
        img = cv2.resize(img, (384, 384))
        
        file_name = file_name.replace("store_3", "store_4")
        
        new_path = os.path.dirname(file_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        cv2.imwrite(file_name, img)



