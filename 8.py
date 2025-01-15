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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/data2/patho-vit_5_23_oc"

seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)



from collections import OrderedDict
from ViT import ViT

batch_size = 4
num_workers = 0
epochs = 10000
test_every = 1

vit = ViT(
    image_size = 96,
    patch_size = 2, 
    channels = 768,
    dim = 768 * 2,
    depth = 35,
    heads = 12 * 2,
    mlp_dim = 768 * 2 * 4,
    num_classes = 2
)

weights = torch.load("/data2/patho-vit_5_23_oc/gpvit_weight.pt")

new_dict = OrderedDict()
for k, v in weights.items():
    if "module.target_encoder" in k:
        new_key = k[22:]
        new_dict[new_key] = v

vit.to(device)
vit.load_state_dict(new_dict, strict = False)
vit = nn.DataParallel(vit)



class layer3dataset(torch.utils.data.Dataset):
    def __init__(self, libraryfile = "", transform=None, subsample=-1):
        file = pd.read_csv(libraryfile)
        self.pathid = file["pathid"]
        self.images = file["images"]
        self.labels = file["labels"]
        
        self.subsample = subsample
        self.transform = transform
    
    def __getitem__(self, index):
        image = torch.load(self.images[index])
        image = np.array(image)
        
        if self.subsample != -1 and self.transform is not None:
            
            image = self.transform(image)
            image = image.unsqueeze(0)
            image = F.interpolate(image, size =(self.subsample, self.subsample))
            image = image.squeeze(0)
            image = image.to(torch.float32)
        
        label = self.labels[index]
        pathid = self.pathid[index]
        return image, label
    
    def __len__(self):
        return len(self.pathid)

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))

train_lib_1 = "/data2/patho-vit_5_23_oc/files_label/store_5_96_train.csv"
valid_lib = "/data2/patho-vit_5_23_oc/files_label/store_5_96_valid.csv"
valid_lib_1 = "/data2/patho-vit_5_23_oc/files_label/store_5_96_exva.csv"

transform = transforms.ToTensor()

train_dataset_1 = layer3dataset(train_lib_1, transform = transform, subsample = 96)
train_loader = torch.utils.data.DataLoader(
    train_dataset_1,
    batch_size = batch_size, shuffle = True,
    num_workers = num_workers
)

valid_dataset = layer3dataset(valid_lib, transform = transform, subsample = 96)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size = batch_size, shuffle = False,
    num_workers = num_workers
)

valid_dataset_1 = layer3dataset(valid_lib_1, transform = transform, subsample = 96)
valid_loader_1 = torch.utils.data.DataLoader(
    valid_dataset_1,
    batch_size = batch_size, shuffle = False,
    num_workers = num_workers
)

cudnn.benchmark = True
w = torch.Tensor([0.3,0.7]) 
criterion = nn.CrossEntropyLoss(w).to(device)
optimizer = torch.optim.AdamW(vit.parameters(), lr=3e-4, weight_decay = 0.05, eps = 1e-4, betas = [0.9, 0.95])

fconv = open(os.path.join("/data2/patho-vit_5_23_oc/convergence.csv"), "w")
fconv.write("epoch, metric, value\n")
fconv.close()

best_ba = 0

global best_ba

total_step_1 =  len(train_loader)

for epoch in range(epochs):
    torch.cuda.empty_cache()
    epoch_accuracy_lung = 0
    epoch_loss_lung = 0
    epoch_accuracy = 0
    epoch_loss = 0
   
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs, _ = vit(images)
        loss = criterion(outputs, labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    
    print("Epoch [{}/{}], Loss_cc: {:.4f}, Acc_cc: {:.4f}".format(epoch+1, epochs, epoch_loss, epoch_accuracy))
    
    fconv = open(os.path.join("/data2/patho-vit_5_23_oc/convergence.csv"), "a")
    fconv.write("{}, loss, {:.4f}\n".format(epoch+1, epoch_loss))
    fconv.write("{}, acc, {:.4f}\n".format(epoch+1, epoch_accuracy))
    fconv.close()
        
    if (epoch+1) % test_every == 0:
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            
            pred = []
            real = []
            
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = vit(images)
                val_loss = criterion(outputs, labels.long())
                
                outputs = outputs.argmax(dim = -1)
                
                acc = (outputs == labels).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
            
                pred.extend(list(outputs.cpu().numpy()))
                real.extend(list(labels.cpu().numpy()))

            pred = np.array(pred)
            real = np.array(real)
            ba = balanced_accuracy_score(real, pred)
          
            c={"prediction": pred, "labels": real}
            df_c = pd.DataFrame(c)
            df_c.to_csv("/data2/patho-vit_5_23_oc/pred_real_inval/pred_{}.csv".format(epoch+1))
              
            eq = np.equal(pred, real)
            sensi = float(np.logical_and(pred==1, eq).sum()) / (real==1).sum()
            speci = float(np.logical_and(pred==0, eq).sum()) / (real==0).sum()

            fconv = open(os.path.join("/data2/patho-vit_5_23_ccsurv/patient/low/8/convergence.csv"), "a")
            fconv.write("{}, epoch_val_loss, {:.4f}\n".format(epoch+1, epoch_val_loss))
            fconv.write("{}, val_acc, {:.4f}\n".format(epoch+1, epoch_val_accuracy))
            fconv.write("{}, ba, {:.4f}\n".format(epoch+1, ba))
            fconv.write("{}, sensi, {:.4f}\n".format(epoch+1, sensi))
            fconv.write("{}, speci, {:.4f}\n".format(epoch+1, speci))
            fconv.close()


            epoch_test_accuracy = 0
            
            pred_test = []
            real_test = []
            
            for images, labels in valid_loader_1:
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = vit(images)

                outputs = outputs.argmax(dim = -1)
                
                test_acc = (outputs == labels).float().mean()
                epoch_test_accuracy += test_acc / len(valid_loader_1)

                pred_test.extend(list(outputs.cpu().numpy()))
                real_test.extend(list(labels.cpu().numpy()))
   
            pred_test = np.array(pred_test)
            real_test = np.array(real_test)
            ba_test = balanced_accuracy_score(real_test, pred_test)
      
            c6={"prediction": pred_test, "labels": real_test}
            df_c6 = pd.DataFrame(c6)
            df_c6.to_csv("/data2/patho-vit_5_23_oc/pred_real_exval/pred_{}.csv".format(epoch+1))
            eq_test = np.equal(pred_test, real_test)
            sensi_test = float(np.logical_and(pred_test==1, eq_test).sum()) / (real_test==1).sum()
            speci_test = float(np.logical_and(pred_test==0, eq_test).sum()) / (real_test==0).sum()

            fconv = open(os.path.join("/data2/patho-vit_5_23_oc/convergence.csv"), "a")
            fconv.write("{}, test_acc, {:.4f}\n".format(epoch+1, epoch_test_accuracy))
            fconv.write("{}, ba_test, {:.4f}\n".format(epoch+1, ba_test))
            fconv.write("{}, sensi_test, {:.4f}\n".format(epoch+1, sensi_test))
            fconv.write("{}, speci_test, {:.4f}\n".format(epoch+1, speci_test))
            fconv.close()
        
    torch.save(vit.state_dict(), os.path.join("/data2/patho-vit_5_23_oc/WSI_level_{}.pth".format(epoch+1)))

    print(
        f"Epoch : {epoch+1} - val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_accuracy:.4f} - ba: {ba:.4f}\n" 
    )
    print(
        f"Epoch : {epoch+1} - test_acc: {epoch_test_accuracy:.4f} - ba_test: {ba_test:.4f}\n" 
    )





























