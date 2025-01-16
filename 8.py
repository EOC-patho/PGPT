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
path = "yourpath"

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

batch_size = 4
num_workers = 0
epochs = 10000
test_every = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = heads * dim_head 
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, channels = 3, emb_dropout = 0., 
                 dim, depth, heads, dim_head = 64, mlp_dim, dropout = 0., pool = 'cls', num_classes):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        #self.mlp_head = nn.Linear(dim, num_classes)
        self.mlp_head = nn.Sequential(
            nn.Linear((num_patches + 1) * dim, 3)
        )
        

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)

        x = self.transformer(x)

        x = torch.reshape(x, (b, (n + 1) * d))
        x = self.to_latent(x)
        
        return self.mlp_head(x)

vit = ViT(
    image_size = 96,
    patch_size = 2, 
    channels = 768,
    dim = 768 * 2,
    depth = 35,
    heads = 12 * 2,
    mlp_dim = 768 * 2 * 4,
    num_classes = 3
)

weights = torch.load("low57.pt")

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

train_lib = "store_5_96_train.csv"
valid_lib = "store_5_96_valid.csv"

transform = transforms.ToTensor()

train_dataset = layer3dataset(train_lib, transform = transform, subsample = 96)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size, shuffle = True,
    num_workers = num_workers
)

valid_dataset = layer3dataset(valid_lib, transform = transform, subsample = 96)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size = batch_size, shuffle = False,
    num_workers = num_workers
)

w = torch.Tensor([0.22, 0.365, 0.415])  
criterion = nn.CrossEntropyLoss(w).to(device)
optimizer = torch.optim.AdamW(vit.parameters(), lr=3e-4, weight_decay = 0.05, eps = 1e-4, betas = [0.9, 0.95])

fconv = open(os.path.join("convergence.csv"), "w")
fconv.write("epoch, metric, value\n")
fconv.close()

best_ba = 0
global best_ba

total_step = len(train_loader)
for epoch in range(epochs):
    epoch_accuracy = 0
    epoch_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = vit(images)
        loss = criterion(outputs, labels.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    
    fconv = open(os.path.join("convergence.csv"), "a")
    fconv.write("{}, loss, {:.4f}\n".format(epoch+1, epoch_loss))
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
                outputs = vit(images)
                outputs = outputs.softmax(dim = -1)
                outputs = outputs.argmax(dim=-1)
                
                acc = (outputs == labels).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
            
                pred.extend(list(outputs.cpu().numpy()))
                real.extend(list(labels.cpu().numpy()))
                
            pred = np.array(pred)
            real = np.array(real)
            ba = balanced_accuracy_score(real, pred)
            
            c={"prediction": pred, "labels": real}
            df_c = pd.DataFrame(c)
            df_c.to_csv("pred_real/pred_{}.csv".format(epoch+1))
            
            neq = np.not_equal(pred, real)
            acc = 1 - float(neq.sum()) / pred.shape[0]
            
            eq = np.equal(pred, real)
            sensi = float(np.logical_and(pred==1, eq).sum()) / (real==1).sum()
            speci = float(np.logical_and(pred==0, eq).sum()) / (real==0).sum()
            
            fpr = float(np.logical_and(pred==1, neq).sum()) / (real==0).sum()
            fnr = float(np.logical_and(pred==0, neq).sum()) / (real==1).sum()
            
            odds_ratio = (float(np.logical_and(pred==1, eq).sum()) * float(np.logical_and(pred==0, eq).sum())) \
                        / (float(np.logical_and(pred==0, neq).sum()) * float(np.logical_and(pred==1, neq).sum()) + 1e-9)
            
            fconv = open(os.path.join("convergence.csv"), "a")
            fconv.write("{}, acc, {:.4f}\n".format(epoch+1, epoch_accuracy))
            fconv.write("{}, val_acc, {:.4f}\n".format(epoch+1, epoch_val_accuracy))
            fconv.write("{}, ba, {:.4f}\n".format(epoch+1, ba))
            fconv.write("{}, sensi, {:.4f}\n".format(epoch+1, sensi))
            fconv.write("{}, speci, {:.4f}\n".format(epoch+1, speci))
            fconv.write("{}, odds_ratio, {:.4f}\n".format(epoch+1, odds_ratio))
            fconv.write("{}, fpr, {}\n".format(epoch+1, fpr))
            fconv.write("{}, fnr, {}\n".format(epoch+1, fnr))
            fconv.close()
              
        torch.save(vit.state_dict(), os.path.join("checkpoint_epoch_{}.pth".format(epoch + 1)))

    if (epoch + 1) % 1 == 0:
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_acc: {epoch_val_accuracy:.4f} - ba: {ba:.4f}\n"
        )






























