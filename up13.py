import os
import math
import random
from sklearn.metrics import balanced_accuracy_score
import copy
from functools import wraps, partial

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.backends.cudnn as cudnn

from einops import repeat, rearrange 
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "yourpath"

seed = 22

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)



batch_size = 260
num_workers = 48
epochs = 10000

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size = (384, 384),
        patch_size = 16,
        enc_mask_scale = (0.85, 1.0),
        pred_mask_scale = (0.15, 0.2),
        aspect_ratio = (0.75, 1.5),
        nenc = 1,
        npred = 4,
        min_keep = 4,
        allow_overlap = False
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  
        self.allow_overlap = allow_overlap  
        self._itr_counter = Value('i', -1)  

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
    
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
       
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
           
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
           
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

transform = transforms.Compose([
   
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))])

train_lib = "yourdataset.db"

from patho_vit.airport1 import pathovitdataset

train_dataset = pathovitdataset(libraryfile = train_lib, transform = transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size, shuffle = True,
    num_workers = num_workers,
    collate_fn = MaskCollator()
)


def apply_mask(x, mask):
    all_x = []
    for m in mask:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance
            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def get_module_device(module):
    return next(module.parameters()).device

def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss = loss.sum(dim = -1).mean()
    return loss

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Attention(nn.Module):
    def __init__(self, dim, heads = 12, dim_head = 64, dropout = 0.):
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

class Context_encoder(nn.Module):
    def __init__(self, *, image_size = 384, patch_size = 16, channels = 3, emb_dropout = 0., 
                 dim = 768, depth = 12, heads = 12, dim_head = 64, mlp_dim = 768 * 4, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

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

    def forward(self, img, mask):
        device = img.device
        
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = apply_mask(x, mask)
        
        return x

class Predictor(nn.Module):
    def __init__(self, *, num_patches = 576, encoder_dim = 768, 
                 decoder_dim = 768, decoder_depth = 8, decoder_heads = 8, decoder_dim_head = 64):
        super().__init__()
        
        self.num_patches = num_patches
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.black_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)

    def forward(self, x, context_mask, target_mask):
        device = x.device
        batch, num_white, _ = x.shape
        
        white_tokens = self.enc_to_dec(x)
        white_tokens += self.decoder_pos_emb(context_mask[0])
        
        black_tokens1 = repeat(self.black_token, '1 1 d -> b n d', b = batch, n = target_mask[0].size(-1))
        black_tokens1 = black_tokens1 + self.decoder_pos_emb(target_mask[0])
        
        black_tokens2 = repeat(self.black_token, '1 1 d -> b n d', b = batch, n = target_mask[1].size(-1))
        black_tokens2 = black_tokens2 + self.decoder_pos_emb(target_mask[1])
        
        black_tokens3 = repeat(self.black_token, '1 1 d -> b n d', b = batch, n = target_mask[2].size(-1))
        black_tokens3 = black_tokens3 + self.decoder_pos_emb(target_mask[2])
        
        black_tokens4 = repeat(self.black_token, '1 1 d -> b n d', b = batch, n = target_mask[3].size(-1))
        black_tokens4 = black_tokens4 + self.decoder_pos_emb(target_mask[3])
        
        decoder_tokens = torch.zeros(batch, self.num_patches, self.decoder_dim, device=device)
        batch_range = torch.arange(batch, device = device)[:, None]
        decoder_tokens[batch_range, context_mask[0]] = white_tokens
        decoder_tokens[batch_range, target_mask[0]] = black_tokens1
        decoder_tokens[batch_range, target_mask[1]] = black_tokens2
        decoder_tokens[batch_range, target_mask[2]] = black_tokens3
        decoder_tokens[batch_range, target_mask[3]] = black_tokens4
        
        decoder_tokens = self.decoder(decoder_tokens)
        
        context = apply_mask(decoder_tokens, target_mask)
        
        return context

class Jepa(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.context_encoder = Context_encoder()
        self.predictor = Predictor()
        self.target_encoder = None
        
        device = get_module_device(Context_encoder())
        self.to(device)
        
        self.forward(torch.randn(2, 3, 384, 384, device = device), torch.ones(2, 576, dtype  = torch.int64), [torch.ones(2, 576, dtype  = torch.int64), 
                                                                                                             torch.ones(2, 576, dtype  = torch.int64), 
                                                                                                              torch.ones(2, 576, dtype  = torch.int64),                                     
                                                                                                              torch.ones(2, 576, dtype  = torch.int64)])
    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.context_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder
    

        
    def forward(self, x, context_mask, target_mask):
        context = self.context_encoder(x, context_mask)
        context_outputs = self.predictor(context, context_mask, target_mask)
        
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_outputs = target_encoder(x, target_mask)
        
        return context_outputs, target_outputs


from collections import OrderedDict
jepa = Jepa()
jepa.to(device)
jepa = nn.DataParallel(jepa)

optimizer = torch.optim.AdamW(jepa.parameters(), lr= 1.5e-4, weight_decay = 0.2, betas = [0.9, 0.95])
fconv = open(os.path.join("gpvit_convergence.csv"), "w")
fconv.write("epoch, metric, value\n")
fconv.close()

best_loss = 10e5
best_epoch_loss = 10e5
loss_curve = []
total_step = len(train_loader)
for epoch in range(epochs):
    epoch_loss = 0
    for i, (images, context_mask, target_mask) in enumerate(train_loader):
        
        context_outputs, target_outputs = jepa(images[0], context_mask, target_mask) 
        loss = loss_fn(context_outputs, target_outputs)
        loss *= 10e5   
           
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            m = 0.9
            for param_q, param_k in zip(jepa.module.context_encoder.parameters(), jepa.module.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
        epoch_loss += loss / total_step

    loss_curve.append(epoch_loss.cpu().detach())
    
    if (epoch+1) % 1 == 0:
   
        if  epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(jepa.state_dict(), "gpvit_weight_epoch_{}.pt".format(epoch+1))
        
        print("Epoch [{}/{}], Loss: {:.10f}"
             .format(epoch+1, epochs, epoch_loss.item()))
        
        fconv = open(os.path.join("gpvit_convergence.csv"), "a")
        fconv.write("{}, loss, {:.10f}\n".format(epoch+1, epoch_loss.item()))
        fconv.close()

