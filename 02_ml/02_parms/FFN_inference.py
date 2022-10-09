import sys, os
import time
from pprint import pprint

start = time.time() 

from datetime import datetime
import hou

HIP = hou.expandString('$HIP')
FFN = os.path.abspath(os.path.join(HIP, '../02_ml/02_parms'))
sys.path.append(FFN)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import backcompat
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image

import FFN_config as config

# import importlib
# importlib.reload(config)

node = hou.pwd()
geo = node.geometry()


# cuda DEVICE config

DEVICE = config.DEVICE


# HYPERPARAMETERS

BATCH_SIZE = 1


# VARIABLES

W = config.W
D = W


# Model Load Location


# PATH = f"{config.ROOT}/02_ml/02_parms/model/v001/v003/Batch_Size_8_LR_0.0001_HS_1000/inf/inference_FFN.pt"
# PATH = f"{config.ROOT}/02_ml/02_parms/model/v002/v002/Batch_Size_8_LR_0.0001_HS_1000/inf/inference_FFN.pt"
PATH = f"{config.ROOT}/02_ml/02_parms/model/v002/v012/Batch_Size_8_LR_0.0001_HS_1000/inf/inference_FFN.pt"

# -- Import Data --

tower = geo.prim(3)
roof = geo.prim(2)
pinnacles = geo.prim(4)

voxelst = tower.allVoxels()
voxelsr = roof.allVoxels()
voxelsp = pinnacles.allVoxels()


# -- Transforms & Functions --

def lin2srgb(lin):
    s = pow(lin, (1.0 / 2.4))
    return s
    
def srgb2lin(srgb):
    l = pow(srgb, 2.4)
    return la

def prepData(data, W, D, C=False):
    input = np.asarray(data, dtype=np.float64)
    input = input.reshape(W,D)
    input = torch.from_numpy(input)
    input = torch.flipud(input)
    if C:
        input = lin2srgb(input)
    return input
    
def reverseData(data, W, D, C=False):
    if C:
        input = srgb2lin(data)
    else:
        input = data
    input = torch.flipud(input)
    input = input.numpy().reshape(W,D)
    input = input.reshape(-1)
    return input

    
# -- Reshape & Remap Data --

towerL = prepData(voxelst, W, D)
roofL = prepData(voxelsr, W, D)
pinnaclesL = prepData(voxelsp, W, D)

R = torch.clamp(roofL, 0, 1)
G = torch.clamp(towerL, 0, 1)
B = torch.clamp(pinnaclesL, 0, 1)

tensorlist = [R, G, B]
imgTensor = torch.stack(tensorlist)
imgTensor = imgTensor.float()
imgTensor = imgTensor[None, :]


# Dataset  

class tDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = imgTensor.shape[0]
        self.input = imgTensor
        self.target = np.ndarray([1, 10])
        self.transform = transform

    def __getitem__(self, index):
        sample = self.input[index], self.target[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

dataset = tDataset()

pred_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

prep = time.time() - start

print(f"Prep Time:               {prep:.4f} s")


# Sanity Check

# examples = iter(pred_loader)
# example_data, example_targets = examples.next()

# #print("Example Input:\n", example_data[0])
# print('--------------------------')
# print("Example Input Shape:   ",example_data[0].shape)
# print('--------------------------')
# print("Example Target:        ",example_targets[0])
# print('--------------------------')
# print("Example Target Shape:  ",example_targets[0].shape)
# print('--------------------------')
# print("Example Target Shape:  ",example_targets[0].type())
# print('--------------------------')
# print("Pred Set Batch Shape:  ", example_data.shape)
# print('--------------------------')
# print("Pred Set Length:       ", dataset.__len__())
# print('--------------------------')


model = torch.jit.load(PATH, map_location=DEVICE)

# model = model.to(DEVICE)

model.eval()

modelload = time.time() - start - prep

print(f"Load Time:               {modelload:.4f} s")
    
with torch.no_grad():
    loss = 0
    error = 0
    n_SAMPLES = 0
    for i, (input, target) in enumerate(pred_loader):
    
        input = input.reshape(-1, 3*128*128).to(DEVICE)
        
        target = target.to(DEVICE)
        
        outputs = model(input)
        
        predicted = F.relu(outputs.to('cpu'))
        
        pnp = predicted.numpy().flatten()

        
        
        
        
    for i,point in enumerate(geo.points()):
        
        point.setAttribValue("pred", pnp.astype(np.float64))
        
inference = time.time() - start - modelload - prep

print(f"Inference Time:          {inference:.4f} s")
print('---------------------------------')
est = time.time() - start
print(f"Estimation Completed in: {est:.4f} s")
print('---------------------------------')

pprint(pnp)