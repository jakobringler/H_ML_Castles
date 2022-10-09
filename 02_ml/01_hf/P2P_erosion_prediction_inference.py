node = hou.pwd()
geo = node.geometry()

import sys, os
import hou
import time

start = time.time()

HIP = hou.expandString('$HIP')
P2P = os.path.abspath(os.path.join(HIP, '../02_ml/01_hf/'))
sys.path.append(P2P)

import torch
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from PIL import Image

import P2P_utils
import P2P_inference
import P2P_config as config
from P2P_generator_model import Generator

# import importlib
# importlib.reload(P2P_inference)
# importlib.reload(P2P_utils)
# importlib.reload(config)

HF = hou.node('../../HF')

samples = HF.parm('gridsamples').eval()

W = samples
D = W


# -- Import Data --

heightfield = geo.prim(0)
foundation = geo.prim(2)
tower = geo.prim(3)
water = geo.prim(4)
debris = geo.prim(5)

voxels = heightfield.allVoxels()
voxelsf = foundation.allVoxels()
voxelst = tower.allVoxels()


# -- Transforms & Functions --

def lin2srgb(lin):
    s = pow(lin, (1.0 / 2.4))
    return s
    
def srgb2lin(srgb):
    l = pow(srgb, 2.4)
    return l

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

heightL = prepData(voxels, W, D, C=True)
foundationL = prepData(voxelsf, W, D)
towerL = prepData(voxelst, W, D)

tensorlist = [heightL, foundationL, towerL]
heightTensor = torch.stack(tensorlist)
heightTensor = heightTensor.float()

heightTensor = P2P_inference.predictJIT(heightTensor)

print("-------------------------")
infTime = time.time() - prep - start
print("Combined Time:  ", round(infTime, 4), "s")
print("-------------------------")


# -- Reverse | Reshape & Remap Data --

heightTensor = heightTensor.double().cpu()

heightWB = reverseData(heightTensor[0, :, :], W, D, C=True)
waterWB = reverseData(heightTensor[1, :, :], W, D)
debrisWB = reverseData(heightTensor[2, :, :], W, D)


# -- Writeback Data to Heightfield --

heightfield.setAllVoxels(heightWB)
water.setAllVoxels(waterWB)
debris.setAllVoxels(debrisWB)

writeBackTime = time.time() - prep - infTime - start

print("Writeback Time: ", round(writeBackTime, 4), "s")