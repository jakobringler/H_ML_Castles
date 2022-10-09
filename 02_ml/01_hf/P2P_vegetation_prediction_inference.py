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
mask = geo.prim(1)
water = geo.prim(4)
trees = geo.prim(6)
swamp = geo.prim(7)
bush = geo.prim(8)

voxels = heightfield.allVoxels()
voxelsm = mask.allVoxels()
voxelsw = water.allVoxels()


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
maskL = prepData(voxelsm, W, D)
waterL = prepData(voxelsw, W, D)

tensorlist = [heightL, maskL, waterL]
heightTensor = torch.stack(tensorlist)
heightTensor = heightTensor.float()

print("-------------------------")
print("------ VEGETATION -------")
print("----- Start Timer -------")
prep = time.time() - start
print("To Tensor Time: ", round(prep, 4), "s")
print("---- Inference Time -----")


# -- Model Evaluation

img = heightTensor

heightTensor = P2P_inference.predictJIT(heightTensor, path="02_ml/01_hf/model/vegetation/v001/v001/BatchSize_4_LR_0.0002")

print("-------------------------")
infTime = time.time() - prep - start
print("Combined Time:  ", round(infTime, 4), "s")
print("-------------------------")


# -- Reverse | Reshape & Remap Data --

heightTensor = heightTensor.double().cpu()

treesWB = reverseData(heightTensor[0, :, :], W, D)
swampWB = reverseData(heightTensor[1, :, :], W, D)
bushWB = reverseData(heightTensor[2, :, :], W, D)


# -- Writeback Data to Heightfield --

trees.setAllVoxels(treesWB)
swamp.setAllVoxels(swampWB)
bush.setAllVoxels(bushWB)

writeBackTime = time.time() - prep - infTime - start

print("Writeback Time: ", round(writeBackTime, 4), "s")