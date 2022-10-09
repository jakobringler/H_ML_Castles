node = hou.pwd()
geo = node.geometry()

import sys, os
import hou
import time

start = time.time()

HIP = hou.expandString('$HIP')
P2P = os.path.abspath(os.path.join(HIP, '../pix2pix'))
sys.path.append(P2P)

import torch
import torch.optim as optim
from torchvision import transforms
import numpy as np
from PIL import Image

import P2P_utils
import P2P_inference
import P2P_config as config
from P2P_generator_model import Generator

import importlib
importlib.reload(P2P_inference)
importlib.reload(P2P_utils)
importlib.reload(config)

HF = hou.node('../HF')

samples = HF.parm('gridsamples').eval()

W = samples
D = W


# -- Import Data --

heightfield = geo.prim(0)

voxels = heightfield.allVoxels()


# -- Transforms --

convGstoRGB = transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    
convRGBtoGs = transforms.functional.rgb_to_grayscale


# -- Reshape & Remap Data --

height = np.asarray(voxels, dtype=np.float64)
height = height.reshape(W,D)
heightC = np.rot90(np.fliplr(height))
heightTensor = torch.from_numpy(heightC).reshape(1,W,D)
heightTensor = convGstoRGB(heightTensor)
heightTensor = heightTensor.float()

print("-------------------------")
print("----- Start Timer -------")
prep = time.time() - start
print("To Tensor Time: ", round(prep, 4), "s")
print("---- Inference Time -----")

# -- Model Evaluation

heightTensor = P2P_inference.predictJIT(heightTensor)

print("-------------------------")
infTime = time.time() - prep - start
print("Combined Time:  ", round(infTime, 4), "s")
print("-------------------------")

# -- Reverse | Reshape & Remap Data --

heightTensor = heightTensor.double().cpu()
heightTensor = convRGBtoGs(heightTensor)
numpyWriteBack = heightTensor.numpy().reshape(W,D)
writeBackC = np.rot90(np.fliplr(numpyWriteBack))
writeBack = writeBackC.reshape(-1)


# -- Writeback Data to Heightfield --

heightfield.setAllVoxels(writeBack)

writeBackTime = time.time() - prep - infTime - start

print("Writeback Time: ", round(writeBackTime, 4), "s")