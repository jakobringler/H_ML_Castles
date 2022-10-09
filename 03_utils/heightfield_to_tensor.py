import hou


node = hou.pwd()
geo = node.geometry()

import os
import torch
import numpy as np
from PIL import Image

HF = hou.node('../HF')

samples = HF.parm('gridsamples').eval()

W = samples
D = W

hRange = 75

HIP = hou.expandString('$HIP')
savePath = os.path.join(HIP, "images/hftest/hftest_tensor.png")

# -- Functions -- 

def fit(val, oldmin, oldmax, newmin, newmax):
    return ( (val - oldmin ) / ( oldmax - oldmin) ) * (newmax - newmin) + newmin

# -- Main --

heightfield = geo.prim(0)

voxels = heightfield.allVoxels()

# prep heightfield

height = np.asarray(voxels, dtype=np.float64)
height = height.reshape(W,D)
heightC = np.rot90(np.fliplr(height))
remappedHeight = 255 * fit(heightC, -hRange, hRange, 0, 1)
heightGS = remappedHeight.astype(np.uint8)

# eval model

heightTensor = torch.from_numpy(heightGS).reshape(1,W,D)

# prep tensor for writeback

numpyWriteBack = heightTensor.numpy().reshape(W,D)
remappedWriteBack = fit(numpyWriteBack/255, 0, 1, -hRange, hRange)
writeBackC = np.rot90(np.fliplr(remappedWriteBack))
writeBack = writeBackC.reshape(-1)

# writeback

heightfield.setAllVoxels(writeBack)

# im = Image.fromarray(np.rot90(heightGS)) # rot 90 to match heightfield output node
# im.save(savePath)