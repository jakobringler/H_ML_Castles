import sys, os
from datetime import datetime

import numpy as np

import torch

import torch.nn as nn
import torch.optim as optim

from torch.utils import backcompat
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image

import CNN_config as config
from CNN_model import CNN

# cuda DEVICE config

DEVICE = config.DEVICE


# HYPERPARAMETERS

NUM_CLASSES = config.CNN_NUM_CLASSES
HIDDEN_SIZE_FC1 = config.HIDDEN_SIZE_FC1
HIDDEN_SIZE_FC2 = config.HIDDEN_SIZE_FC2
ROOT = config.ROOT

PATH = f"{ROOT}/02_ml/02_parms/model/v003/v005/Batch_Size_8_LR_0.001/model_cnn.pth"

# Model 

width = 128

inputsize = 3
conv1out = 6 
kernelSize = 5

conv2out = 16

poolSize = 2
stride = 2

fc1out = HIDDEN_SIZE_FC1
fc2out = HIDDEN_SIZE_FC2

filterSize = 5
padding = 0
convStride = 1
newSize = int((((((width - filterSize + 2 * padding)/convStride + 1)/poolSize) - filterSize + 2 * padding)/convStride + 1)/poolSize)
# model = NeuralNetAdv(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)
model = CNN(inputsize, conv1out, kernelSize, poolSize, stride, conv2out, newSize, fc1out, fc2out, NUM_CLASSES).to(DEVICE)

# Import Trained Model

model.load_state_dict(torch.load(PATH, map_location=DEVICE))

model = model

scripted_model = torch.jit.script(model)

scripted_model.save(f"{ROOT}/02_ml/02_parms/model/v003/v005/Batch_Size_8_LR_0.001/inf/inference_CNN.pt")