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

import FFN_config as config
from FFN_model import NeuralNet, NeuralNetAdv

# cuda DEVICE config

DEVICE = config.DEVICE


# HYPERPARAMETERS

INPUT_SIZE = config.INPUT_SIZE # 3 x 128 x 128 / rgb * height * width
HIDDEN_SIZE = config.HIDDEN_SIZE
NUM_CLASSES = config.NUM_CLASSES # 10 tower parameters
ROOT = config.ROOT

PATH = f"{ROOT}/02_ml/02_parms/model/v002/v012/Batch_Size_8_LR_0.0001_HS_1000/model_ffn.pth"

# Model 

model = NeuralNetAdv(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)


# Import Trained Model

model.load_state_dict(torch.load(PATH, map_location=DEVICE))

model = model

scripted_model = torch.jit.script(model)

scripted_model.save(f"{ROOT}/02_ml/02_parms/model/v002/v012/Batch_Size_8_LR_0.0001_HS_1000/inf/inference_FFN.pt")