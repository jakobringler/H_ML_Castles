import torch
import numpy as np

# cuda DEVICE config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS

ROOT = "/mnt/workssd/10_Projects/2108_BA_Houdini_ML/Abgabe/Praxis" # change this path to the project root
VERSION = "v003"
RUN = "v005"
W = 128
SAMPLES = 10000
INPUT_SIZE = 49152 # 3 x 128 x 128 / rgb * height * width
CNN_NUM_CLASSES = 3 # 0 nothing, 1 roof, 2 pinnacles 
CNN_BATCH_SIZE = 8 
CNN_NUM_EPOCHS = 25
CNN_LEARNING_RATE = 0.001
HIDDEN_SIZE_FC1 = 120
HIDDEN_SIZE_FC2 = 84
TRAIN_SPLIT = int(SAMPLES*0.9)
TEST_SPLIT = int(SAMPLES*0.1)
HYPERPARAMETERSEARCH = False