import torch
import numpy as np

# cuda DEVICE config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HYPERPARAMETERS

ROOT = "/mnt/workssd/10_Projects/2108_BA_Houdini_ML/Abgabe/Praxis"
VERSION = "v003"
RUN = "v005"
W = 128
SAMPLES = 10000
INPUT_SIZE = 49152 # 3 x 128 x 128 / rgb * height * width
HIDDEN_SIZE = 5000
NUM_CLASSES = 10 # 10 tower parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 8 # 8
LEARNING_RATE = 0.0001

TRAIN_SPLIT = int(SAMPLES*0.9)
TEST_SPLIT = int(SAMPLES*0.1)
HYPERPARAMETERSEARCH = False