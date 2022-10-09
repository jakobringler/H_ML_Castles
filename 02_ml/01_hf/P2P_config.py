# this script is partially based on Aladdin Perssons Pix2Pix Implementation 
# for more info check the license file

import torch
from torchvision import transforms

VERSION = "v001" # select data version
RUN = "v001" # set training run version
APPLICATION = "vegetation" # erosion, vegetation
ROOT = "/path/to/rootfolder" # Set this to local Project root
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = f"{ROOT}/04_data/{APPLICATION}/{VERSION}/train"
VAL_DIR = f"{ROOT}/04_data/{APPLICATION}/{VERSION}/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
HYPERPARAMETER_SEARCH = False
CHECKPOINT_DISC = f"{ROOT}/02_ml/01_hf/model/{APPLICATION}/{VERSION}/{RUN}"
CHECKPOINT_GEN = f"{ROOT}/02_ml/01_hf/model/{APPLICATION}/{VERSION}/{RUN}"
INFERENCE_GEN = f"{ROOT}/02_ml/01_hf/model/{APPLICATION}/{VERSION}/{RUN}"


both_transform = transforms.Compose(
    [
        transforms.Resize([256]),
    ], 
)

transform_only_input = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
)

transform_only_mask = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ]
)