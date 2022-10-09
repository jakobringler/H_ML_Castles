import os
import torch
from P2P_utils import createInferenceModel, load_checkpoint
import torch.nn as nn
import torch.optim as optim
import P2P_config as config
from P2P_dataset import HeightfieldDataset
from P2P_generator_model import Generator
from P2P_discriminator_model import Discriminator
from P2P_train import BATCH_SIZES, LEARNING_RATES
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

for batch_size in BATCH_SIZES:
    for learning_rate in LEARNING_RATES:

        gen = Generator(in_channels=3, features=64).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        load_checkpoint(
            os.path.join(config.CHECKPOINT_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "gen.pth.tar"), gen, opt_gen, learning_rate,
        )

        createInferenceModel(gen, batch_size, learning_rate)