# this script is partially based on Aladdin Perssons Pix2Pix Implementation 
# for more info check the license file

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import P2P_config as config
from P2P_dataset import HeightfieldDataset
from P2P_utils import save_checkpoint, load_checkpoint, save_some_examples
from P2P_generator_model import Generator
from P2P_discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

b = config.BATCH_SIZE
l = config.LEARNING_RATE

# hyperparameter steps when searching

if config.HYPERPARAMETER_SEARCH:  
    BATCH_SIZES = [int(b/16), int(b/4), b, int(b*2), int(b*4)]   
    LEARNING_RATES = [l*0.1, l*0.5, l*1, l*2, l*10]
else:
    BATCH_SIZES = [b]   
    LEARNING_RATES = [l]
    
    batch_size = b
    learning_rate = l


def main():

    # for batch_size in BATCH_SIZES:
    #     for learning_rate in LEARNING_RATES:
    
    # tensorboard initialization
    
    step = 0
    
    writer = SummaryWriter(f'{config.ROOT}/02_ml/01_hf/runs/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}/tensorboard')
    
    #define models and respective optimizers & loss functions
    
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # load model from checkpoint

    if config.LOAD_MODEL:
        load_checkpoint(
            os.path.join(config.CHECKPOINT_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "gen.pth.tar"), gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            os.path.join(config.CHECKPOINT_DISC, f"BatchSize_{batch_size}_LR_{learning_rate}", "disc.pth.tar"), disc, opt_disc, config.LEARNING_RATE,
        )

    # load dataset

    train_dataset = HeightfieldDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = HeightfieldDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # training loop
    
    for epoch in range(config.NUM_EPOCHS):

        loop = tqdm(train_loader, leave=True)

        global GeneratorLoss
        global DiscriminatorLoss
        global L1Loss
        global losses
        
        losses = []

        # x, y = next(iter(train_loader))
        
        # batch loop
        
        for idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            # Train Discriminator
            with torch.cuda.amp.autocast():
                y_fake = gen(x)
                D_real = disc(x, y)
                D_real_loss = BCE(D_real, torch.ones_like(D_real))
                D_fake = disc(x, y_fake.detach())
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train generator
            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                L1 = L1_LOSS(y_fake, y) * config.L1_LAMBDA
                G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )
                
            L1Loss = L1  
            GeneratorLoss = G_loss
            DiscriminatorLoss = D_loss
            losses.append(L1Loss)
            
            # track loss data in tensorboard
            
            writer.add_scalar(f'Chart1 - L1 Loss', L1Loss , global_step=step)   
            writer.add_scalars('Chart2', {'D_real_loss': D_real_loss,
                                            'G_fake_loss': G_fake_loss,
                                            } , global_step=step)
            writer.add_scalars(f'Chart3', {'Discriminator Loss': DiscriminatorLoss,
                                            'Generator Loss': GeneratorLoss, 
                                            }, global_step=step)

            
            step += 1

        # save checkpoint

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=os.path.join(config.CHECKPOINT_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "gen.pth.tar"))
            save_checkpoint(disc, opt_disc, filename=os.path.join(config.CHECKPOINT_DISC, f"BatchSize_{batch_size}_LR_{learning_rate}", "disc.pth.tar"))

        save_some_examples(gen, val_loader, epoch, folder=f"{config.ROOT}/02_ml/01_hf/evaluation/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}")
        
        writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                        {'loss': sum(losses)/len(losses)})


if __name__ == "__main__":
    main()