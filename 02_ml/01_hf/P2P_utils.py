# this script is partly based on Aladdin Perssons Pix2Pix Implementation 
# for more info check the license file

import os
import torch
import P2P_config as config
from torchvision.utils import save_image

# utility functions to use for saving and loading checkpoints while training as well as inference applications

def save_some_examples(gen, val_loader, epoch, folder):
    
    path = f"{config.ROOT}/02_ml/01_hf/evaluation/{config.APPLICATION}/{config.VERSION}/{config.RUN}"
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake # * 0.5 + 0.5 # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x , folder + f"/input_{epoch}.png") # * 0.5 + 0.5
        if epoch == 1:
            save_image(y , folder + f"/label_{epoch}.png") # * 0.5 + 0.5
    gen.train()


def save_checkpoint(model, optimizer, filename=f"{config.ROOT}/02_ml/01_hf/model/{config.APPLICATION}/my_checkpoint.pth.tar"):
    
    gen_path = config.CHECKPOINT_GEN
    
    if not os.path.isdir(gen_path):
        os.mkdir(gen_path)
    
    disc_path = config.CHECKPOINT_DISC
    
    if not os.path.isdir(disc_path):
        os.mkdir(disc_path)
    
    path = os.path.dirname(filename)
        
    if not os.path.isdir(path):
        os.mkdir(path)
    
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        

def inference(gen, imageTensor):
    
    x = imageTensor 
    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake #  * 0.5 + 0.5  # remove normalization
        return y_fake
    
    
def inferenceJIT(imageTensor, batch_size, learning_rate, path="02_ml/01_hf/model/erosion/v006/v001/BatchSize_16_LR_0.0002"):
    
    import time
    start = time.time() 
    
    x = imageTensor 
    
    
    model = torch.jit.load(os.path.join(config.ROOT, path, "inf/inference_gen.pt"))

    # model = torch.jit.load(os.path.join(config.INFERENCE_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "inf/inference_gen.pt"))
    
    load = time.time() - start
    print("Load Time:      ", round(load, 4), "s")
    
    model = model.to(config.DEVICE)
    model.eval()
    
    with torch.no_grad():
        prep = time.time() - start - load
        print("Prep Time:      ", round(prep, 4), "s")
        y_fake = model(x)
        eval = time.time() - start - load - prep
        print("Eval Time:      ", round(eval, 4), "s")
        return y_fake
    

def createInferenceModel(gen, batch_size, learning_rate):
    
    path = os.path.join(config.INFERENCE_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "inf")
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    model = gen
    scripted_model = torch.jit.script(model)
    
    scripted_model.save(os.path.join(path, "inference_gen.pt"))