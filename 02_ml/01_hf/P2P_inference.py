import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from P2P_utils import inference, load_checkpoint, inferenceJIT
import torch.optim as optim
import P2P_config as config
from P2P_generator_model import Generator

from PIL import Image

torch.backends.cudnn.benchmark = True

learning_rate = config.LEARNING_RATE
batch_size = config.BATCH_SIZE

#different kinds of inference (full model & JIT)

def predict(imageTensor):
    
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    load_checkpoint(
        os.path.join(config.CHECKPOINT_GEN, f"BatchSize_{batch_size}_LR_{learning_rate}", "gen.pth.tar"), gen, opt_gen, learning_rate,
    )
    
    imageTensor = imageTensor  
    imageTensor = imageTensor[None, :]
    
    output = inference(gen, imageTensor)
    
    return output


def predictJIT(imageTensor, path="02_ml/01_hf/model/erosion/v006/v001/BatchSize_16_LR_0.0002"):
    
    imageTensor = imageTensor.to(config.DEVICE)  
    imageTensor = imageTensor[None, :]
    
    output = inferenceJIT(imageTensor, batch_size, learning_rate, path)
    
    output = output[0]
    
    return output


def predictTesting(path="02_ml/01_hf/model/vegetation/v001/v001/BatchSize_4_LR_0.0002"):
    
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    
    img = Image.open(f"{config.ROOT}/02_ml/01_hf/evaluation/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}/input_0.png")
    convert_tensor = transforms.ToTensor()
    
    imageTensor = convert_tensor(img)
    imageTensor = imageTensor[None, :]
    
    output = inference(gen, imageTensor, path)
    
    path = f"{config.ROOT}/02_ml/01_hf/casetesting/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}"
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    save_image(output , f"{config.ROOT}/02_ml/01_hf/casetesting/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}/prediction.png")
    
    
def predictTestingJIT(path="/02_ml/01_hf/model/vegetation/v001/v001/BatchSize_4_LR_0.0002"):
    
    img = Image.open(f"{config.ROOT}/02_ml/01_hf/evaluation/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}/input_0.png")
    # img = Image.open(f"/home/jakob/Desktop/testimg.png")
    convert_tensor = transforms.ToTensor()
    
    imageTensor = convert_tensor(img)
    
    # print(imageTensor)
    
    imageTensor = imageTensor[None, :]
    imageTensor = imageTensor.to(config.DEVICE)
    
    # print(imageTensor)
    
    output = inferenceJIT(imageTensor, batch_size, learning_rate, path)
    
    # print(output)
    
    path = f"{config.ROOT}/02_ml/01_hf/casetesting/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}"
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    save_image(output , f"{config.ROOT}/02_ml/01_hf/casetesting/{config.APPLICATION}/{config.VERSION}/{config.RUN}/BatchSize_{batch_size}_LR_{learning_rate}/prediction.png")


if __name__ == '__main__':
    predictTestingJIT()