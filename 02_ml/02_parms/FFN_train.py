import os, sys
import hou

HIP = hou.expandString('$HIP')
FFN = os.path.abspath(os.path.join(HIP, '../FFN'))
sys.path.append(FFN)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.utils import backcompat

import numpy as np

import math

from PIL import Image

from torch.utils.tensorboard import SummaryWriter

import timeit
from datetime import datetime as dt

import FFN_config as config 
from FFN_model import NeuralNet

import importlib
importlib.reload(config)


# start timer

start = timeit.default_timer()


# cuda DEVICE config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# HYPERPARAMETERS

SAMPLES = len(upstream_items)
INPUT_SIZE = config.INPUT_SIZE
HIDDEN_SIZE = config.HIDDEN_SIZE
NUM_CLASSES = config.NUM_CLASSES
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
TRAIN_SPLIT = int(SAMPLES*0.8)
TEST_SPLIT = int(SAMPLES*0.2)

ROOT = config.ROOT
VERSION = config.VERSION
RUN = config.RUN


# variables

if config.HYPERPARAMETERSEARCH:
    BATCH_SIZES = [int(BATCH_SIZE/12), int(BATCH_SIZE/6), int(BATCH_SIZE)]
    LEARNING_RATES = [LEARNING_RATE/10, LEARNING_RATE, LEARNING_RATE*10]
    HIDDEN_SIZES = [int(HIDDEN_SIZE/10), int(HIDDEN_SIZE), int(HIDDEN_SIZE*10)]
       
else: 
    BATCH_SIZES = [BATCH_SIZE]
    LEARNING_RATES = [LEARNING_RATE]
    HIDDEN_SIZES = [HIDDEN_SIZE]

towerDataset = []


# Model Output Location

hip = hou.expandString('$HIP') 

path = os.path.join(hip, "../02_ml/02_parms/model/model_ffn.pth")


# create dataset

class tDataset(Dataset):
    
    def __init__(self):
        # data loading
        xy = towerDataset
        x,y = zip(*xy)
        self.x = x
        self.y = y
        self.n_SAMPLES = len(towerDataset)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # len (dataset)
        return self.n_SAMPLES

# read attributes

for upstream_item in upstream_items:
    sampleIndex = SAMPLES - 1
    if upstream_item.index > sampleIndex:
        pass
    else:
        work_item = upstream_item
        
        # extract attributes
        
        i = int(work_item.index)
        dir = str(work_item.attribValue("directory"))
        file = str(work_item.attribValue("filename"))
        width = float(work_item.attribValue("width"))
        taper = float(work_item.attribValue("taper"))
        balc_width = float(work_item.attribValue("balc_width"))
        balc_height = float(work_item.attribValue("balc_height"))
        balc_trans = float(work_item.attribValue("balc_trans"))
        roof = float(work_item.attribValue("roof"))
        roof_height = float(work_item.attribValue("roof_height"))
        bow = float(work_item.attribValue("bow"))
        overshoot = float(work_item.attribValue("overshoot"))
        pinnacles = float(work_item.attribValue("pinnacles"))
        
        # join to path
    
        path = os.path.join(hou.expandString(dir), file)
        
        # load iamge & convert to tensor
        
        img = Image.open(path)
        
        convert_tensor = transforms.ToTensor()
        
        imgTensor = convert_tensor(img)
    
        # assemble attributes to tensor
        
        result = torch.tensor([width, taper, balc_width, balc_height, balc_trans, roof, roof_height, bow, overshoot, pinnacles])
        
        #result = result.type(torch.LongTensor)
        
        tower = imgTensor, result
        
        towerDataset.append(tower) 


# prepare Dataset
    
dataset = tDataset()

train_set, test_set = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT])


# Model 

model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(DEVICE)


for batch_size in BATCH_SIZES:
    for learning_rate in LEARNING_RATES:
        for hidden_size in HIDDEN_SIZES:
            
            # Tensorboard Setup        

            writerpath = f'{ROOT}/02_ml/02_parms/tensorboard/{VERSION}/{RUN}/BatchSize_{batch_size}_LR_{learning_rate}_HS_{hidden_size}/tensorboard'

            if not os.path.exists(writerpath):
                os.makedirs(writerpath)
                
            writer = SummaryWriter(writerpath)
                
                   
            # Loaders
            
            train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

            test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


            # Sanity Check 

            examples = iter(train_loader)
            example_data, example_targets = examples.next()

            print("Example Input:\n", example_data[0])
            print('--------------------------')
            print("Example Input Shape:   ",example_data[0].shape)
            print('--------------------------')
            print("Example Target:        ",example_targets[0])
            print('--------------------------')
            print("Example Target Shape:  ",example_targets[0].shape)
            print('--------------------------')
            print("Example Target Shape:  ",example_targets[0].type())
            print('--------------------------')
            print("Train Set Batch Shape: ",example_data.shape)
            print('--------------------------')
            print("Train Set Length:      ",train_set.__len__())
            print('--------------------------')
            print("Test Set Length:       ",test_set.__len__())
            print('--------------------------')
            

            # Loss

            criterion = nn.L1Loss()
            mse = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                
            # Training Loop

            print('-------  TRAINING   ------')
            print('--------------------------')

            step = 0
            losses = []
            accuracies = []

            n_totalsteps = len(train_loader)

            for epoch in range(NUM_EPOCHS):
                for i, (input, target) in enumerate(train_loader):
                    
                    input = input.reshape(-1, 3*128*128).to(DEVICE)
                    target = target.to(DEVICE)
                    
                    # forward
                           
                    outputs = model(input)
                    loss = criterion(outputs, target) 
                    
                    with torch.no_grad():
                        accuracy = (target - outputs) / target
                        accuracy = 1 - torch.mean(torch.clamp(accuracy, 0, 1))
                    
                    writer.add_scalar(f'Chart1/Loss', loss , global_step=step)
                    writer.add_scalar(f'Chart1/Accuracy', accuracy , global_step=step)
                    
                    # backward
                         
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    step += 1 # * batch_size

                    if (epoch+1) % 10 == 0:
                        losses.append(loss)
                        accuracies.append(accuracy)
                        print(f'epoch {epoch+1} / {NUM_EPOCHS}, step {i+1}/{n_totalsteps}, loss {loss.item():.4f}')
            if config.HYPERPARAMETERSEARCH:
                writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, 'hsize': hidden_size},
                                {'accuracy': sum(accuracies)/len(accuracies), 'loss': sum(losses)/len(losses)})


            # Save Model
            
            path = f"{ROOT}/02_ml/02_parms/model/{VERSION}/{RUN}/Batch_Size_{batch_size}_LR_{learning_rate}_HS_{hidden_size}"
            file = "model_ffn.pth"

            if not os.path.exists(path):
                os.makedirs(path)

            modelDir = os.path.join(path, file)

            print('--------------------------')

            torch.save(model.state_dict(), modelDir)

            print("Model saved at: ",modelDir)
            print('--------------------------')
            print('------- MODEL EVAL  ------')


            # Test Model

            model.eval()

            losses = 0
            mses = 0

            with torch.no_grad():
                loss = 0
                error = 0
                n_SAMPLES = 0
                for i, (input, target) in enumerate(test_loader):
                
                    input = input.reshape(-1, 3*128*128).to(DEVICE)
                    
                    target = target.to(DEVICE)
                    
                    outputs = model(input)

                    error = mse(outputs, target)
                    loss = criterion(outputs, target)
                    
                    mses += error
                    losses += loss
                    
                    n_SAMPLES += 1

                meanSError = mses / n_SAMPLES
                avgLoss = losses / n_SAMPLES
                
                print('-----------------------------')
                print("Mean Squared Error: ", round(meanSError.item(),5))
                print('-----------------------------')
                print("Average Loss:       ", round(avgLoss.item(),5))
                print('-----------------------------')
                    
                    
stop = timeit.default_timer()
print("--- Finished in ---")
print('Time: ', round(stop - start,3), " Seconds")
print(dt.now())