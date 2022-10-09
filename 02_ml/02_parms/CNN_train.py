import os, sys
import hou

HIP = hou.expandString('$HIP')
FFN = os.path.abspath(os.path.join(HIP, '../02_ml/02_parms'))
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

import CNN_config as config 
import CNN_model

import importlib
importlib.reload(config)
importlib.reload(CNN_model)

from CNN_model import CNN

# start timer

start = timeit.default_timer()


# cuda DEVICE config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# HYPERPARAMETERS

SAMPLES = len(upstream_items)
INPUT_SIZE = config.INPUT_SIZE
HIDDEN_SIZE_FC1 = config.HIDDEN_SIZE_FC1
HIDDEN_SIZE_FC2 = config.HIDDEN_SIZE_FC2
NUM_CLASSES = config.CNN_NUM_CLASSES
NUM_EPOCHS = config.CNN_NUM_EPOCHS
BATCH_SIZE = config.CNN_BATCH_SIZE
LEARNING_RATE = config.CNN_LEARNING_RATE
TRAIN_SPLIT = int(SAMPLES*0.8)
TEST_SPLIT = int(SAMPLES*0.2)

ROOT = config.ROOT
VERSION = config.VERSION
RUN = config.RUN


# variables

if config.HYPERPARAMETERSEARCH:
    BATCH_SIZES = [int(BATCH_SIZE/12), int(BATCH_SIZE/6), int(BATCH_SIZE)]
    LEARNING_RATES = [LEARNING_RATE/10, LEARNING_RATE, LEARNING_RATE*10]
       
else: 
    BATCH_SIZES = [BATCH_SIZE]
    LEARNING_RATES = [LEARNING_RATE]

towerDataset = []

classes = ('nothing', 'roof', 'pinnacles')


# Model Output Location

hip = hou.expandString('$HIP') 
path = os.path.join(hip, "../02_ml/02_parms/model/model_cnn.pth")


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
        roof = float(work_item.attribValue("roof"))
        pinnacles = float(work_item.attribValue("pinnacles"))
        classification = 0
        
        if roof == 1:
            classification = 1
        elif pinnacles == 1:
            classification = 2
        
        # join to path
    
        path = os.path.join(hou.expandString(dir), file)
        
        # load iamge & convert to tensor
        
        img = Image.open(path)        
        convert_tensor = transforms.ToTensor()        
        imgTensor = convert_tensor(img)
    
        # assemble attributes to tensor
        
        result = torch.tensor(classification)       
        result = result.type(torch.LongTensor)        
        tower = imgTensor, result        
        towerDataset.append(tower) 


# prepare Dataset
    
dataset = tDataset()

train_set, test_set = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT])

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

model = CNN(inputsize, conv1out, kernelSize, poolSize, stride, conv2out, newSize, fc1out, fc2out, NUM_CLASSES).to(DEVICE)


for batch_size in BATCH_SIZES:
    for learning_rate in LEARNING_RATES:


        # Tensorboard Setup        

        writerpath = f'{ROOT}/02_ml/02_parms/tensorboard/{VERSION}/{RUN}/BatchSize_{batch_size}_LR_{learning_rate}/tensorboard'

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
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            
            
        # Training Loop

        print('-------  TRAINING   ------')
        print('--------------------------')

        step = 0
        losses = []
        accuracies = []

        n_totalsteps = len(train_loader)

        for epoch in range(NUM_EPOCHS):

            for i, (input, target) in enumerate(train_loader):

                input = input.to(DEVICE)
                target = target.to(DEVICE)
                
                # forward
                        
                outputs = model(input)
                loss = criterion(outputs, target) 
                
                _, predicted = torch.max(outputs, 1)

                writer.add_scalar(f'Chart1/Loss', loss , global_step=step)
                # writer.add_scalar(f'Chart1/Accuracy', accuracy , global_step=step)

                # backward
                        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step += 1

                if (i+1) % 500 == 0:
                    print(f'epoch {epoch+1} / {NUM_EPOCHS}, step {i+1}/{n_totalsteps}, loss {loss.item():.4f}')
            '''        
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                n_class_correct = [0 for i in range(10)]
                n_class_samples = [0 for i in range(10)]
                
                for input, target in test_loader:

                    input = input.to(DEVICE)
                    target = target.to(DEVICE)
                    outputs = model(input)
                    # max returns (value ,index)
                    _, predicted = torch.max(outputs, 1)
                    n_samples += target.size(0)
                    n_correct += (predicted == target).sum().item()
                    
                    for i in range(batch_size):
                        label = target[i]
                        pred = predicted[i]
                        if (label == pred):
                            n_class_correct[label] += 1
                        n_class_samples[label] += 1
                    acc = 100.0 * n_correct / n_samples
                    print(f'Accuracy of the network: {acc} %')

                    for i in range(3):
                        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                        print(f'Accuracy of {classes[i]}: {acc} %')
            '''
                    
            
        if config.HYPERPARAMETERSEARCH:
            writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                            {'accuracy': sum(accuracies)/len(accuracies), 'loss': sum(losses)/len(losses)})


        # Save Model
        
        path = f"{ROOT}/02_ml/02_parms/model/{VERSION}/{RUN}/Batch_Size_{batch_size}_LR_{learning_rate}"
        p = f"/{VERSION}/{RUN}/Batch_Size_{batch_size}_LR_{learning_rate}"
        file = "model_cnn.pth"

        if not os.path.exists(path):
            os.makedirs(path)

        modelDir = os.path.join(path, file)

        print('--------------------------')

        torch.save(model.state_dict(), modelDir)

        print("Model saved at: ", os.path.join(p, file))
        print('--------------------------')
        print('------- MODEL EVAL  ------')


        # Test Model
        
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]
            
            for input, target in test_loader:

                input = input.to(DEVICE)
                target = target.to(DEVICE)
                outputs = model(input)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += target.size(0)
                n_correct += (predicted == target).sum().item()
                
                for i in range(batch_size):
                    label = target[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1
                    
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network: {acc} %')

            for i in range(3):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                print(f'Accuracy of {classes[i]}: {acc} %')
                        
                    
stop = timeit.default_timer()
print("--- Finished in ---")
print('Time: ', round(stop - start,3), " Seconds")
print(dt.now())