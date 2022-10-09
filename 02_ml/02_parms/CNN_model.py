import torch
import torch.nn as nn
import torch.nn.functional as F
import CNN_config as config
    
    
class CNN(nn.Module):
    def __init__(self, inputsize, conv1out, kernelSize, poolSize, stride, conv2out, newSize, fc1out, fc2out, NUM_CLASSES):
        super(CNN, self).__init__()
        self.conv2out = conv2out
        self.newSize = newSize
        self.conv1 = nn.Conv2d(inputsize, conv1out, kernelSize)    
        self.pool = nn.MaxPool2d(poolSize, stride)               
        self.conv2 = nn.Conv2d(conv1out, conv2out, kernelSize)      
        self.fc1 = nn.Linear(conv2out * newSize * newSize, fc1out)  
        self.fc2 = nn.Linear(fc1out, fc2out)                      
        self.fc3 = nn.Linear(fc2out, NUM_CLASSES)                       

    def forward(self, x):
                                                               
        x = self.pool(F.relu(self.conv1(x)))                  
        x = self.pool(F.relu(self.conv2(x)))                   
        x = x.view(-1, self.conv2out * self.newSize * self.newSize) 
        x = F.relu(self.fc1(x))                              
        x = F.relu(self.fc2(x))                              
        x = self.fc3(x)                   
        return x
    
    
    
def testCNN(): 
    
    width = 128

    inputsize = 3 # 3 channels (rgb)
    conv1out = 6 
    kernelSize = 5

    conv2out = 16

    poolSize = 2
    stride = 2

    fc1out = config.HIDDEN_SIZE_FC1
    fc2out = config.HIDDEN_SIZE_FC2

    filterSize = 5
    padding = 0
    convStride = 1

    newSize = int((((((width - filterSize + 2 * padding)/convStride + 1)/poolSize) - filterSize + 2 * padding)/convStride + 1)/poolSize)
    
    x = torch.randn((1, 3, 128, 128))
    # x = x.reshape(-1, 3*128*128)
    model = CNN(inputsize, conv1out, kernelSize, poolSize, stride, conv2out, newSize, fc1out, fc2out, config.CNN_NUM_CLASSES)
    
    print(model)
    
    preds = model(x)
    
    print(preds)
    print(preds.shape)
    

if __name__ == "__main__":
    testCNN()