import torch
import torch.nn as nn
import FFN_config as config

class NeuralNet(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
    
class NeuralNetAdv(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES):
        super(NeuralNetAdv, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE/2))
        self.l3 = nn.Linear(int(HIDDEN_SIZE/2), int(HIDDEN_SIZE/8))
        self.l4 = nn.Linear(int(HIDDEN_SIZE/8), NUM_CLASSES)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

def testNeuralNetAdv():
    x = torch.randn((1, 3, 128, 128))
    x = x.reshape(-1, 3*128*128)
    model = NeuralNetAdv(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_CLASSES)
    preds = model(x)
    
    print(preds.shape)
    

if __name__ == "__main__":
    testNeuralNetAdv()