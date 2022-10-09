import torch
from FFN_model import NeuralNetAdv
import FFN_config as config

def test():
    x = torch.randn((1, 3, 128, 128))
    x = x.reshape(-1, 3*128*128)
    model = NeuralNetAdv(config.INPUT_SIZE, config.HIDDEN_SIZE, config.NUM_CLASSES)
    preds = model(x)
    
    print(preds.shape)
    
if __name__ == "__main__":
    test()