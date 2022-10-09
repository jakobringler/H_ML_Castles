import torch
import torch.nn as nn
import FFN_config as config


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES):
        super(RNN, self).__init__()
        self.num_layers = NUM_LAYERS
        self.hidden_size = HIDDEN_SIZE
        self.rnn = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        # x -> batch_size, seq, input_size
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)
        out, _ = self.rnn(x, h0)
        # batch_size, seq_lenght, hidden_size
        # (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        out = self.fc(out)
        
        return out