import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.Functional as F 
from torch.autograd import Variable

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

def prepare_sequence():
    pass

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirection=True)


    def init_hidden(self, batch_size):
        return Variable(torch.zeros(1, self.batch_size, self.hidden_size))

    def forward(self, input, hidden):
        _, h_n = self.gru(input, hidden)
        return h_n

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4096,1024)
        self.gru_low = GRU(1024,256)
        self.fc2 = nn.Linear(256,256)
        self.gru_high = GRU(256,256)


