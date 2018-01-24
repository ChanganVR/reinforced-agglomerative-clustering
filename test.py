import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
import inspect
from torch.nn.utils import rnn
from utils_pad import pack_sequence
from utils_pad import pad_sequence
from utils_pad import prepare_sequence

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size_low, hidden_size_high):
        super(DQN, self).__init__()
        self.hidden_size_low = hidden_size_low
        self.hidden_size_high = hidden_size_high
        self.hidden_low, self.hidden_high = self.init_hidden()
        self.gru_low = nn.GRU(input_size, hidden_size_low, batch_first=False, bidirectional=False)
        self.gru_high = nn.GRU(hidden_size_low, hidden_size_high, batch_first=True, bidirectional=False)

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size_low).type(FloatTensor)), Variable(torch.zeros(1,1,self.hidden_size_high).type(FloatTensor))

    def forward(self, input):
        batch = input[1][0]
        repeat_hidden_low = self.hidden_low.repeat(1,batch,1)
        _, cluster_rep = self.gru_low(input, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        return state_rep


# a = Variable(torch.randn(5,10)).type(FloatTensor)
# b = Variable(torch.randn(4,10)).type(FloatTensor)
# c = Variable(torch.randn(3,10)).type(FloatTensor)
# packed_seq = pack_sequence([a, b, c])

features = np.random.randn(10,20)
partition = [[1,3,5,7],[2],[0,4],[6,8],[9]]

input = prepare_sequence(features, partition)

model = DQN(20,5,3)
model.cuda()

h = model(input)