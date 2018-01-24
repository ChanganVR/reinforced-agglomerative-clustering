import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
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

        self.state_fc = nn.Linear(hidden_size_high, 128)
        self.cluster_fc = nn.Linear(hidden_size_high, 128)
        

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size_low).type(FloatTensor)), Variable(torch.zeros(1,1,self.hidden_size_high).type(FloatTensor))

    def forward(self, input):
        batch = input[1][0]
        repeat_hidden_low = self.hidden_low.repeat(1,batch,1)
        _, cluster_rep = self.gru_low(input, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        return state_rep




model = DQN()
model.cuda()


gamma = 1
eps_start = 0.95
eps_end = 0.05

n_episodes = 1

for i_episode in range(n_episodes):
    features, partition = env.reset()




