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

class DQRN(nn.Module):
    def __init__(self, input_size, hidden_size_low, hidden_size_high):
        super(DQRN, self).__init__()
        self.hidden_size_low = hidden_size_low
        self.hidden_size_high = hidden_size_high
        self.hidden_low, self.hidden_high = self.init_hidden()
        self.gru_low = nn.GRU(input_size, hidden_size_low, batch_first=False, bidirectional=False)
        self.gru_high = nn.GRU(hidden_size_low, hidden_size_high, batch_first=True, bidirectional=False)

        self.state_fc = nn.Linear(hidden_size_high, 4)
        self.cluster_fc = nn.Linear(hidden_size_low, 4)
        self.agent_fc1 = nn.Linear(8, 4)
        self.agent_fc2 = nn.Linear(4, 1)

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size_low).type(FloatTensor)), Variable(torch.zeros(1,1,self.hidden_size_high).type(FloatTensor))

    def forward(self, input):
        batch = input[1][0]
        repeat_hidden_low = self.hidden_low.repeat(1,batch,1)
        _, cluster_rep = self.gru_low(input, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        cluster_rep = torch.squeeze(cluster_rep)
        state_rep = torch.squeeze(state_rep)

        cluster_rep = F.tanh(self.cluster_fc(cluster_rep))
        state_rep = F.tanh(self.state_fc(state_rep))
        q_table = Variable(torch.zeros(batch*(batch-1)/2))

        # print 'cluster_rep', cluster_rep
        # print 'state_rep', state_rep

        count = 0
        for i in range(batch):
            for j in range(i):
                merge_cluster = cluster_rep[i,:] + cluster_rep[j,:]
                merge_rep = torch.cat([state_rep, merge_cluster])
                q = F.relu(self.agent_fc1(merge_rep))
                q = self.agent_fc2(q)
                q_table[count] = q 

                count += 1

        q_table = nn.Softmax()(q_table)
        # print 'softmax q_table', q_table
        return q_table