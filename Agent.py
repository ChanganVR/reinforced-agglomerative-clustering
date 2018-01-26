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

        self.state_fc = nn.Linear(hidden_size_high, 16)
        self.cluster_fc = nn.Linear(hidden_size_low, 16)
        self.agent_fc1 = nn.Linear(32, 32)
        self.agent_fc2 = nn.Linear(32, 1)

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size_low).type(FloatTensor)), Variable(torch.zeros(1,1,self.hidden_size_high).type(FloatTensor))

    def forward(self, input):
        partition, images = input
        n_images = images.size(0)
        n_cluster = len(partition)

        packed_seq = prepare_sequence(partition, images)

        repeat_hidden_low = self.hidden_low.repeat(1,n_cluster,1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        cluster_rep = torch.squeeze(cluster_rep)
        # print cluster_rep
        state_rep = torch.squeeze(state_rep)

        cluster_rep = F.relu(self.cluster_fc(cluster_rep))
        state_rep = F.relu(self.state_fc(state_rep))
        q_table = Variable(torch.zeros(n_cluster*(n_cluster-1)/2).type(FloatTensor))

        count = 0
        for i in range(n_cluster):
            for j in range(i):
                merge_cluster = cluster_rep[i,:] + cluster_rep[j,:]
                merge_rep = torch.cat([state_rep, merge_cluster])
                q = F.relu(self.agent_fc1(merge_rep))
                q = self.agent_fc2(q)
                q_table[count] = q 

                count += 1

        q_table = nn.Softmax()(q_table)
        return q_table

class CONV_DQRN(nn.Module):
    def __init__(self, hidden_fc, hidden_size_low, hidden_size_high):
        super(CONV_DQRN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.fc = nn.Linear(1024,hidden_fc)

        self.hidden_size_low = hidden_size_low
        self.hidden_size_high = hidden_size_high
        self.hidden_low, self.hidden_high = self.init_hidden()
        self.gru_low = nn.GRU(hidden_fc, hidden_size_low, batch_first=False, bidirectional=False)
        self.gru_high = nn.GRU(hidden_size_low, hidden_size_high, batch_first=True, bidirectional=False)

        self.state_fc = nn.Linear(hidden_size_high, 16)
        self.cluster_fc = nn.Linear(hidden_size_low, 16)
        self.agent_fc1 = nn.Linear(32, 32)
        self.agent_fc2 = nn.Linear(32, 1)

    def init_hidden(self):
        return Variable(torch.zeros(1,1,self.hidden_size_low).type(FloatTensor)), Variable(torch.zeros(1,1,self.hidden_size_high).type(FloatTensor))

    def forward(self, input):
        partition, images = input
        n_images = images.size(0)
        n_cluster = len(partition)

        images = Variable(images).type(FloatTensor).view(-1,1,28,28)
        features = F.max_pool2d(F.relu(self.conv1(images)),2)
        features = F.max_pool2d(F.relu(self.conv2(features)),2)
        features = features.view(n_images,-1)
        features = self.fc(features)

        seq_list = [features[row,:] for row in partition]
        packed_seq = pack_sequence(seq_list)
        # packed_seq = prepare_sequence(partition, features)

        repeat_hidden_low = self.hidden_low.repeat(1,n_cluster,1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        cluster_rep = torch.squeeze(cluster_rep)
        state_rep = torch.squeeze(state_rep)

        cluster_rep = F.relu(self.cluster_fc(cluster_rep))
        state_rep = F.relu(self.state_fc(state_rep))
        q_table = Variable(torch.zeros(n_cluster*(n_cluster-1)/2).type(FloatTensor))

        count = 0
        for i in range(n_cluster):
            for j in range(i):
                merge_cluster = cluster_rep[i,:] + cluster_rep[j,:]
                merge_rep = torch.cat([state_rep, merge_cluster])
                q = F.relu(self.agent_fc1(merge_rep))
                q = self.agent_fc2(q)
                q_table[count] = q 

                count += 1

        q_table = nn.Softmax()(q_table)
        return q_table