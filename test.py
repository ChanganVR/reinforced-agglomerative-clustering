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

from Agent import DQRN

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

# a = Variable(torch.randn(5,10)).type(FloatTensor)
# b = Variable(torch.randn(4,10)).type(FloatTensor)
# c = Variable(torch.randn(3,10)).type(FloatTensor)
# packed_seq = pack_sequence([a, b, c])

features = np.random.randn(10,20)
partition = [[1,3,5,7],[2],[0,4],[6,8],[9]]

# input = prepare_sequence(features, partition)

# model = DQRN(20,5,3)
# model.cuda()

# q = model(input)

# print q.data
# print q.data.max(0)[1]
