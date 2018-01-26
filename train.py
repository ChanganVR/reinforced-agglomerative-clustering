import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
from utils_pad import prepare_sequence
from Agent import DQRN
from Agent import CONV_DQRN
from env import env
from itertools import count

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

def pair_from_index(index):
    i = int((2*index+0.25)**0.5+0.5)
    j = index - i*(i-1)/2

    return env.Action(i,j)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, exp):
        if len(memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exp
        self.position = (self.position+1)%self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action():
    # input = prepare_sequence(partition, images, volatile=True)
    input = [partition, images]
    n_cluster = len(partition)
    n_action = n_cluster*(n_cluster-1)/2

    sample = random.random()
    eps_thresh = eps_end + (eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
    if sample > eps_thresh:
        action = model(input).data.max(0)[1]
    else:
        action = LongTensor([random.randrange(n_action)])

    return action

def optimize():

    if len(memory) < batch_size:
        return

    for i_replay in range(batch_size):
        exp = ReplayMemory.sample(1)

        partition = exp[0]
        next_partition = exp[2]
        action = exp[1]
        reward = exp[3]
        images = exp[4]
        
        # input = prepare_sequence(replay_partition, images)
        input = [replay_partition, images]
        q = model(input)[action]

        if next_partition is None:
            target_q = reward
        else:
            # next_input = prepare_sequence(replay_partition, images, volatile=True)
            images.volatile = True
            next_input = [next_partition, images]
            next_q = model(next_input).max(0)[0]
            next_q.volatile = False
            images.volatile = False
            target_q = reward + gamma*next_q

        F.smooth_l1_loss(q, target_q)
        optimizer.zero_grad()
        loss.backward()
        # for param in model.parameters():
        #     param.grad.data.clamp(-1,1)
        optimizer.step()


gamma = 1
eps_start = 0.95
eps_end = 0.05
eps_decay = 2000
batch_size = 200

n_episodes = 1000
data_dir = 'dataset'
sampling_size = 50
t_stop = 30
clustering_env = env.Env(data_dir, sampling_size)

# model = DQRN(784,32,32)
model = CONV_DQRN(32,32)
model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
memory = ReplayMemory(10000)

steps_done = 0
for i_episode in range(n_episodes):
    clustering_env.set_seed(0)
    partition, images = clustering_env.reset()
    partition = partition.cluster_assignments
    images = np.concatenate(images).reshape((sampling_size,-1))
    images = torch.from_numpy(images)
    # print images.type

    episode_reward = 0
    reward_list = []
    for t in count():
        action = select_action()

        action_pair = pair_from_index(action[0])
        reward, next_partition = clustering_env.step(action_pair)
        next_partition = next_partition.cluster_assignments

        episode_reward += reward
        reward_list.append(reward)
        reward = FloatTensor([reward])

        exp = [partition, action, next_partition, reward, images]
        memory.push(exp)
        steps_done += 1

        if t == t_stop:
            next_partition = None
            print 'episode reward:', episode_reward
            # print 'reward history:', reward_list
            break

        partition = next_partition





