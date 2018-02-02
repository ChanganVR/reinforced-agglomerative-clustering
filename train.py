from __future__ import print_function
from __future__ import division
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
    i = int((2 * index + 0.25) ** 0.5 + 0.5)
    j = int(index - i * (i - 1) / 2)

    return env.Action(i, j)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, exp):
        if len(memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_full(self):
        return (len(self.memory)==self.capacity)

    def __len__(self):
        return len(self.memory)

# @profile
def select_action(phase='train'):
    # input = prepare_sequence(partition, images, volatile=True)
    input = [partition, images]
    n_cluster = len(partition)
    n_action = n_cluster * (n_cluster - 1) / 2

    sample = random.random()
    # eps_thresh = eps_end + (eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
    eps_thresh = eps_end + (eps_start - eps_end) * math.exp(-1. * i_episode / eps_decay)
    if (phase == 'test') or sample > eps_thresh:
        action = model(input).data.max(0)[1]
    else:
        action = LongTensor([random.randrange(n_action)])

    return action

# @profile
def optimize():
    if len(memory) < 20*batch_size:
        return

    all_replay = memory.sample(batch_size)
    for i_replay in range(batch_size):
        replay = all_replay[i_replay]
        # print len(replay)

        replay_partition = replay[0]
        replay_next_partition = replay[2]

        replay_action = Variable(replay[1])
        replay_reward = Variable(replay[3])
        replay_images = replay[4]

        # input = prepare_sequence(replay_partition, images)
        replay_input = [replay_partition, replay_images]
        q = model(replay_input)[replay_action]

        if replay_next_partition is None:
            target_q = replay_reward
        else:
            # next_input = prepare_sequence(replay_partition, images, volatile=True)
            # replay_images.volatile = True
            replay_next_input = [replay_next_partition, replay_images]
            result = model(replay_next_input).max(0)[0]
            # next_q = Variable(torch.zeros(1).type(FloatTensor), volatile=True)
            next_q = Variable(result.data, requires_grad=False)
            # next_q[:] = result[:]

            # next_q.volatile = False
            # replay_images.volatile = False
            target_q = replay_reward + gamma * next_q

        # q.register_hook(print)

        loss = F.smooth_l1_loss(q, target_q)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp(-1, 1)
        optimizer.step()

gamma = 1
eps_start = 0.95
eps_end = 0.05
# eps_start = 1
# eps_end = 1
eps_decay = 1000
batch_size = 100

n_episodes = 5000
data_dir = 'dataset'
sampling_size = 30
t_stop = 9
clustering_env = env.Env(data_dir, sampling_size, reward='global_purity')
train_max = 5000
test_max = 1

model = DQRN(sampling_size,784,32,32)
# model = CONV_DQRN(128, 32, 32)
model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
memory = ReplayMemory(10000)

steps_done = 0
train_count = 0
test_count = 0

all_test_purity = []
test_purity = [0]*test_max

for i_episode in range(n_episodes):
    phase = 'train' if (i_episode % 10 < 9) else 'test'

    if phase == 'train':
        clustering_env.set_seed(0)
        train_count += 1
    else:
        clustering_env.set_seed(0)
        test_count += 1

    partition, images, _ = clustering_env.reset()

    random.seed()
    partition = partition.cluster_assignments
    images = np.concatenate(images).reshape((sampling_size, -1))
    images = torch.from_numpy(images).type(FloatTensor)

    episode_reward = 0
    reward_list = []


    for t in count():
        action = select_action(phase)

        action_pair = pair_from_index(action[0])
        reward, next_partition, purity = clustering_env.step(action_pair)
        next_partition = next_partition.cluster_assignments

        episode_reward += reward
        reward_list.append(reward)
        reward = FloatTensor([reward])

        if t == t_stop:
            final_partition = next_partition
            next_partition = None

        if phase == 'train':
            exp = [partition, action, next_partition, reward, images]
            memory.push(exp)
            optimize()

        steps_done += 1

        if t == t_stop:
            if phase == 'test':
                test_purity[test_count%test_max] = purity
                if test_count%test_max == test_max-1:
                    avg_purity = sum(test_purity)/test_max
                    all_test_purity.append(avg_purity)
                    print('episode', i_episode, 'average test purity:', avg_purity)
            break

        partition = next_partition
