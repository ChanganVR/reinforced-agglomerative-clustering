from __future__ import print_function
from __future__ import division
import numpy as np
import random
import math
import time
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


if 1:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


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
    input = [[partition], Variable(images)]
    n_cluster = len(partition)
    n_action = n_cluster * (n_cluster - 1) / 2

    sample = random.random()
    # eps_thresh = eps_end + (eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
    eps_thresh = eps_end + (eps_start - eps_end) * math.exp(-1. * i_episode / eps_decay)
    if (phase == 'test') or sample > eps_thresh:
        action = model(input)[0].data.max(0)[1]
    else:
        action = LongTensor([random.randrange(n_action)])

    return action

# @profile
def optimize():
    if len(memory) < 20*batch_size:
        return

    start = time.time()
    all_replay = memory.sample(batch_size)
    for i_replay in range(batch_size):
        replay = all_replay[i_replay]

        replay_partition = replay[0]
        replay_next_partition = replay[2]

        replay_action = Variable(replay[1])
        replay_reward = Variable(replay[3])
        replay_images = replay[4]

        replay_input = [[replay_partition], Variable(replay_images)]
        q = model(replay_input)[0][replay_action]

        if replay_next_partition is None:
            target_q = replay_reward
        else:
            replay_next_input = [[replay_next_partition], Variable(replay_images)]
            result = model(replay_next_input)[0].max(0)[0]
            next_q = Variable(result.data, requires_grad=False)
            target_q = replay_reward + gamma * next_q

        loss = F.smooth_l1_loss(q, target_q)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp(-1, 1)
        optimizer.step()

    print('non batch time: ', time.time()-start)

# @profile
def optimize_batch():
    if len(memory) < 10*batch_size:
        return

    start = time.time()
    replay_batch = memory.sample(batch_size)
    replay_partition = [replay[0] for replay in replay_batch]
    replay_next_partition = [replay[2] for replay in replay_batch]

    replay_reward = Variable(torch.cat([replay[3] for replay in replay_batch]))
    replay_images = torch.cat([Variable(replay[4]) for replay in replay_batch])

    replay_input = [replay_partition, replay_images]
    replay_action = Variable(torch.cat([replay[1] for replay in replay_batch]))
    # replay_action_count = LongTensor([len(partition) for partition in replay_partition])
    # replay_action_cumsum = torch.cumsum(replay_action_count)
    # replay_action_cumsum = torch.cat([LongTensor([0]), replay_action_cumsum[:-1]])
    # replay_action = Variable([replay_action[x]+replay_action_cumsum[x] for x in range(batch_size)])

    q = torch.cat([output[replay_action[idx]] for idx,output in enumerate(model(replay_input))])

    non_final_mask = ByteTensor([replay[2] is not None for replay in replay_batch])
    non_final_images = torch.cat([Variable(replay[4], volatile=True) for replay in replay_batch if replay[2] is not None])
    non_final_next_partition = [replay[2] for replay in replay_batch if replay[2] is not None]
    non_final_input = [non_final_next_partition, non_final_images]

    next_q = Variable(torch.zeros(batch_size).type(FloatTensor))

    if not DOUBLE_Q:
        next_q[non_final_mask] = torch.cat([output.max(0)[0] for output in model(non_final_input)])
    else:
        ref_argmax = torch.cat([output.max(0)[1] for output in model(non_final_input)])
        next_q[non_final_mask] = torch.cat([output[ref_argmax[idx]] for idx,output in enumerate(model_ref(non_final_input))])

    next_q.volatile = False
    target_q = replay_reward + gamma * next_q

    loss = F.smooth_l1_loss(q, target_q)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

    if DOUBLE_Q and i_episode%update_ref == 0:
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            param_ref.data = param_ref.data*(1-update_factor) + param.data*update_factor

    # print('batch time: ', time.time()-start)


gamma = 1
eps_start = 0.95
eps_end = 0.05
eps_decay = 50000
batch_size = 80

data_dir = 'dataset'
sampling_size = 10
t_stop = 4
clustering_env = env.Env(data_dir, sampling_size, reward='global_purity')

train_max = 10000
test_max = 1
inductive = False
epoch_episode_train = 100
epoch_episode = epoch_episode_train+test_max
n_episodes = 1000000

DOUBLE_Q = False
update_ref = 1
update_factor = 0.1

hidden_low = 256
hidden_high = 256
model = DQRN(sampling_size,784,hidden_low,hidden_high)
model_ref = DQRN(sampling_size,784,hidden_low,hidden_high)
# model = CONV_DQRN(sampling_size, 1024, 32, 32)

model.load_state_dict(torch.load('/local-scratch/chenleic/cluster_models/model_500_snapshot.pt'))
model.cuda()
model_ref.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
memory = ReplayMemory(50000)

steps_done = 0
train_count = 0
test_count = 0

all_test_purity = []
test_purity = [0]*test_max
train_purity = [0]*train_max

start = time.time()
epoch_start = start
optimize_time = 0
for i_episode in range(n_episodes):
    # phase = 'train' if (i_episode%100 < 90) else 'test'
    phase = 'train' if (i_episode%epoch_episode < epoch_episode-test_max) else 'test'

    if phase == 'train':
        partition, images = clustering_env.reset(seed=train_count%train_max)
        train_count += 1
    if phase == 'test':
        partition, images = clustering_env.reset(seed=test_count%test_max+train_max*inductive)
        test_count += 1

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
            optimize_start = time.time()
            optimize_batch()
            optimize_time += time.time() - optimize_start

        steps_done += 1

        if t == t_stop:
            if phase == 'train':
                # if train_count%10 == 0:
                #     optimize_start = time.time()
                #     optimize_batch()
                #     optimize_time += time.time()-optimize_start

                train_purity[train_count%train_max] = purity
                if train_count%(epoch_episode_train) == 0:
                    train_time = time.time()-start
                    start = time.time()

            if phase == 'test':
                # print('episode', i_episode, 'purity:', purity)
                test_purity[test_count%test_max] = purity
                if test_count%test_max == 0:
                    avg_test = sum(test_purity)/test_max
                    avg_train = sum(train_purity)/train_max
                    all_test_purity.append(avg_test)
                    # print('episode', i_episode, 'average test purity:', avg_purity)
                    # print('all purity', test_purity)
                    test_time = time.time()-start
                    epoch_time = time.time()-epoch_start
                    start = time.time()
                    epoch_start = start
                    print('Episode {} average test purity: {:.4f}, average train purity: {:.4f}'.format(i_episode, avg_test, avg_train))
                    print('epoch_time: %.03f, train time: %.03f, optimize time: %.03f, test time: %.03f\n'%(epoch_time, train_time, optimize_time, test_time))
                    optimize_time = 0
            break

        partition = next_partition

    # if i_episode%10000 == 1000:
    #     print('Episode {} average train purity: {:.4f}'.format(i_episode, avg_train))
    #     torch.save(model.state_dict(), '/local-scratch/chenleic/cluster_models/model_10000.pt')

    # if len(memory) > 20*batch_size:
    #     print('Episode {} takes {:.2f}s'.format(i_episode, time.time()-start))
