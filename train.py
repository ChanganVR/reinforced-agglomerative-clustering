from __future__ import print_function
from __future__ import division
import numpy as np
import random
import math
import time
import logging
from logging import handlers
import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils_pad import prepare_sequence
from utils_pad import prep_partition
from Agent import DQRN
from Agent import CONV_DQRN
from Agent import SET_DQN
from env import env
from itertools import count
from feature_net import mnist_cnn
from time import gmtime, strftime

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

def shuffle_exp(exp):
    partition, action, next_partition = exp[:3]
    n_cluster = len(partition)
    perm = range(n_cluster)
    random.shuffle(perm)

    perm_partition = [0]*n_cluster
    for i in range(n_cluster):
        perm_partition[perm[i]] = partition[i]

    action = action[0]
    a_i = int((2 * action + 0.25) ** 0.5 + 0.5)
    a_j = int(action - a_i * (a_i - 1) / 2)
    a_j, a_i = sorted([perm[a_i], perm[a_j]])
    perm_action = LongTensor([int(a_i*(a_i-1)/2+a_j)])

    if next_partition is None:
        perm_next_partition = None
    else:
        perm_next_partition = next_partition[:]
        random.shuffle(perm_next_partition)

    perm_exp = [perm_partition, perm_action, perm_next_partition]
    perm_exp.extend(exp[3:])

    return perm_exp


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, exp):
        if len(memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, shuffle=False):
        if not shuffle:
            return random.sample(self.memory, batch_size)
        else:
            return [shuffle_exp(x) for x in random.sample(self.memory, batch_size)]

    def is_full(self):
        return len(self.memory) == self.capacity

    def __len__(self):
        return len(self.memory)


# @profile
def select_action(partition, images, phase):
    # input = prepare_sequence(partition, images, volatile=True)
    # input = [[partition], Variable(images)]
    train_aux, _ = prep_partition([partition])
    input = [Variable(images)] + train_aux
    n_cluster = len(partition)
    n_action = n_cluster * (n_cluster - 1) / 2

    sample = random.random()
    eps_thresh = eps_end + (eps_start - eps_end) * math.exp(-1. * i_episode / eps_decay)
    if (phase == 'test') or sample > eps_thresh:
        # output, _, _ = model(input)
        # action = output[0].data.max(0)[1]
        output, _ = model(input)
        action = output.data.max(0)[1]
    else:
        action = LongTensor([random.randrange(n_action)])

    return action


# @profile
def optimize():
    if len(memory) < 20 * batch_size:
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

    logger.info('non batch time: ', time.time() - start)


@profile
def optimize_batch():
    if len(memory) < start_mul * batch_size:
        return

    start = time.time()
    replay_batch = memory.sample(batch_size, shuffle=True)
    replay_partition = [replay[0] for replay in replay_batch]
    replay_next_partition = [replay[2] for replay in replay_batch]

    replay_reward = Variable(torch.cat([replay[3] for replay in replay_batch]))
    replay_images = torch.cat([Variable(replay[4]) for replay in replay_batch])

    # replay_input = [replay_partition, replay_images]
    batch_aux, batch_action_cumsum = prep_partition(replay_partition)
    replay_input = [replay_images] + batch_aux
    replay_action = torch.cat([replay[1] for replay in replay_batch])
    replay_action2 = replay_action + batch_action_cumsum

    # q_out, q_out2, _ = model(replay_input)
    # q = torch.cat([output[replay_action[idx]] for idx, output in enumerate(q_out)])
    q_out, _ = model(replay_input)
    q = q_out[replay_action2]

    non_final_mask = ByteTensor([replay[2] is not None for replay in replay_batch])
    non_final_images = torch.cat(
        [Variable(replay[4], volatile=True) for replay in replay_batch if replay[2] is not None])
    non_final_next_partition = [replay[2] for replay in replay_batch if replay[2] is not None]
    # non_final_input = [non_final_next_partition, non_final_images]
    next_train_aux, next_action_cumsum = prep_partition(non_final_next_partition)
    non_final_input = [non_final_images] + next_train_aux

    next_q = Variable(torch.zeros(batch_size).type(FloatTensor))

    if not DOUBLE_Q:
        # output, output2, output_expand = model(non_final_input)
        # next_q[non_final_mask] = torch.cat([x.max(0)[0] for x in output])
        # tmp1 = torch.cat([x.max(0)[0] for x in output])
        # tmp2 = output_expand.max(1)[0]
        # assert(torch.equal(tmp1, tmp2))

        _, next_output_expand = model(non_final_input)
        next_q[non_final_mask] = next_output_expand.max(1)[0]

    else:
        ref_argmax = torch.cat([output.max(0)[1] for output in model(non_final_input)])
        next_q[non_final_mask] = torch.cat(
            [output[ref_argmax[idx]] for idx, output in enumerate(model_ref(non_final_input))])

    next_q.volatile = False
    target_q = replay_reward + gamma * next_q

    loss = F.smooth_l1_loss(q, target_q)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

    if DOUBLE_Q and i_episode % update_ref == 0:
        for param, param_ref in zip(model.parameters(), model_ref.parameters()):
            param_ref.data = param_ref.data * (1 - update_factor) + param.data * update_factor


# @profile
def run_episode(seed, phase, current_env, print_partition=False):
    partition, images, _ = current_env.reset(phase, seed=seed)
    partition = partition.cluster_assignments
    images = np.concatenate(images).reshape((sampling_size, -1))
    images = torch.from_numpy(images).type(FloatTensor)

    # episode_reward = 0
    # reward_list = []
    random.seed()
    for t in count():
        action = select_action(partition, images, phase)
        action_pair = pair_from_index(action[0])
        reward, next_partition, purity = current_env.step(action_pair)

        next_partition = next_partition.cluster_assignments

        if print_partition:
            logger.info('step %d partition: %s' % (t + 2, next_partition))

        # episode_reward += reward
        # reward_list.append(reward)
        if t == t_stop:
            final_partition = next_partition
            next_partition = None
            max_cluster = max([len(p) for p in final_partition])

        if phase == 'train':
            reward = FloatTensor([reward])
            exp = [partition, action, next_partition, reward, images]
            memory.push(exp)
            optimize_batch()

        if t == t_stop:
            break

        partition = next_partition

    return purity, max_cluster


def test(split, current_env):
    random.seed(0)
    if split == 'train':
        test_seeds = range(test_episodes)
    else:
        test_seeds = [random.random()+train_seed_size for _ in range(test_episodes)]

    test_purity = [0] * test_episodes
    max_clusters = []
    for i_test, seed in enumerate(test_seeds):
        test_purity[i_test], max_cluster = run_episode(seed=seed, phase='test', current_env=current_env)
        max_clusters.append(max_cluster)

    avg_test = sum(test_purity) / len(test_purity)
    avg_cluster_max = sum(max_clusters) / len(max_clusters)
    logger.info('average max cluster size: {}'.format(avg_cluster_max))

    return avg_test

def log_hparams():
    logger.info('gamma: {}'.format(gamma))
    logger.info('eps_start: {}'.format(eps_start))
    logger.info('eps_end: {}'.format(eps_end))
    logger.info('eps_decay: {}'.format(eps_decay))
    logger.info('batch_size: {}'.format(batch_size))
    logger.info('start_mul: {}'.format(start_mul))
    logger.info('train_seed_size: {}'.format(train_seed_size))
    logger.info('test_episodes: {}'.format(test_episodes))
    logger.info('epoch_episode_train: {}'.format(epoch_episode_train))
    logger.info('n_episodes: {}'.format(n_episodes))
    logger.info('learning_rate: {}'.format(learning_rate))
    logger.info('sampling_size: {}'.format(sampling_size))
    logger.info('t_stop: {}'.format(t_stop))
    logger.info('memory_size: {}'.format(memory_size))


# model configuration
gamma = 1
eps_start = 0.95
eps_end = 0.05
eps_decay = 50000
batch_size = 100
start_mul = 20

# number of different training subset
train_seed_size = 10000
test_episodes = 100
epoch_episode_train = 200
n_episodes = 30000

DOUBLE_Q = False
update_ref = 1
update_factor = 0.1
learning_rate = 1e-6

sampling_size = 10
t_stop = 4
memory_size = 50000


# path configuration
data_dir = 'dataset'
if not os.path.exists('results'):
    os.mkdir('results')


log_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
log_file = 'results/{}.log'.format(log_time)


logger = logging.getLogger('')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s, %(levelname)s: %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = handlers.RotatingFileHandler(log_file, mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

log_hparams()
first_opt = math.ceil((batch_size * start_mul) / (t_stop + 1))
logger.info('first optimized in episode {}'.format(first_opt))
train_env = env.Env(data_dir, sampling_size, reward='global_purity', split='train')
val_env = env.Env(data_dir, sampling_size, reward='global_purity', split='val')
test_env = env.Env(data_dir, sampling_size, reward='global_purity', split='test')

model = SET_DQN()
model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_size)

for i_episode in range(n_episodes):

    seed = i_episode % train_seed_size
    run_episode(seed, phase='train', current_env=train_env)

    if i_episode == 0 or ((i_episode >= first_opt) and (i_episode - first_opt) % epoch_episode_train == 0):
        p_train = test(split='train', current_env=train_env)
        p_val = test(split='val', current_env=val_env)
        p_test = test(split='test', current_env=test_env)
        logger.info('Episode {} train purity: {:.4f}, val purity: {:.4f}, test purity: {:.4f}'.
                    format(i_episode, p_train, p_val, p_test))
