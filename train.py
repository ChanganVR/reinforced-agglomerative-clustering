import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.autograd import Variable
from utils_pad import prepare_sequence
from Agent import DQRN

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

def pair_from_index(index):
    i = int((2*index+0.25)**0.5+0.5)
    j = index - i*(i-1)/2

    return i,j

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

model = DQRN()
model.cuda()

memory = ReplayMemory(10000)

def optimize():
    if len(memory) < batch_size:
        continue

    for i_replay in range(batch_size):
        exp = ReplayMemory.sample(1)

        partition = exp[0]
        next_partition = exp[2]
        action = exp[1]
        reward = exp[3]
        
        input = prepare_sequence(replay_partition, features)
        q = model(input)[action]

        if next_partition is None:
            target_q = reward
        else:
            next_input = prepare_sequence(replay_partition, features, volatile=True)
            next_q = model(next_input).max(0)[0]
            next_q.volatile = False
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
batch_size = 1

n_episodes = 1

for i_episode in range(n_episodes):
    features, partition = env.reset()

    for t in count():
        input = prepare_sequence(partition, features, volatile=True)
        n_cluster = len(partition)
        n_action = n_cluster*(n_cluster-1)/2

        sample = random.random()
        eps_thresh = eps_end + (eps_start-eps_end)*math.exp(-1.*steps_done/eps_decay)
        if sample > eps_thresh:
            action = model(input).data.max(0)[1]
        else:
            action = LongTensor([random.randrange(n_action)])

        action_pair = pair_from_index(action)
        next_partition, reward, done = env.step(action_pair)
        reward = Tensor([reward])

        exp = [partition, action, next_partition, reward]
        memory.push(exp)

        if done:
            break

        partition = next_partition





