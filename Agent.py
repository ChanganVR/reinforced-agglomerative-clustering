import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import inspect
# from torch.nn.utils import pack_sequence
from utils_pad import pack_sequence
from utils_pad import merge_partition
# from utils_pad import pad_sequence
from utils_pad import prepare_sequence
from utils_pad import sort_partition

# from torch.nn.utils.rnn import pack_sequence

if 1:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class DQRN(nn.Module):
    def __init__(self, n_sample, input_size, hidden_size_low, hidden_size_high):
        super(DQRN, self).__init__()
        self.hidden_size_low = hidden_size_low
        self.hidden_size_high = hidden_size_high
        self.hidden_low, self.hidden_high = self.init_hidden()
        self.gru_low = nn.GRU(input_size, hidden_size_low, batch_first=False, bidirectional=False)
        self.gru_high = nn.GRU(hidden_size_low, hidden_size_high, batch_first=True, bidirectional=False)

        self.state_fc = nn.Linear(hidden_size_high, 1024)
        self.cluster_fc = nn.Linear(hidden_size_low, 1024)
        self.agent_fc1 = nn.Linear(2048, 1024)
        self.agent_fc2 = nn.Linear(1024, 1)

        n_action = int(n_sample * (n_sample - 1) / 2)
        self.row_idx = LongTensor([0] * n_action)
        self.col_idx = LongTensor([0] * n_action)
        count = 0
        for i in range(n_sample):
            for j in range(i):
                self.row_idx[count] = i
                self.col_idx[count] = j
                count += 1

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size_low).type(FloatTensor)), Variable(
            torch.zeros(1, 1, self.hidden_size_high).type(FloatTensor))

    # @profile
    def forward_old(self, input):
        partition, images = input
        partition = partition[0]
        n_images = images.size(0)
        n_cluster = len(partition)

        packed_seq = prepare_sequence(partition, images)

        repeat_hidden_low = self.hidden_low.repeat(1, n_cluster, 1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        cluster_rep = torch.squeeze(cluster_rep)
        state_rep = torch.squeeze(state_rep)

        state_rep = F.relu(self.state_fc(state_rep))
        cluster_rep = F.relu(self.cluster_fc(cluster_rep))

        n_action = int(n_cluster * (n_cluster - 1) / 2)
        row_idx = self.row_idx[:n_action]
        col_idx = self.col_idx[:n_action]
        merge_cluster = cluster_rep[row_idx, :] + cluster_rep[col_idx, :]
        tile_state = state_rep.repeat(n_action, 1)
        merge_rep = torch.cat([tile_state, merge_cluster], dim=1)
        q_table = F.relu(self.agent_fc1(merge_rep))
        q_table = self.agent_fc2(q_table)
        q_table = nn.Softmax(dim=0)(q_table)

        return q_table

    # @profile
    def forward(self, input):
        partition_batch, images_batch = input
        batch_size = len(partition_batch)
        # n_cluster = len(partition)

        all_cluster, inversed_argsort, partition_member = merge_partition(partition_batch)
        # packed_seq = prepare_sequence(all_cluster, images_batch)
        packed_seq = pack_sequence([images_batch[cluster] for cluster in all_cluster])

        n_cluster = len(inversed_argsort)
        repeat_hidden_low = self.hidden_low.repeat(1, n_cluster, 1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        cluster_rep = torch.squeeze(cluster_rep)

        cluster_rep = cluster_rep[inversed_argsort, :]
        sorted_partition_member, _, inversed_member_argsort = sort_partition(partition_member)
        # packed_cluster_rep = prepare_sequence(partition_member, cluster_rep)
        packed_cluster_rep = pack_sequence([cluster_rep[sorted_partition_member[x], :] for x in range(batch_size)])

        repeat_hidden_high = self.hidden_high.repeat(1, batch_size, 1)
        _, state_rep = self.gru_high(packed_cluster_rep, repeat_hidden_high)
        # state_rep = torch.squeeze(state_rep)
        state_rep = state_rep.view(-1, self.hidden_size_high)
        state_rep = state_rep[inversed_member_argsort]

        state_rep = F.relu(self.state_fc(state_rep))
        cluster_rep = F.relu(self.cluster_fc(cluster_rep))

        n_action = LongTensor([len(partition_batch[x]) * (len(partition_batch[x]) - 1) / 2 for x in range(batch_size)])
        n_cluster = LongTensor([len(partition_batch[x]) for x in range(batch_size)])
        if batch_size == 1:
            cluster_cumsum = LongTensor([0])
            action_cumsum = LongTensor([0])
        else:
            cluster_cumsum = torch.cumsum(n_cluster, dim=0)
            cluster_cumsum = torch.cat([LongTensor([0]), cluster_cumsum[:-1]])
            action_cumsum = torch.cumsum(n_action, dim=0)
            action_cumsum = torch.cat([LongTensor([0]), action_cumsum[:-1]])

        # row_idx = LongTensor([x + cluster_cumsum[y] for y in range(batch_size) for x in self.row_idx[:n_action[y]]])
        # col_idx = LongTensor([x + cluster_cumsum[y] for y in range(batch_size) for x in self.col_idx[:n_action[y]]])

        row_idx = torch.cat([self.row_idx[:n_action[y]] + cluster_cumsum[y] for y in range(batch_size)])
        col_idx = torch.cat([self.col_idx[:n_action[y]] + cluster_cumsum[y] for y in range(batch_size)])

        # assert(torch.equal(row_idx, row_idx2))
        # assert(torch.equal(col_idx, col_idx2))

        merge_cluster = cluster_rep[row_idx, :] + cluster_rep[col_idx, :]

        tile_state = torch.cat([state_rep[x, :].repeat(n_action[x], 1) for x in range(batch_size)])
        merge_rep = torch.cat([tile_state, merge_cluster], dim=1)
        q_table = F.relu(self.agent_fc1(merge_rep))
        q_table = self.agent_fc2(q_table)

        q_table = [nn.Softmax(dim=0)(q_table[action_cumsum[x]:action_cumsum[x] + n_action[x]]) for x in
                   range(batch_size)]
        # q_table = nn.Softmax(dim=0)(q_table)

        return q_table


class CONV_DQRN(nn.Module):
    def __init__(self, n_sample, hidden_fc, hidden_size_low, hidden_size_high):
        super(CONV_DQRN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc = nn.Linear(1024, hidden_fc)

        self.hidden_size_low = hidden_size_low
        self.hidden_size_high = hidden_size_high
        self.hidden_low, self.hidden_high = self.init_hidden()
        self.gru_low = nn.GRU(hidden_fc, hidden_size_low, batch_first=False, bidirectional=False)
        self.gru_high = nn.GRU(hidden_size_low, hidden_size_high, batch_first=True, bidirectional=False)

        self.state_fc = nn.Linear(hidden_size_high, 32)
        self.cluster_fc = nn.Linear(hidden_size_low, 32)
        self.agent_fc1 = nn.Linear(64, 32)
        self.agent_fc2 = nn.Linear(32, 1)

        n_action = n_sample * (n_sample - 1) / 2
        self.row_idx = LongTensor([0] * n_action)
        self.col_idx = LongTensor([0] * n_action)
        count = 0
        for i in range(n_sample):
            for j in range(i):
                self.row_idx[count] = i
                self.col_idx[count] = j
                count += 1

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size_low).type(FloatTensor)), Variable(
            torch.zeros(1, 1, self.hidden_size_high).type(FloatTensor))

    def forward_old(self, input):
        partition, images = input
        n_images = images.size(0)
        n_cluster = len(partition)

        images = Variable(images).type(FloatTensor).view(-1, 1, 28, 28)
        features = F.max_pool2d(F.relu(self.conv1(images)), 2)
        features = F.max_pool2d(F.relu(self.conv2(features)), 2)
        features = features.view(n_images, -1)
        features = self.fc(features)

        seq_list = [features[row, :] for row in partition]
        packed_seq = pack_sequence(seq_list)
        # packed_seq = prepare_sequence(partition, features)

        repeat_hidden_low = self.hidden_low.repeat(1, n_cluster, 1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        _, state_rep = self.gru_high(cluster_rep, self.hidden_high)

        cluster_rep = torch.squeeze(cluster_rep)
        state_rep = torch.squeeze(state_rep)

        cluster_rep = F.relu(self.cluster_fc(cluster_rep))
        state_rep = F.relu(self.state_fc(state_rep))

        # q_table = Variable(torch.zeros(n_cluster*(n_cluster-1)/2).type(FloatTensor))
        # count = 0
        # for i in range(n_cluster):
        #     for j in range(i):
        #         merge_cluster = cluster_rep[i,:] + cluster_rep[j,:]
        #         merge_rep = torch.cat([state_rep, merge_cluster])
        #         q = F.relu(self.agent_fc1(merge_rep))
        #         q = self.agent_fc2(q)
        #         q_table[count] = q
        #
        #         count += 1
        # q_table = nn.Softmax()(q_table)

        n_action = n_cluster * (n_cluster - 1) / 2
        row_idx = self.row_idx[:n_action]
        col_idx = self.col_idx[:n_action]
        merge_cluster = cluster_rep[row_idx, :] + cluster_rep[col_idx, :]
        tile_state = state_rep.repeat(n_action, 1)
        merge_rep = torch.cat([tile_state, merge_cluster], dim=1)
        q_table = F.relu(self.agent_fc1(merge_rep))
        q_table = self.agent_fc2(q_table)
        q_table = nn.Softmax(dim=0)(q_table)

        return q_table

    # @profile
    def forward(self, input):
        partition_batch, images_batch = input
        batch_size = len(partition_batch)
        n_images = images_batch.size(0)

        all_cluster, inversed_argsort, partition_member = merge_partition(partition_batch)

        features = images_batch.view(-1, 1, 28, 28)
        features = F.max_pool2d(F.relu(self.conv1(features)), 2)
        features = F.max_pool2d(F.relu(self.conv2(features)), 2)
        features = self.fc(features.view(n_images, -1))
        packed_seq = pack_sequence([features[cluster] for cluster in all_cluster])

        n_cluster = len(inversed_argsort)
        repeat_hidden_low = self.hidden_low.repeat(1, n_cluster, 1)
        _, cluster_rep = self.gru_low(packed_seq, repeat_hidden_low)
        cluster_rep = torch.squeeze(cluster_rep)

        cluster_rep = cluster_rep[inversed_argsort, :]
        sorted_partition_member, _, inversed_member_argsort = sort_partition(partition_member)
        packed_cluster_rep = pack_sequence([cluster_rep[sorted_partition_member[x], :] for x in range(batch_size)])

        repeat_hidden_high = self.hidden_high.repeat(1, batch_size, 1)
        _, state_rep = self.gru_high(packed_cluster_rep, repeat_hidden_high)
        state_rep = state_rep.view(-1, self.hidden_size_high)
        state_rep = state_rep[inversed_member_argsort]

        state_rep = F.relu(self.state_fc(state_rep))
        cluster_rep = F.relu(self.cluster_fc(cluster_rep))

        n_action = LongTensor([len(partition_batch[x]) * (len(partition_batch[x]) - 1) / 2 for x in range(batch_size)])
        n_cluster = LongTensor([len(partition_batch[x]) for x in range(batch_size)])
        if batch_size == 1:
            cluster_cumsum = LongTensor([0])
            action_cumsum = LongTensor([0])
        else:
            cluster_cumsum = torch.cumsum(n_cluster, dim=0)
            cluster_cumsum = torch.cat([LongTensor([0]), cluster_cumsum[:-1]])
            action_cumsum = torch.cumsum(n_action, dim=0)
            action_cumsum = torch.cat([LongTensor([0]), action_cumsum[:-1]])

        row_idx = torch.cat([self.row_idx[:n_action[y]] + cluster_cumsum[y] for y in range(batch_size)])
        col_idx = torch.cat([self.col_idx[:n_action[y]] + cluster_cumsum[y] for y in range(batch_size)])

        merge_cluster = cluster_rep[row_idx, :] + cluster_rep[col_idx, :]

        tile_state = torch.cat([state_rep[x, :].repeat(n_action[x], 1) for x in range(batch_size)])
        merge_rep = torch.cat([tile_state, merge_cluster], dim=1)
        q_table = F.relu(self.agent_fc1(merge_rep))
        q_table = self.agent_fc2(q_table)

        q_table = [nn.Softmax(dim=0)(q_table[action_cumsum[x]:action_cumsum[x] + n_action[x]]) for x in
                   range(batch_size)]

        return q_table


class SET_DQN(nn.Module):
    def __init__(self, external_feature=False):
        super(SET_DQN, self).__init__()
        self.external_feature = external_feature
        h_gate = 1024
        h_cluster = 1024
        h_action = 1024
        # dim_image = 784
        dim_image = 1024
        h_state = 1024

        if not self.external_feature:
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.fc_gate1 = nn.Linear(2 * dim_image, h_gate)
        # self.fc_gate1 = nn.Linear(784*2,h_gate)
        self.fc_gate2 = nn.Linear(h_gate, 1)
        # self.fc_mode1 = nn.Linear()
        # self.fc_mode2 = nn.Linear()
        self.fc_cluster1 = nn.Linear(dim_image, h_cluster)
        self.fc_cluster2 = nn.Linear(h_cluster, h_cluster)

        self.fc_gate_state1 = nn.Linear(h_cluster, h_gate)
        self.fc_gate_state2 = nn.Linear(h_gate, 1)
        self.fc_state1 = nn.Linear(h_cluster, h_state)
        self.fc_state2 = nn.Linear(h_state, h_state)

        self.fc_action1 = nn.Linear(2 * h_cluster + h_state, h_action)
        self.fc_action2 = nn.Linear(h_action, 1)

    # @profile
    def forward(self, input):

        images, select_i, select_j, action_siblings, p_mat, r_mat, a_mat, c_mat = input
        # images = Variable(images)
        # batch_size = len(action_siblings)
        # n_partition = len(partitions)

        # all_means = torch.cat([torch.mean(images[partitions[x]],dim=0).view(1,-1) for x in range(n_partition)])
        # tile_means = torch.cat([all_means[x,...].view(1,-1) for x in image_partition_ids])


        p_mat = Variable(p_mat.to_dense())
        c_mat = Variable(c_mat.to_dense())
        r_mat = Variable(r_mat.view(-1, 1))
        a_mat = Variable(a_mat)

        n_images = images.size(0)
        if not self.external_feature:
            images = images.view(-1, 1, 28, 28)
            images = F.max_pool2d(F.relu(self.conv1(images)), 2)
            images = F.max_pool2d(F.relu(self.conv2(images)), 2)
            images = images.view(n_images, -1)

        all_means = torch.mm(torch.t(p_mat), images)  # of size n_partitions*dim_feature
        all_means = torch.mul(all_means, r_mat)
        tile_means = torch.mm(p_mat, all_means)
        gate_input = torch.cat([images, tile_means], dim=1)

        gate_output_shared = F.relu(self.fc_gate1(gate_input))
        gate_output_shared = self.fc_gate2(gate_output_shared)
        # gate_output = torch.cat([nn.Softmax(dim=0)(gate_output_shared[partitions[x],...]) for x in range(n_partition)], dim=0)

        gate_output_exp = torch.exp(gate_output_shared)  # of size n_image*1
        exp_sum = torch.mm(torch.t(p_mat), gate_output_exp)  # of size n_partition*1
        tile_exp_sum = torch.mm(p_mat, exp_sum)
        gate_output = torch.div(gate_output_exp, tile_exp_sum)
        # assert(torch.equal(gate_output, gate_output2))

        cluster_input_shared = torch.mul(images, gate_output)  # of size n_image*dim_feature
        # cluster_input = torch.cat([torch.sum(cluster_input_shared[partitions[x]],dim=0).view(1,-1) for x in range(n_partition)], dim=0)
        cluster_input = torch.mm(torch.t(p_mat), cluster_input_shared)

        # if not torch.equal(cluster_input.data, cluster_input2.data):
        #     incon = torch.sum(torch.eq(cluster_input.data, cluster_input2.data))
        #     print 'incon rate', (incon+0.0)/torch.numel(cluster_input.data), torch.numel(cluster_input.data)-incon, torch.numel(cluster_input.data)

        cluster_output = F.relu(self.fc_cluster1(cluster_input))
        cluster_output = self.fc_cluster2(cluster_output)

        state_gate = F.relu(self.fc_gate_state1(cluster_output))
        state_gate = self.fc_gate_state2(state_gate)
        state_gate_exp = torch.exp(state_gate)
        state_gate_exp_sum = torch.mm(torch.t(c_mat), state_gate_exp)
        state_gate_exp_sum = torch.mm(c_mat, state_gate_exp_sum)
        state_gate = torch.div(state_gate_exp, state_gate_exp_sum)
        state = torch.mul(cluster_output, state_gate)
        state = torch.mm(torch.t(c_mat), state)
        state = F.relu(self.fc_state1(state))
        state = self.fc_state2(state)
        state = torch.mm(a_mat, state)

        cluster_pairs = torch.cat([state, cluster_output[select_i, ...], cluster_output[select_j, ...]], dim=1)
        q_table = F.relu(self.fc_action1(cluster_pairs))
        q_table = self.fc_action2(q_table)

        # q_table1 = [nn.Softmax(dim=0)(q_table[action_siblings[x]]) for x in range(batch_size)]

        q_exp = torch.exp(q_table)  # of size total_action*1
        q_exp_sum = torch.mm(torch.t(a_mat), q_exp)  # of size batch_size*1
        q_tile_sum = torch.mm(a_mat, q_exp_sum)
        q_table = torch.div(q_exp, q_tile_sum)
        q_table_expand = torch.mul(torch.t(a_mat), q_table.view(1, -1))

        return q_table, q_table_expand


if __name__ == '__main__':
    images = torch.rand(10, 784)
    partitions = [[0], [1, 2], [3, 4], [5, 6, 7, 8], [9]]
    partition_owner = [0, 1, 1, 2, 2, 3, 3, 3, 3, 4]
    select_i = [1, 2, 2, 4]
    select_j = [0, 0, 1, 3]
    action_siblings = [[0, 0, 0], [1]]

    input = [images, select_i, select_j, action_siblings]
    model = SET_DQN()
    model(input)
