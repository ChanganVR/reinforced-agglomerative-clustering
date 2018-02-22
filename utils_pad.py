from collections import namedtuple

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import itertools
import numpy as np
import collections

if 1:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

def get_select(n_cluster, start):
    n_action = n_cluster*(n_cluster-1)/2
    select_i = [start]*n_action
    select_j = [start]*n_action
    count = 0
    for i in range(n_cluster):
        for j in range(i):
            select_i[count] += i
            select_j[count] += j
            count += 1

    return select_i, select_j

def prep_partition(partitions):
    cluster_batch_id = [idx for idx,p in enumerate(partitions) for x in p]
    chained_partitions = list(itertools.chain.from_iterable(partitions))

    image_batch_id = [cluster_batch_id[idx] for idx,p in enumerate(chained_partitions) for x in p]
    batch_cumsum = [0] + np.cumsum([get_partition_length(p) for p in partitions]).tolist()
    cumed_partitions = [[x+batch_cumsum[cluster_batch_id[id]] for x in cluster] for id,cluster in enumerate(chained_partitions)]

    # partition_owner indicates the owner cluster for each element
    n_images = sum([len(p) for p in chained_partitions])
    partition_owner = [-1]*n_images
    for idx,p in enumerate(cumed_partitions):
        for e in p:
            partition_owner[e] = idx

    n_partitions = len(chained_partitions)
    partition_size = [len(p) for p in chained_partitions]
    p_mat = np.zeros((n_images, n_partitions))
    r_mat = np.reciprocal(np.array(partition_size).astype(float))

    for i_p in range(n_partitions):
        p_mat[cumed_partitions[i_p],i_p] = 1

    p_mat = torch.from_numpy(p_mat).type(FloatTensor)
    r_mat = torch.from_numpy(r_mat).type(FloatTensor)

    select_i = []
    select_j = []
    action_siblings = []
    cluster_count = [len(p) for p in partitions]
    start = 0
    action_cum = 0
    for i in range(len(partitions)):
        this_i, this_j = get_select(cluster_count[i], start)
        select_i.extend(this_i)
        select_j.extend(this_j)
        n_action = cluster_count[i]*(cluster_count[i]-1)/2
        start += cluster_count[i]
        action_siblings.append(range(action_cum, action_cum+n_action))
        action_cum += n_action

    batch_size = len(partitions)
    batch_action_count = [x*(x-1)/2 for x in cluster_count]
    batch_action_cumsum = np.cumsum([0]+batch_action_count)
    a_mat = np.zeros((batch_action_cumsum[-1], batch_size))
    for i_b in range(batch_size):
        a_mat[batch_action_cumsum[i_b]:batch_action_cumsum[i_b+1],i_b] = 1

    a_mat = torch.from_numpy(a_mat).type(FloatTensor)

    train_aux = [cumed_partitions, partition_owner, select_i, select_j, action_siblings, p_mat, r_mat, a_mat]
    batch_action_cumsum = torch.from_numpy(batch_action_cumsum[:-1]).type(LongTensor)

    return train_aux, batch_action_cumsum


def get_partition_length(partition):
    return len(list(itertools.chain.from_iterable(partition)))

def sort_partition(partition):
    cluster_size = [len(cluster) for cluster in partition]
    argsort = sorted(range(len(cluster_size)), reverse=True, key=cluster_size.__getitem__)
    inversed_argsort = sorted(range(len(argsort)), key=argsort.__getitem__)
    sorted_partition = [partition[x] for x in argsort]

    return sorted_partition, argsort, inversed_argsort

def merge_partition(partition_batch):
    batch_size = len(partition_batch)
    cluster_count = [len(partition) for partition in partition_batch]
    cluster_count_cumsum = [0] + np.cumsum(cluster_count).tolist()
    partition_member = [range(cluster_count_cumsum[x], cluster_count_cumsum[x]+cluster_count[x]) for x in range(batch_size)]

    all_cluster = list(itertools.chain.from_iterable(partition_batch))
    cluster_owner = [owner_id for owner_id,partition in enumerate(partition_batch) for cluster in partition]

    all_cluster_sorted, cluster_argsort, inversed_argsort = sort_partition(all_cluster)
    cluster_owner_sorted = [cluster_owner[x] for x in cluster_argsort]
    # cluster_argsort = sorted(range(len(cluster_length)), reverse=True, key=cluster_length.__getitem__)
    # all_cluster_sorted = [all_cluster[x] for x in cluster_argsort]
    # inversed_argsort = sorted(range(len(cluster_argsort)), key=cluster_argsort.__getitem__)

    partition_cumsum = [0] + np.cumsum([get_partition_length(p) for p in partition_batch]).tolist()
    all_cluster_sorted = [[x+partition_cumsum[cluster_owner_sorted[id]] for x in cluster] for id,cluster in enumerate(all_cluster_sorted)]

    return all_cluster_sorted, inversed_argsort, partition_member


def prepare_sequence(partition, features, volatile=False):
    # features = torch.from_numpy(features)
    # partition = sorted(partition, key=len, reverse=True)

    seq_list = [Variable(features[LongTensor(row),:], volatile=volatile).type(FloatTensor) for row in partition]
    packed_seq = pack_sequence(seq_list)

    return packed_seq


# @profile
def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    ``pad_sequence`` stacks a list of Variables along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = Variable(torch.ones(25, 300))
        >>> b = Variable(torch.ones(22, 300))
        >>> c = Variable(torch.ones(15, 300))
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Variable of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Variables
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = Variable(sequences[0].data.new(*out_dims).zero_())
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
                raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


# @profile
def pack_sequence(sequences):
    r"""Packs a list of variable length Variables

    ``sequences`` should be a list of Variables of size ``Lx*``, where L is
    the length of a sequence and * is any number of trailing dimensions,
    including zero. They should be sorted in the order of decreasing length.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = Variable(torch.Tensor([1,2,3]))
        >>> b = Variable(torch.Tensor([4,5]))
        >>> c = Variable(torch.Tensor([6]))
        >>> pack_sequence([a, b, c]])
        PackedSequence(data=
         1
         4
         6
         2
         5
         3
        [torch.FloatTensor of size 6]
        , batch_sizes=[3, 2, 1])


    Arguments:
        sequences (list[Variable]): A list of sequences of decreasing length.

    Returns:
        a :class:`PackedSequence` object
    """
    return pack_padded_sequence(pad_sequence(sequences), [v.size(0) for v in sequences])

if __name__ == '__main__':
    partition_batch = [[[0,1,3],[2]],[[0,3],[1,2],[4],[5,6,7,8]],[[0,1,2,3,4,5],[6,7]]]
    print(prep_partition(partition_batch))