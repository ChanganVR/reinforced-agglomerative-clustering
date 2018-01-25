from collections import namedtuple

import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

FloatTensor = torch.cuda.FloatTensor

def prepare_sequence(features, partition, volatile=False):
    features = torch.from_numpy(features)
    partition = sorted(partition, key=len, reverse=True)
    seq_list = [Variable(features[row,:], volatile=volatile).type(FloatTensor) for row in partition]
    packed_seq = pack_sequence(seq_list)

    return packed_seq

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