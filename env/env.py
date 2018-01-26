import os
import struct
import numpy as np
from collections import defaultdict
import random
from .tree import Tree


class Action(object):
    def __init__(self, a=0, b=0, done=False):
        """
        inputs are No. of clusters to merge
        :param a:
        :param b:
        """
        self.a = a
        self.b = b
        self.done = done


class State(object):
    def __init__(self, cluster_assignments):
        """
        Only cluster assignments will be passed from environment to agent
        Features are passed when the env is reset
        :param cluster_assignments:
        """
        self.cluster_assignments = cluster_assignments

    def __str__(self):
        print(self.cluster_assignments)
        return ''


class Env(object):
    def __init__(self, data_dir, sampling_size, class_num=5, dataset='mnist', train=True, reward='local_purity'):
        if dataset == 'mnist':
            images, labels = mnist_read(train, data_dir)
            label_dict = defaultdict(list)
            for i in range(labels.shape[0]):
                label_dict[labels[i]].append(images[i])
            print("Number of classes in mnist:", len(label_dict))
            for key in label_dict:
                print("Number of images in digit ", key, " is ", len(label_dict[key]))
            self.label_dict = label_dict
        else:
            raise ValueError("dataset does not exist")
        if sampling_size % class_num != 0:
            raise ValueError("Sampling size should be a multiple of class number")
        self.sampling_size = sampling_size
        self.class_num = class_num
        self.train = train
        self.sampled_features = []
        self.sampled_labels = []
        self.tree = None
        assert reward in ['local_purity', 'global_purity', 'uniform']
        self.reward = reward

    def set_seed(self, seed):
        random.seed(seed)

    def reset(self):
        """
        Define state as the combination of cluster assignments and sampled_features
        :return:
        """
        sampled_features = []
        sampled_labels = []
        sampled_classes = random.sample(self.label_dict.keys(), self.class_num)
        for key in sampled_classes:
            sampled_features += random.sample(self.label_dict[key], int(self.sampling_size/self.class_num))
            sampled_labels += [key] * int(self.sampling_size/self.class_num)

        # shuffle features and labels and keep the order
        combined = list(zip(sampled_features, sampled_labels))
        random.shuffle(combined)
        sampled_features[:], sampled_labels[:] = zip(*combined)
        self.sampled_features = sampled_features
        self.sampled_labels = sampled_labels

        # create a new tree using sampled data
        self.tree = Tree(self.class_num, sampled_labels, self.reward)
        assignments = self.tree.get_assignment()
        purity = self.tree.current_purity()
        return State(assignments), self.sampled_features, purity

    def step(self, action):
        """
        Take action as input and returns reward, state pair
        :param action:
        :return:
        """
        if not isinstance(action, Action):
            raise ValueError("Input is not an instance of class Action")
        if action.done:
            return None, None
        reward = self.tree.merge(action.a, action.b)
        assignments = self.tree.get_assignment()
        purity = self.tree.current_purity()
        if self.train:
            return reward, State(assignments), purity
        else:
            return State(assignments), purity


def mnist_read(train, path):
    """
    Python function for importing the MNIST data set.  It returns a list tuple
    with the second element being the label and the first element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if train:
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), -1)

    return images, labels