import os
import struct
import numpy as np
from collections import defaultdict
import random
from .tree import Tree
import logging

logger = logging.getLogger()


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
    def __init__(self, data_dir, sampling_size, class_num=5, dataset='mnist', phase='train', reward='local_purity'):
        if dataset == 'mnist':
            label_dict = load_mnist(phase, data_dir)
            self.label_dict = label_dict
        else:
            raise NotImplementedError
        if sampling_size % class_num != 0:
            raise ValueError("Sampling size should be a multiple of class number")
        assert phase in ['train', 'val', 'test'], 'Phase does not exist'
        assert reward in ['local_purity', 'global_purity', 'uniform']

        self.sampling_size = sampling_size
        self.class_num = class_num
        self.phase = phase
        self.sampled_features = []
        self.sampled_labels = []
        self.tree = None
        self.reward = reward

    def reset(self, seed=None, steps=0):
        """
        Define state as the combination of cluster assignments and sampled_features
        :return:
        """
        # check if steps is valid
        if steps < 0 or steps > self.sampling_size - self.class_num:
            raise ValueError('Steps argument is invalid')
        random.seed(seed)
        sampled_features = []
        sampled_labels = []
        for key in self.label_dict:
            sampled_features += random.sample(self.label_dict[key], int(self.sampling_size/self.class_num))
            sampled_labels += [key] * int(self.sampling_size/self.class_num)

        # shuffle features and labels and keep the order
        combined = list(zip(sampled_features, sampled_labels))
        random.shuffle(combined)
        sampled_features[:], sampled_labels[:] = zip(*combined)
        self.sampled_features = sampled_features
        self.sampled_labels = sampled_labels
        # print({i: label for i, label in enumerate(sampled_labels)})

        # create a new tree using sampled data
        self.tree = Tree(self.class_num, sampled_labels, self.reward)
        # do steps number correct merge
        for i in range(steps):
            self.tree.step()
        assignments = self.tree.get_assignment()
        # purity = self.tree.current_purity()
        return State(assignments), self.sampled_features

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
        if self.phase == 'train':
            return reward, State(assignments), purity
        else:
            return reward, State(assignments), purity


def load_mnist(phase, path):
    """
    Python function for importing the MNIST data set.  It returns a list tuple
    with the second element being the label and the first element
    being a numpy.uint8 2D array of pixel data for the given image.
    """
    def read_file(image_file, label_file):
        # Load everything in some numpy arrays
        with open(label_file, 'rb') as fo:
            fo.read(8)
            labels = np.fromfile(fo, dtype=np.int8)
        with open(image_file, 'rb') as fo:
            fo.read(16)
            images = np.fromfile(fo, dtype=np.uint8).reshape(len(labels), -1)
        return images, labels

    train_image_file = os.path.join(path, 'train-images-idx3-ubyte')
    train_label_file = os.path.join(path, 'train-labels-idx1-ubyte')
    test_image_file = os.path.join(path, 't10k-images-idx3-ubyte')
    test_label_file = os.path.join(path, 't10k-labels-idx1-ubyte')
    if phase == 'train':
        images, labels = read_file(train_image_file, train_label_file)
        numbers = [0, 1, 2, 3, 4]
    elif phase == 'val':
        images, labels = read_file(test_image_file, test_label_file)
        numbers = [0, 1, 2, 3, 4]
    else:
        train_images, train_labels = read_file(train_image_file, train_label_file)
        test_images, test_labels = read_file(test_image_file, test_label_file)
        images = np.concatenate([train_images, test_images])
        labels = np.concatenate([train_labels, test_labels])
        numbers = [5, 6, 7, 8, 9]
    logger.info('Numbers used in {} phase are {}'.format(phase, ' '.join([str(x) for x in numbers])))

    label_dict = defaultdict(list)
    for i in range(labels.shape[0]):
        if labels[i] in numbers:
            label_dict[labels[i]].append(images[i])

    logger.info("Number of images: {}".format(' '.join([str(len(label_dict[key])) for key in sorted(label_dict.keys())])))
    return label_dict
