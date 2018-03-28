from __future__ import division
from __future__ import print_function
from collections import Counter
import random
import copy
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np


class Node(object):
    def __init__(self, number, data):
        """
        Each cluster is represented as a node, organized as tree structure
        :param number: No. of cluster
        :param data: For leaf node, stores None. For non-leaf node, stores a list of all leaves in its subtree
                    For root, stores its children
        """
        self.number = number
        self.data = data


class Tree(object):
    def __init__(self, class_num, labels, reward):
        self.labels = labels
        self.leaf_num = len(self.labels)
        leaf_nodes = [Node(i, None) for i in range(len(labels))]
        self.root = Node(-1, leaf_nodes)
        self.class_num = class_num
        self.counter = len(labels)
        # map the indices in shuffled assignment to indices in self.data
        self.last_assignment_dict = {}
        self.reward = reward
        # 1st, 2nd column: indices of merged cluster in ith iteration,
        # 3rd: distance, 4th: number of observation in the new cluster
        self.linkage = np.ndarray([self.leaf_num-1, 4])
        self.root_cluster_indices = list(range(self.leaf_num))
        self.steps = 0

    def is_done(self):
        return len(self.root.data) == self.class_num

    def merge(self, a, b):
        # first check if a and b is valid
        assert len(self.root_cluster_indices) == len(self.root.data)
        if a not in range(len(self.root.data)) or b not in range(len(self.root.data)):
            raise ValueError("Action does not exist")
        else:
            cluster_a = self.root.data[self.last_assignment_dict[a]]
            cluster_b = self.root.data[self.last_assignment_dict[b]]
        if self.is_done():
            raise ValueError("Episode is terminated")

        # merge a and b
        data = []
        if cluster_a.data is None:
            data.append(cluster_a.number)
        else:
            data += cluster_a.data
        if cluster_b.data is None:
            data.append(cluster_b.number)
        else:
            data += cluster_b.data
        new_cluster = Node(self.counter, data)
        self.counter += 1
        reward = self.compute_reward(cluster_a, cluster_b, new_cluster)

        # remove a,b from root and create new cluster
        self.root.data.remove(cluster_a)
        self.root.data.remove(cluster_b)
        self.root.data.append(new_cluster)

        # store this merge operation in linkage
        self.linkage[self.steps, 0] = self.root_cluster_indices[self.last_assignment_dict[a]]
        self.linkage[self.steps, 1] = self.root_cluster_indices[self.last_assignment_dict[b]]
        self.linkage[self.steps, 2] = len(data)
        self.linkage[self.steps, 3] = len(data)
        # update the indices of new cluster
        self.root_cluster_indices = [x for i, x in enumerate(self.root_cluster_indices) if i not in
                                     [self.last_assignment_dict[a], self.last_assignment_dict[b]]]
        self.root_cluster_indices.append(self.leaf_num+self.steps)
        self.steps += 1

        return reward

    def draw_dendrogram(self):
        # finish rest merging
        for i in range(self.class_num-1):
            self.linkage[self.steps, 0] = self.root_cluster_indices[0]
            self.linkage[self.steps, 1] = self.root_cluster_indices[1]
            self.linkage[self.steps, 2] = sum([len(x.data) if x.data is not None else 1 for x in self.root.data[:i+2]])
            self.linkage[self.steps, 3] = sum([len(x.data) if x.data is not None else 1 for x in self.root.data[:i+2]])
            # update the indices of new cluster
            self.root_cluster_indices = [x for i, x in enumerate(self.root_cluster_indices) if i not in [0, 1]]
            self.root_cluster_indices.append(self.leaf_num + self.steps + i)
            self.steps += 1

        plt.figure()
        hierarchy.set_link_color_palette(None)
        hierarchy.dendrogram(self.linkage, labels=self.labels)
        plt.show()

    def step(self):
        # do one-step correct merging
        assignments = self.get_assignment()
        candidate_pairs = [(i, j) for i in range(len(assignments)) for j in range(i)]
        random.shuffle(candidate_pairs)
        for (a, b) in candidate_pairs:
            cluster_a = self.root.data[self.last_assignment_dict[a]]
            cluster_b = self.root.data[self.last_assignment_dict[b]]
            labels = []
            if cluster_a.data is None:
                labels.append(self.labels[cluster_a.number])
            else:
                labels += [self.labels[leaf_num] for leaf_num in cluster_a.data]
            if cluster_b.data is None:
                labels.append(self.labels[cluster_b.number])
            else:
                labels += [self.labels[leaf_num] for leaf_num in cluster_b.data]
            # successfully find two clusters containing one label
            if len(set(labels)) == 1:
                self.merge(a, b)
                return a, b

    def compute_reward(self, cluster_a, cluster_b, new_cluster):
        """
        local_purity:   compute local purity/entropy change
        global_purity:  compute global purity change
        uniform:        +1 for successful merging, 0 for unsuccessful merging, terminate (times size factor)

        Define reward as the purity difference between before merging and after merging
        :param cluster_a:
        :param cluster_b:
        :return:
        """
        dominant_num_before = (self.dominant_num(cluster_a) + self.dominant_num(cluster_b))
        dominant_num_after = self.dominant_num(new_cluster)
        if self.reward == 'uniform':
            if dominant_num_before == dominant_num_after:
                reward = 1
            else:
                reward = 0
        else:
            if self.reward == 'local_purity':
                total_num = len(new_cluster.data)
            else:
                total_num = self.leaf_num
            # if merging is not correct, then purity drops and reward will be negative
            reward = (dominant_num_after - dominant_num_before) / total_num

        return reward

    def current_purity(self):
        dominant_nums = [self.dominant_num(cluster) for cluster in self.root.data]
        return sum(dominant_nums) / self.leaf_num

    def dominant_num(self, cluster):
        if cluster.data is None:
            return 1
        else:
            # find most common label in the cluster
            counter = Counter()
            for point in cluster.data:
                counter[self.labels[point]] += 1
            _, count = counter.most_common(1)[0]
            return count

    def get_assignment(self):
        """
        Return a random shuffled list of assignment [[cluster 1 elements], [cluster 2 elements], ...]
        """
        # the indices in assignments are the indices in self.root.data
        assignments = []
        for i in range(len(self.root.data)):
            # each child in root.data is current cluster
            if self.root.data[i].data is None:
                assignments.append([self.root.data[i].number])
            else:
                assignments.append(self.root.data[i].data)

        # shuffle assignments since the representation should be permutation invariant
        shuffled_assignments = copy.deepcopy(assignments)
        random.shuffle(shuffled_assignments)
        self.last_assignment_dict = {}
        # last assignment dict stores the original indices of new cluster order
        for i, cluster in enumerate(shuffled_assignments):
            self.last_assignment_dict[i] = assignments.index(cluster)

        return shuffled_assignments

