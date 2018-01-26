from collections import Counter


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
    def __init__(self, class_num, labels):
        self.labels = labels
        # print(labels)
        leaf_nodes = [Node(i, None) for i in range(len(labels))]
        self.root = Node(-1, leaf_nodes)
        self.class_num = class_num
        self.counter = len(labels)
        self.last_assignment_dict = {}

    def is_done(self):
        return len(self.root.data) == self.class_num

    def merge(self, a, b):
        # first check if a and b is valid
        if a not in range(len(self.root.data)) or b not in range(len(self.root.data)):
            raise ValueError("Action does not exist")
        else:
            cluster_a = self.root.data[self.last_assignment_dict[a]]
            cluster_b = self.root.data[self.last_assignment_dict[b]]
        # Todo: when to terminate the episode
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

        # remove a,b from root and and new cluster
        self.root.data.remove(cluster_a)
        self.root.data.remove(cluster_b)
        self.root.data.append(new_cluster)

        return reward

    def compute_reward(self, cluster_a, cluster_b, new_cluster):
        """
        Define reward as the purity difference between before merging and after merging
        :param cluster_a:
        :param cluster_b:
        :return:
        """
        # calculate purity before merging
        total_num = len(self.root.data)
        purity_before = (self.dominant_num(cluster_a) + self.dominant_num(cluster_b))/(total_num+0.0)
        purity_after = self.dominant_num(new_cluster)/(total_num+0.0)
        # if merging is not correct, then purity drops and reward will be negative
        reward = purity_after - purity_before
        return reward

    def dominant_num(self, cluster):
        if cluster.data is None:
            return 1
        else:
            # find most common label in the cluster
            counter = Counter()
            for point in cluster.data:
                counter[self.labels[point-1]] += 1
            _, count = counter.most_common(1)[0]
            return count

    def get_assignment(self):
        """
        Return a list of assignment [[cluster 1 elements], [cluster 2 elements], ...]
        the list is sorted according to the size of each cluster and each cluster is sorted wrt its element number
        """
        assignments = []
        for i in range(len(self.root.data)):
            # each child in root.data is current cluster
            if self.root.data[i].data is None:
                assignments.append([self.root.data[i].number])
            else:
                assignments.append(sorted(self.root.data[i].data))

        # sort assignments
        sorted_assignments = sorted(assignments, key=len, reverse=True)
        self.last_assignment_dict = {}
        # last assignment dict stores the original indices of new cluster order
        for i, cluster in enumerate(sorted_assignments):
            self.last_assignment_dict[i] = assignments.index(cluster)

        return sorted_assignments

