from torchvision import models, transforms
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class LeNet(nn.Module):
    def __init__(self, num_classes):
        """
        Input size: 1x32x32
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
        return out


def load_mnist(path):
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
    train_images, train_labels = read_file(train_image_file, train_label_file)
    test_images, test_labels = read_file(test_image_file, test_label_file)
    images = np.concatenate([train_images, test_images])
    labels = np.concatenate([train_labels, test_labels])

    return images, labels


class Mnist(Dataset):
    def __init__(self, images, labels, phase):
        if phase == 'train':
            self.numbers = [0, 1, 2, 3, 4, 5, 6]
        else:
            self.numbers = [7, 8, 9]
        indices = np.isin(labels, self.numbers)
        self.images = images[indices]
        self.labels = labels[indices]
        self.images = np.reshape(self.images, [-1, 28, 28, 1])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),
                                             transforms.Resize((32, 32)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(self.images[item]), torch.LongTensor([int(self.labels[item]-min(self.numbers))])


def train(model, data_loaders, dataset_sizes, criterion, optimizer, num_epochs):
    model.train(True)
    for i in range(num_epochs):
        running_loss = 0
        running_corrects = 0
        for data in data_loaders['train']:
            images, labels = data
            labels = torch.squeeze(labels)
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = model(images)
            _, predicts = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0] * images.size(0)
            running_corrects += torch.sum(predicts.data == labels.data)
        loss = running_loss / dataset_sizes['train']
        acc = running_corrects / dataset_sizes['train']
        print('In epoch {}, loss is {:.2f}, accuracy is {:.4f}'.format(i, loss, acc))


def test(model, data_loaders, split, raw_pixel=False):
    # extract test features
    if split == 'train':
        model.train(True)
    else:
        model.train(False)
    label_list = []
    feature_list = []
    for data in data_loaders[split]:
        images, labels = data
        label_list.append(torch.squeeze(labels).numpy())
        # feature size: 16*5*5
        if raw_pixel:
            feature_list.append(np.reshape(images.numpy(), (images.shape[0], -1)))
        else:
            feature_list.append(model(Variable(images.cuda())).data.cpu().numpy())
    test_labels = np.concatenate(label_list)
    test_features = np.concatenate(feature_list)

    # kmeans
    if split == 'train':
        clustering = KMeans(n_clusters=7, random_state=0).fit(test_features)
    else:
        clustering = KMeans(n_clusters=3, random_state=0).fit(test_features)
    purity = purity_score(test_labels, clustering.labels_)
    print('Purity score in split {}: {}'.format(split, purity))


def purity_score(y_true, y_pred):
    # matrix which will hold the majority-voted labels
    y_labeled_voted = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurrence of classes between two consecutive bin
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_labeled_voted[y_pred == cluster] = winner

    return accuracy_score(y_true, y_labeled_voted)


def main():
    num_epochs = 5

    images, labels = load_mnist('dataset')
    datasets = {phase: Mnist(images, labels, phase) for phase in ['train', 'test']}
    dataset_sizes = {phase: len(Mnist(images, labels, phase)) for phase in ['train', 'test']}
    data_loaders = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=100, shuffle=True,)
                    for phase in ['train', 'test']}
    model = LeNet(num_classes=7)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    test(model, data_loaders, split='train', raw_pixel=False)
    test(model, data_loaders, split='test', raw_pixel=False)
    train(model, data_loaders, dataset_sizes, criterion, optimizer, num_epochs)
    test(model, data_loaders, split='train', raw_pixel=False)
    test(model, data_loaders, split='test', raw_pixel=False)


if __name__ == '__main__':
    main()
