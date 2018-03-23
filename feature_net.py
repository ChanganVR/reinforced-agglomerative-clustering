from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from env import env

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor

class mnist_cnn(nn.Module):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)
        self.fc1 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,10)

    def forward(self, input):
        # input = input.view(-1,1,28,28)
        x = F.max_pool2d(F.relu(self.conv1(input.view(-1,1,28,28))),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x1 = F.relu(self.fc1(x.view(-1,1024)))
        x2 = self.fc2(x1)

        return x1, x2

class mnist_vae(nn.Module):
    def __init__(self):
        super(mnist_vae, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=5)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5)

        self.dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
        self.fc_mu = nn.Linear(1024,128)
        self.fc_sigma = nn.Linear(1024,128)
        self.fc_decoder1 = nn.Linear(128,512)
        self.fc_decoder2 = nn.Linear(512,784)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(input.view(-1,1,28,28))),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2).view(-1,1024)
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        sigma = torch.exp(log_sigma)
        r =
if __name__ == '__main__':
    data_dir = 'dataset'
    train_images, train_labels = env.load_mnist(True, data_dir)
    test_images, test_labels = env.load_mnist(False, data_dir)

    train_images = FloatTensor(train_images)
    train_labels = LongTensor(train_labels.astype(int))
    test_images = Variable(FloatTensor(test_images), volatile=True)
    test_labels = LongTensor(test_labels.astype(int))

    n_train = train_images.size(0)
    n_test = test_images.size(0)

    model = mnist_cnn()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 200
    for i_step in range(200):
        start = (i_step*batch_size)%n_train
        end = ((i_step+1)*batch_size)%n_train
        image_batch = Variable(train_images[start:end,...])
        label_batch = Variable(train_labels[start:end,...])

        _, pred = model(image_batch)
        loss = F.cross_entropy(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), '/local-scratch/chenleic/cluster_models/mnist_model.pt')

