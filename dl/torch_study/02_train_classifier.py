"""
tutorials from pytorch.org, traing a classifier
Ref: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

The diagrams of the network could be seen in https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input x: 32x32
        x = self.pool(F.relu(self.conv1(x)))  # 6@14x14
        x = self.pool(F.relu(self.conv2(x)))  # 16@5x5
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))  # 120
        x = F.relu(self.fc2(x))  # 84
        x = self.fc3(x)  # 10
        return x


def imshow(img):
    """
    Show image
    """
    img = img / 2 + 0.5  # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # load and normalizing CIFAR10
    data_root = '~/MacDocuments/Code/DataSet/CIFAR/'  # folder to save the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # show some of the training images for fun
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(['{0:5s}'.format(classes[labels[j]]) for j in range(4)]))

    # define net
    net = Net()

    # define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network
