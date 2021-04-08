"""
tutorials from pytorch.org, the neural networks.
Ref: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

NOTE:
The network diagram in Ref is not corresponding to the code, especially the output of nn.Conv2d. The diagrams is 
corresponding to the network in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # input x: 32x32
        # max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 6@15x15
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 16@6x6
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))  # 120
        x = F.relu(self.fc2(x))  # 84
        x = self.fc3(x)  # 10
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)

    # print the learnable parameters
    params = list(net.parameters())
    print(f'paramater length: {len(params)}')
    print(f'param[0] size: {params[0].size()}')

    # network output
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(f'out = {out}')

    # loss function
    output = net(input)
    target = torch.randn(10)  # a dummy target for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f'loss = {loss}')
    # see the graph of computations
    print(f'{loss.grad_fn} ==> {loss.grad_fn.next_functions[0][0]}'
          f' ==> {loss.grad_fn.next_functions[0][0].next_functions[0][0]}')

    # backprop
    net.zero_grad()
    print(f'conv1.bias.grad before backward: {net.conv1.bias.grad}')
    loss.backward()
    print(f'conv1.bias.grad after backward: {net.conv1.bias.grad}')