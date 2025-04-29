"""
使用LeNet进行图像分类
Ref:
1. https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
"""

import logging, coloredlogs, datetime
import numpy as np
import torch
from torch import nn
from torch.utils import data
import d2l
import matplotlib.pyplot as plt
from IPython import display
from pathlib import Path


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale='linear',
        yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    def save_and_show(self, save_file: Path, show=True):
        self.fig.savefig(save_file, format='svg', bbox_inches='tight')

        # show
        if show:
            self.fig.canvas.flush_events()
            plt.waitforbuttonpress(timeout=0.5)


class MyTrainer:
    def __init__(self):
        self.__show_fig: bool = True  # whether to show fig
        self.__save_file: Path = (Path('./temp') / f'{Path(__file__).stem}.svg').resolve()  # save file

    def run(self) -> None:
        logging.info(f'save figure to file: {self.__save_file}')
        self.__save_file.parent.mkdir(parents=True, exist_ok=True)

        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

        # 初始化模型参数
        net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
        logging.info(f'net: {net}')
        # print output shape
        X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
        for n, layer in enumerate(net):
            X = layer(X)
            logging.info(f'[{n}] {layer.__class__.__name__}, \toutput shape: {X.shape}')

        # 训练
        lr, num_epochs = 0.9, 100
        self.__train(net, train_iter, test_iter, num_epochs, lr, device='cuda:0')

        plt.show(block=True)

    def __init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def __train(self, net: nn.Module, train_iter, test_iter, num_epochs: int, lr: float, device: str):
        """训练模型"""
        # init
        net.apply(self.__init_weights)
        logging.info(f'train on device: {device}')
        net.to(device)

        # define loss and optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = Animator(
            xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc']
        )
        timer, num_batches = d2l.Timer(), len(train_iter)

        # trains
        for epoch in range(num_epochs):
            # 训练损失总和、训练准确度总和、样本数
            metric = Accumulator(3)

            # 将模型设置为训练模式
            net.train()

            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                optimizer.step()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], self.__cal_accuracy(y_hat, y), X.shape[0])
                timer.stop()

                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]

                # plot
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))

            test_acc = self.__evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
            logging.info(
                f'epoch: {epoch + 1}/{num_epochs}, train loss: {train_loss:.5f}, train acc: {train_acc:.5f}'
                f', test acc: {test_acc:.5f}, used time: {timer.tik}'
            )

            # plot and show
            animator.save_and_show(self.__save_file, self.__show_fig)

        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    def __evaluate_accuracy_gpu(self, net, data_iter, device=None):
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device

        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BETR微调所需
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                metric.add(self.__cal_accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    def __cal_accuracy(self, y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())


if __name__ == '__main__':
    # config logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"/tmp/{Path(__file__).stem}.{datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')}.log"
            ),
        ],
    )
    coloredlogs.install(fmt="[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s")

    # run
    trainer = MyTrainer()
    trainer.run()
