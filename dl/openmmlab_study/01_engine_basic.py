"""
MMEngine示例

Ref:
    1.15分钟上手MMEngine:  https://mmengine.readthedocs.io/zh-cn/latest/get_started/15_minutes.html
"""

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.runner import Runner


# 1. 构建模型
class MMResNet50(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


# 2. 构建数据集和数据加载器
norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
train_dataloader = DataLoader(
    batch_size=32,
    shuffle=True,
    dataset=torchvision.datasets.CIFAR10(
        './data/cifar10',
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**norm_cfg),
            ]
        ),
    ),
)
val_dataloader = DataLoader(
    batch_size=32,
    shuffle=False,
    dataset=torchvision.datasets.CIFAR10(
        './data/cifar10',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(**norm_cfg)]),
    ),
)


# 3. 构建评测指标
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至self.results
        self.results.append({'batch_size': len(gt), 'correct': (score.argmax(dim=1) == gt).sum().cpu()})

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保所有评测指标结果的字典, 其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)


# 4. 构建执行器并执行任务
runner = Runner(
    # 用于训练和验证的模型, 需要满足特定的接口需求
    model=MMResNet50(),
    # 工作路径, 用以保存训练日志, 权重等信息
    work_dir='./output/WorkDir',
    # 训练数据加载器, 需要满足PyTorch数据加载协议
    train_dataloader=train_dataloader,
    # 优化器包装, 用于模型优化, 并提供AMP, 梯度累积等附加功能
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    # 训练配置, 用于指定训练周期, 验证间隔等信息
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    # 验证数据加载器, 需要满足PyTorch数据加载协议
    val_dataloader=val_dataloader,
    # 验证配置, 用于指定验证所需要的额外参数
    val_cfg=dict(),
    # 用于验证的评测器, 这里使用默认评测器, 并评测指标
    val_evaluator=dict(type=Accuracy),
)


# 开始训练
runner.train()
