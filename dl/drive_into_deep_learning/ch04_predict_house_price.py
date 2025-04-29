"""
第4节练习, 预测房价
Ref:
    https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html#id2
"""

import logging, coloredlogs, datetime
from pathlib import Path
import hashlib, tarfile, zipfile, requests
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import d2l


class Predictor:
    def __init__(self):
        self.__data_hub: Dict = dict()
        self.__dat_url = 'http://d2l-data.s3-accelerate.amazonaws.com/'

    def run(self) -> None:
        # init data hub
        self.__data_hub['kaggle_house_train'] = (
            self.__dat_url + 'kaggle_house_pred_train.csv',
            '585e9cc93e70b39160e7921475f9bcd7d31219ce',
        )
        self.__data_hub['kaggle_house_test'] = (
            self.__dat_url + 'kaggle_house_pred_test.csv',
            'fa19780a7b011d9b009e8bff8e99922a8ee2eb90',
        )

        # 读取数据
        train_data = pd.read_csv(self.__download('kaggle_house_train'))
        test_data = pd.read_csv(self.__download('kaggle_house_test'))
        logging.info(f'train shape: {train_data.shape}, test shape: {test_data.shape}')
        logging.info(f'show data:\n{train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]}')

        # 第1列为ID, 只用来标识数据, 不参与训练
        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

        # 数据预处理
        # 数值类数据进行标准化处理, 即0均值, 单位方差
        numerical_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numerical_features] = all_features[numerical_features].apply(lambda x: (x - x.mean()) / (x.std()))
        all_features[numerical_features] = all_features[numerical_features].fillna(0)
        # 类别类数据进行独热编码处理
        all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
        logging.info(f'after preprocess, all features shape: {all_features.shape}')

        n_train = train_data.shape[0]
        train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
        test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
        train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

        # 训练
        loss = nn.MSELoss()
        in_features = train_features.shape[1]
        net = nn.Sequential(nn.Linear(in_features, 1))
        k, num_epochs, lr, weight_decay, batch_size = 5, 300, 10, 0.0, 64
        trains_loss, valid_loss = self.__k_fold(
            net, loss, k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size
        )
        logging.info(f'k-fold, train loss: {trains_loss:.5f}, valid loss: {valid_loss:.5f}')

        plt.show(block=True)

    def __download(self, name, cache_dir: Path = Path("./data")):
        """下载一个data_hub中的文件, 并返回本地文件名"""
        assert name in self.__data_hub, f"'{name}' is not in the data hub"
        url, sha1_hash = self.__data_hub[name]
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_name: Path = cache_dir / url.split('/')[-1]

        # check if the file already downloaded using sha1 hash
        if file_name.exists():
            sha1 = hashlib.sha1()
            with open(file_name, 'rb') as fs:
                while True:
                    data = fs.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return file_name

        # download the file
        logging.info(f'download "{name}" from {url} to {file_name}')
        r = requests.get(url, stream=True, verify=True)
        with open(file_name, 'wb') as fs:
            fs.write(r.content)
        return file_name

    def __download_extract(self, name: str, folder: Path = None) -> None:
        """下载并解压zip/tar文件"""
        file_name = self.__download(name)
        base_dir, ext = file_name.stem, file_name.suffix
        data_dir = file_name.with_suffix('')

        if ext == '.zip':
            fs = zipfile.ZipFile(file_name, 'r')
        elif ext in ('.tar', '.gz'):
            fs = tarfile.open(file_name, 'r')
        else:
            logging.fatal(f'invalid file extension: {ext}, file name: {file_name}')

        fs.extractall(base_dir)
        return base_dir / folder if folder else data_dir

    def __download_all(self) -> None:
        """下载所有文件"""
        for name in self.__data_hub:
            self.__download(name)

    def __log_rmse(self, net, loss, features, labels):
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()

    def __train(
        self,
        net,
        loss,
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_epochs,
        learning_rate,
        weight_decay,
        batch_size,
    ):
        train_loss, test_loss = [], []
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
        # 使用Adam优化算法
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_loss.append(self.__log_rmse(net, loss, train_features, train_labels))
            if test_labels is not None:
                test_loss.append(self.__log_rmse(net, loss, test_features, test_labels))
        return train_loss, test_loss

    def __get_k_fold_data(self, k, i, X, y):
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], dim=0)
                y_train = torch.cat([y_train, y_part], dim=0)
        return X_train, y_train, X_valid, y_valid

    def __k_fold(self, net, loss, k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        train_loss_sum, valid_loss_sum = 0, 0
        for i in range(k):
            data = self.__get_k_fold_data(k, i, X_train, y_train)
            train_loss, valid_loss = self.__train(net, loss, *data, num_epochs, learning_rate, weight_decay, batch_size)
            train_loss_sum += train_loss[-1]
            valid_loss_sum += valid_loss[-1]
            if i == 0:
                d2l.plot(
                    list(range(1, num_epochs + 1)),
                    [train_loss, valid_loss],
                    xlabel='epoch',
                    ylabel='rmse',
                    xlim=[1, num_epochs],
                    legend=['train', 'valid'],
                    yscale='log',
                )
            logging.info(f'fold {i+1}, train log rmse: {train_loss[-1]:.5f}, valid log rmse: {valid_loss[-1]:.5f}')
        return train_loss_sum / k, valid_loss_sum / k


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
    predictor = Predictor()
    predictor.run()
