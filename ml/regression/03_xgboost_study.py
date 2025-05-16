"""
XGBoost学习使用
Ref:
1. https://www.cnblogs.com/pinard/p/11114748.html
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


class XGBoostStudy:
    def __init__(self):
        # 1. 生成模拟回归数据
        X, y = make_regression(n_samples=1000, n_features=10, n_targets=2, noise=0.1, random_state=42)
        # X: 特征数据, y: 目标变量

        # 2. 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 训练集用于模型训练，测试集用于模型评估

    def run(self):
        # 构建XGBoost回归模型
        print(f'simple train')
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        self.__train(model)

        # 网络搜索调参
        print(f'grid search')
        self.__search(model)

        # 使用调参后的模型重新进行训练及预测
        print(f're train and pred using best params')
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.2, max_depth=3, random_state=42)
        self.__train(model)

    def __train(self, model: xgb.XGBRegressor):
        # 4. 训练模型
        model.fit(
            self.X_train,
            self.y_train,
            verbose=True,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        )
        # 使用训练集数据拟合模型

        # 5. 预测
        y_pred = model.predict(self.X_test)
        # 用训练好的模型对测试集进行预测

        # 6. 评估模型
        mse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2_train = r2_score(self.y_train, model.predict(self.X_train))
        r2_test = r2_score(self.y_test, y_pred)
        print(f'train error, r2: {r2_train:.4f}')
        print(f"均方误差(MSE): {mse:.4f}, r2: {r2_test:.4f}")

    def __search(self, model: xgb.XGBRegressor):
        # 网格搜索调参, 第1步, 先调n_estimators和max_depth
        if False:
            grid_search = GridSearchCV(
                estimator=model, param_grid={'n_estimators': [100, 200], 'max_depth': [3, 4, 5, 6, 7, 8]}
            )
            grid_search.fit(self.X_train, self.y_train)
            print(f'best score: {grid_search.best_score_:.4f}')
            print(f'best params: {grid_search.best_params_}')
            # result
            # best score: 0.9252
            # best params: {'max_depth': 3, 'n_estimators': 200}

        # 网格搜索调参, 第2步, 再调learning_rate
        grid_search = GridSearchCV(estimator=model, param_grid={'learning_rate': [0.03, 0.05, 0.1, 0.2, 0.3]})
        grid_search.fit(self.X_train, self.y_train)
        print(f'best score: {grid_search.best_score_:.4f}')
        print(f'best params: {grid_search.best_params_}')


def basic():
    xgb_study = XGBoostStudy()
    xgb_study.run()


if __name__ == '__main__':
    basic()
