#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块名称: knn
描述: knn的功能实现
作者: Lee
日期: 2024/10/5
版本: 1.0
"""

import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, use_sklearn =True, k=3):
        self.use_sklearn = use_sklearn
        self.k = k
        if use_sklearn:
            self.model = KNeighborsClassifier(n_neighbors=k)

    def load_dataset(self, name):
        """加载数据集"""
        if name == 'iris':
            data = load_iris()
        elif name == 'wine':
            data = load_wine()
        else:
            raise ValueError("Unsupported dataset. Choose 'iris' or 'wine'.")
        return data['data'], data['target']

    def fit(self, X, y):
        """根据选择使用 sklearn 或 numpy 实现 KNN"""
        if self.use_sklearn:
            self.model.fit(X, y)
        else:
            self.X_train = X
            self.y_train = y

    def predict(self, X):
        """根据选择使用 sklearn 或 numpy 进行预测"""
        if self.use_sklearn:
            return self.model.predict(X)
        else:
            return self.knn_numpy(self.X_train, self.y_train, X, self.k)

    def knn_numpy(self, X_train, y_train, X_test, k):
        """手动实现的 KNN 使用 numpy"""
        distances = np.sqrt(((X_train - X_test[:, np.newaxis]) ** 2).sum(axis=2))
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        top_k_labels = y_train[nearest_neighbors]
        return mode(top_k_labels, axis=1)[0]


if __name__ == "__main__":
    # 示例使用
    knn = KNNClassifier(use_sklearn=False)  # 设置为 numpy 实现
    X, y = knn.load_dataset('iris')
    knn.fit(X, y)
    predictions = knn.predict(X[:10])
    print(predictions)