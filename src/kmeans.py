#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块名称: kmeans
描述: kmeans的功能实现
作者: Lee
日期: 2024/10/1
版本: 1.0
"""
import random
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def distance(x, y):
    z = np.expand_dims(x, axis=1) - y
    z = np.square(z)
    z = np.sqrt(np.sum(z, axis=2))
    return z

class KMeansDataCluster:
    def __init__(self, k=2):
        """初始化数据和聚类数量。"""
        self.data = np.array([[1, 0], [1, 2], [1, 4], [4, 4], [4, 2], [4, 0]])
        self.K = k
        self.labels = None
        self.centroids = None

    def validate_k(self):
        if self.K > len(self.data):
            raise ValueError("聚类数量必须小于或等于数据点数量。")

    def show_plots(self):
        plt.rcParams['font.family'] = 'SimHei'
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap="viridis")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=300, alpha=0.5)
        plt.title("K-Means 聚类")
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        plt.show()

    def sklearn_kmeans(self):
        """执行 K-Means 聚类并可视化结果。"""
        self.validate_k()

        kmeans = KMeans(n_clusters=self.K, random_state=0)
        kmeans.fit(self.data)
        self.labels = kmeans.labels_  # 存储聚类标签到实例变量
        self.centroids = kmeans.cluster_centers_  # 存储聚类中心到实例变量

        print("聚类中心：")
        print(self.centroids)
        print("聚类标签：")
        print(self.labels)
        self.show_plots()
        return self.centroids, self.labels  # 返回最终的聚类中心和标签

    def numpy_kmeans(self, max_iter=30):
        """执行 K-Means 聚类，使用 NumPy 实现。"""
        self.validate_k()
        data = np.asarray(self.data, dtype=np.float32)
        n_samples, n_features = data.shape

        # 随机初始化簇中心
        indices = random.sample(range(n_samples), self.K)
        self.centroids = np.copy(data[indices])
        self.labels = np.zeros(data.shape[0], dtype=np.int32)

        for i in range(max_iter):
            dis = distance(data, self.centroids)
            self.labels = np.argmin(dis, axis=1)
            onehot = np.zeros(n_samples * self.K, dtype=np.float32)
            onehot[self.labels + np.arange(n_samples) * self.K] = 1.
            onehot = np.reshape(onehot, (n_samples, self.K))

            new_centroids = np.matmul(np.transpose(onehot, (1, 0)), data)
            new_centroids = new_centroids / np.expand_dims(np.sum(onehot, axis=0), axis=1)
            self.centroids = new_centroids

        print("聚类中心：")
        print(self.centroids)
        print("聚类标签：")
        print(self.labels)
        self.show_plots()
        return self.centroids, self.labels  # 返回最终的聚类中心和标签


class RandomKMeansDataCluster:
    def __init__(self, k=2):
        """初始化数据和聚类数量。"""
        self.data = np.array([[1, 0], [1, 2], [1, 4], [4, 4], [4, 2], [4, 0]])
        self.K = k
        self.labels = None
        self.centroids = None

    def validate_k(self):
        if self.K > len(self.data):
            raise ValueError("聚类数量必须小于或等于数据点数量。")

if __name__ == "__main__":
    kmeansData = KMeansDataCluster()
    kmeansData.sklearn_kmeans()
    kmeansData.numpy_kmeans()
