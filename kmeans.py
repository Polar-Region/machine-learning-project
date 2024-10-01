from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

class SimpleData:
    def __init__(self, k=2):
        """初始化数据和聚类数量。"""
        self.data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        self.K = k
        self.labels = None
        self.centroids = None

    def sklearn(self):
        """执行 K-Means 聚类并可视化结果。"""
        if self.K > len(self.data):
            raise ValueError("聚类数量必须小于或等于数据点数量。")

        kmeans = KMeans(n_clusters=self.K, random_state=0)
        kmeans.fit(self.data)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        print("聚类中心：")
        print(self.centroids)
        print("聚类标签：")
        print(self.labels)

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap="viridis")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', s=300, alpha=0.5)
        plt.title("K-Means 聚类")
        plt.xlabel("特征 1")
        plt.ylabel("特征 2")
        plt.show()

        return self.labels, self.centroids  # 返回标签和中心

if __name__ == "__main__":
    simple = SimpleData(k=2)
    simple.sklearn()
