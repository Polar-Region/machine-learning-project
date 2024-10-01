import unittest
from src import KMeansDataCluster

class TestSimpleData(unittest.TestCase):

    def setUp(self):
        """在每个测试之前运行。"""
        self.simple_data = KMeansDataCluster(k=2)

    def test_sklearn_cluster_centers(self):
        """测试 sklearn 聚类中心的形状和数目。"""
        self.simple_data.sklearn_kmeans()
        self.assertEqual(self.simple_data.K, 2)
        self.assertEqual(self.simple_data.data.shape[0], 6)
        self.assertEqual(self.simple_data.centroids.shape[0], self.simple_data.K)

    def test_sklearn_labels(self):
        """测试 sklearn 标签的长度与数据点数量一致。"""
        _ , labels = self.simple_data.sklearn_kmeans()  # 解包返回值
        self.assertEqual(len(labels), 6)

    def test_numpy_cluster_centers(self):
        """测试 NumPy 聚类中心的形状和数目。"""
        self.simple_data.numpy_kmeans()
        self.assertEqual(self.simple_data.K, 2)
        self.assertEqual(self.simple_data.data.shape[0], 6)
        self.assertEqual(self.simple_data.centroids.shape[0], self.simple_data.K)

    def test_numpy_labels(self):
        """测试 NumPy 标签的长度与数据点数量一致。"""
        _ , labels = self.simple_data.numpy_kmeans()  # 解包返回值
        self.assertEqual(len(labels), 6)

    def test_invalid_k(self):
        """测试当 K 大于数据点数量时是否抛出异常。"""
        with self.assertRaises(ValueError):
            KMeansDataCluster(k=10).sklearn_kmeans()  # 验证 sklearn 方法
        with self.assertRaises(ValueError):
            KMeansDataCluster(k=10).numpy_kmeans()  # 验证 NumPy 方法

if __name__ == "__main__":
    unittest.main()
