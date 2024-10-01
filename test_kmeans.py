import unittest
from kmeans import SimpleData  # 替换为实际模块名

class TestSimpleData(unittest.TestCase):

    def setUp(self):
        """在每个测试之前运行。"""
        self.simple_data = SimpleData(k=2)

    def test_cluster_centers(self):
        """测试聚类中心的形状和数目。"""
        self.simple_data.sklearn()
        self.assertEqual(self.simple_data.K, 2)
        self.assertEqual(self.simple_data.data.shape[0], 6)

    def test_labels(self):
        """测试标签的长度与数据点数量一致。"""
        labels, _ = self.simple_data.sklearn()  # 解包返回值
        self.assertEqual(len(labels), 6)

    def test_invalid_k(self):
        """测试当 K 大于数据点数量时是否抛出异常。"""
        with self.assertRaises(ValueError):
            SimpleData(k=10).sklearn()

if __name__ == "__main__":
    unittest.main()
