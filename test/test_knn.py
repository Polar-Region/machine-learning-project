#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from src import KNNClassifier

class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        """初始化测试用例"""
        self.knn_numpy = KNNClassifier(use_sklearn=False, k=3, dataset='iris')
        self.knn_sklearn = KNNClassifier(use_sklearn=True, k=3, dataset='iris')
        self.X, self.y = self.knn_numpy.load_dataset()

    def test_load_dataset(self):
        """测试加载数据集"""
        X_iris, y_iris = self.knn_numpy.load_dataset()
        self.assertEqual(X_iris.shape, (150, 4))
        self.assertEqual(len(y_iris), 150)

        self.knn_numpy.dataset = 'wine'
        X_wine, y_wine = self.knn_numpy.load_dataset()
        self.assertEqual(X_wine.shape, (178, 13))
        self.assertEqual(len(y_wine), 178)

    def test_sklearn_implementation(self):
        """测试 sklearn 的 KNN 实现"""
        self.knn_sklearn.fit(self.X, self.y)
        predictions = self.knn_sklearn.predict(self.X[:10])
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_numpy_implementation(self):
        """测试 numpy 的 KNN 实现"""
        self.knn_numpy.fit(self.X, self.y)
        predictions = self.knn_numpy.predict(self.X[:10])
        self.assertEqual(predictions.shape, (10,))
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_invalid_dataset(self):
        """测试无效数据集的错误处理"""
        knn_invalid = KNNClassifier(dataset='invalid_dataset')
        with self.assertRaises(ValueError):
            knn_invalid.load_dataset()

if __name__ == "__main__":
    unittest.main()
