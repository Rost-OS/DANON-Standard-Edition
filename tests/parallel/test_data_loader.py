import unittest
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from danon.parallel.distributed.data.loader import (
    DistributedSampler,
    AdaptiveBatchSampler,
    EnhancedDistributedDataLoader
)

class TestDistributedDataLoader(unittest.TestCase):
    def setUp(self):
        # 创建模拟数据集
        self.num_samples = 1000
        self.feature_dim = 32
        self.x = torch.randn(self.num_samples, self.feature_dim)
        self.y = torch.randint(0, 2, (self.num_samples,))
        self.dataset = TensorDataset(self.x, self.y)
        
    def test_distributed_sampler(self):
        sampler = DistributedSampler(
            dataset=self.dataset,
            num_replicas=2,
            rank=0
        )
        
        # 检查样本数量
        self.assertEqual(len(sampler), self.num_samples // 2)
        
        # 检查索引生成
        indices = list(sampler)
        self.assertEqual(len(indices), len(sampler))
        self.assertTrue(all(0 <= idx < self.num_samples for idx in indices))
        
        # 测试不同epoch的随机性
        indices1 = list(sampler)
        sampler.set_epoch(1)
        indices2 = list(sampler)
        self.assertNotEqual(indices1, indices2)
        
    def test_adaptive_batch_sampler(self):
        sampler = AdaptiveBatchSampler(
            dataset=self.dataset,
            initial_batch_size=32,
            min_batch_size=16,
            max_batch_size=64,
            num_replicas=2,
            rank=0
        )
        
        # 基本功能测试
        batches = list(sampler)
        self.assertTrue(all(16 <= len(batch) <= 64 for batch in batches))
        
        # 测试性能指标更新
        sampler.update_metrics(
            batch_time=0.1,
            memory_used=1000,
            gradient_norm=1.0
        )
        
        # 检查批大小调整
        for _ in range(100):
            sampler.update_metrics(0.2, 2000, 2.0)
        current_batch_size = sampler.current_batch_size
        self.assertTrue(16 <= current_batch_size <= 64)
        
    def test_enhanced_distributed_data_loader(self):
        loader = EnhancedDistributedDataLoader(
            dataset=self.dataset,
            batch_size=32,
            num_replicas=2,
            rank=0
        )
        
        # 基本功能测试
        for batch_idx, (data, target) in enumerate(loader):
            self.assertTrue(16 <= len(data) <= 64)
            self.assertEqual(data.shape[1], self.feature_dim)
            
        # 测试批大小统计
        loader.update_batch_metrics(0.1, 1000, 1.0)
        stats = loader.get_batch_size_stats()
        
        self.assertIn('current_batch_size', stats)
        self.assertIn('min_batch_size', stats)
        self.assertIn('max_batch_size', stats)
        self.assertIn('avg_batch_time', stats)
        self.assertIn('avg_memory_usage', stats)
        self.assertIn('avg_gradient_norm', stats)
        
        # 测试epoch设置
        loader.set_epoch(1)
        
if __name__ == '__main__':
    unittest.main() 