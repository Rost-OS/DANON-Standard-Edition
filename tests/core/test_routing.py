import unittest
import torch
import torch.nn as nn
from danon.core.routing import (
    DynamicRouter,
    EnhancedDynamicRouter
)

class TestDynamicRouting(unittest.TestCase):
    def setUp(self):
        self.dim = 256
        self.num_routes = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_dynamic_router(self):
        model = DynamicRouter(
            dim=self.dim,
            num_routes=self.num_routes
        ).to(self.device)
        
        # 创建测试数据
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.dim).to(self.device)
        
        # 创建模拟计算函数
        compute_functions = [
            nn.Linear(self.dim, self.dim).to(self.device)
            for _ in range(self.num_routes)
        ]
        
        # 基本功能测试
        output, meta = model(x, compute_functions)
        self.assertEqual(output.shape, x.shape)
        self.assertIn('route_probabilities', meta)
        self.assertIn('importance_scores', meta)
        
        # 检查路由概率
        route_probs = meta['route_probabilities']
        self.assertEqual(route_probs.shape[-1], self.num_routes)
        self.assertTrue(torch.allclose(route_probs.sum(dim=-1), torch.ones_like(route_probs.sum(dim=-1))))
        
    def test_enhanced_dynamic_router(self):
        model = EnhancedDynamicRouter(
            dim=self.dim,
            num_routes=self.num_routes,
            num_experts=2
        ).to(self.device)
        
        # 创建测试数据
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.dim).to(self.device)
        
        # 创建模拟计算函数
        compute_functions = [
            nn.Linear(self.dim, self.dim).to(self.device)
            for _ in range(self.num_routes)
        ]
        
        # 基本功能测试
        output, meta = model(x, compute_functions)
        self.assertEqual(output.shape, x.shape)
        
        # 检查元信息
        self.assertIn('route_probabilities', meta)
        self.assertIn('importance_scores', meta)
        self.assertIn('routing_features', meta)
        self.assertIn('route_history', meta)
        self.assertIn('importance_history', meta)
        
        # 检查历史追踪
        self.assertEqual(meta['route_history'].shape[0], self.num_routes)
        self.assertEqual(meta['importance_history'].shape[0], self.num_routes)
        
        # 测试多次调用的一致性
        for _ in range(5):
            out1, _ = model(x, compute_functions)
            out2, _ = model(x, compute_functions)
            diff = (out1 - out2).abs().mean()
            self.assertLess(diff, 1e-6)
            
        # 测试路由统计信息
        stats = model.get_route_stats()
        self.assertIn('route_usage', stats)
        self.assertIn('route_importance', stats)
        self.assertIn('route_entropy', stats)
        
if __name__ == '__main__':
    unittest.main() 