import unittest
import torch
import torch.nn as nn
import numpy as np
from danon.core.attention import (
    MSRAConfig,
    MSRAModel,
    EnhancedSelfCalibratingAttention
)
from danon.core.routing import EnhancedDynamicRouter
from danon.parallel.distributed.data.loader import EnhancedDistributedDataLoader
from danon.parallel.distributed.data.amp import EnhancedMixedPrecisionTraining

class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = MSRAConfig(
            hidden_size=256,
            num_levels=3,
            chunk_size=64,
            num_layers=4
        )
        
    def test_attention_extreme_sequences(self):
        """测试极端序列长度"""
        model = EnhancedSelfCalibratingAttention(self.config).to(self.device)
        
        # 测试最小序列
        x_min = torch.randn(1, 1, self.config.hidden_size).to(self.device)
        output_min, _ = model(x_min)
        self.assertEqual(output_min.shape, x_min.shape)
        
        # 测试超长序列
        x_long = torch.randn(1, 10000, self.config.hidden_size).to(self.device)
        output_long, _ = model(x_long)
        self.assertEqual(output_long.shape, x_long.shape)
        
        # 测试不规则序列长度
        x_irregular = torch.randn(2, 123, self.config.hidden_size).to(self.device)
        output_irregular, _ = model(x_irregular)
        self.assertEqual(output_irregular.shape, x_irregular.shape)
        
    def test_attention_numerical_stability(self):
        """测试数值稳定性"""
        model = EnhancedSelfCalibratingAttention(self.config).to(self.device)
        
        # 测试极小值
        x_tiny = torch.full((2, 32, self.config.hidden_size), 1e-10).to(self.device)
        output_tiny, _ = model(x_tiny)
        self.assertTrue(torch.isfinite(output_tiny).all())
        
        # 测试极大值
        x_huge = torch.full((2, 32, self.config.hidden_size), 1e10).to(self.device)
        output_huge, _ = model(x_huge)
        self.assertTrue(torch.isfinite(output_huge).all())
        
        # 测试NaN处理
        x_nan = torch.randn(2, 32, self.config.hidden_size).to(self.device)
        x_nan[0, 0, 0] = float('nan')
        output_nan, _ = model(x_nan)
        self.assertTrue(torch.isfinite(output_nan).all())
        
    def test_routing_edge_cases(self):
        """测试路由边界情况"""
        router = EnhancedDynamicRouter(
            dim=self.config.hidden_size,
            num_routes=4
        ).to(self.device)
        
        # 测试单一样本
        x_single = torch.randn(1, 1, self.config.hidden_size).to(self.device)
        compute_functions = [
            nn.Linear(self.config.hidden_size, self.config.hidden_size).to(self.device)
            for _ in range(router.num_routes)
        ]
        output_single, meta = router(x_single, compute_functions)
        self.assertEqual(output_single.shape, x_single.shape)
        
        # 测试不均衡路由
        x_unbalanced = torch.cat([
            torch.full((1, 32, self.config.hidden_size), -1e5),
            torch.full((1, 32, self.config.hidden_size), 1e5)
        ]).to(self.device)
        output_unbalanced, meta = router(x_unbalanced, compute_functions)
        self.assertTrue(torch.isfinite(output_unbalanced).all())
        
        # 测试梯度极值
        x_grad = torch.randn(2, 32, self.config.hidden_size).to(self.device)
        x_grad.requires_grad = True
        output_grad, _ = router(x_grad, compute_functions)
        loss = output_grad.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(x_grad.grad).all())
        
    def test_data_loader_edge_cases(self):
        """测试数据加载边界情况"""
        # 创建不规则大小的数据集
        irregular_sizes = [(1, 123), (4, 567), (2, 89)]
        dataset = []
        for batch_size, seq_len in irregular_sizes:
            x = torch.randn(batch_size, seq_len, self.config.hidden_size)
            y = torch.randn(batch_size, seq_len, self.config.hidden_size)
            dataset.extend([(x_i, y_i) for x_i, y_i in zip(x, y)])
            
        loader = EnhancedDistributedDataLoader(
            dataset=dataset,
            batch_size=2,
            num_replicas=1,
            rank=0,
            drop_last=False
        )
        
        # 验证所有批次
        for batch_idx, (data, target) in enumerate(loader):
            self.assertEqual(data.shape[-1], self.config.hidden_size)
            self.assertEqual(target.shape[-1], self.config.hidden_size)
            
    def test_mixed_precision_edge_cases(self):
        """测试混合精度训练边界情况"""
        amp = EnhancedMixedPrecisionTraining(enabled=True)
        model = MSRAModel(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 测试梯度溢出
        x = torch.full((2, 32, self.config.hidden_size), 1e4).to(self.device)
        y = torch.full((2, 32, self.config.hidden_size), 1e4).to(self.device)
        
        with amp.autocast():
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
        scaled_loss = amp.scale_loss(loss)
        scaled_loss.backward()
        
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm()
                for p in model.parameters()
                if p.grad is not None
            ])
        )
        
        # 验证梯度缩放是否正常工作
        amp.step(optimizer, loss, grad_norm)
        self.assertTrue(all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ))
        
        # 测试精度切换
        for _ in range(100):
            with amp.autocast():
                output = model(x)
                loss = nn.MSELoss()(output, y)
                
            scaled_loss = amp.scale_loss(loss)
            scaled_loss.backward()
            
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.norm()
                    for p in model.parameters()
                    if p.grad is not None
                ])
            )
            
            amp.step(optimizer, loss, grad_norm)
            optimizer.zero_grad()
            
        # 检查性能统计
        stats = amp.get_performance_stats()
        self.assertIn('precision', stats)
        self.assertIn('throughput', stats)
        self.assertIn('memory', stats)
        self.assertIn('convergence', stats)
        
    def test_memory_edge_cases(self):
        """测试内存边界情况"""
        model = MSRAModel(self.config).to(self.device)
        
        # 测试大批量处理
        try:
            x_large = torch.randn(1000, 1000, self.config.hidden_size).to(self.device)
            with torch.no_grad():
                output = model(x_large)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("内存溢出被正确捕获")
            else:
                raise e
                
        # 测试渐进式批处理
        x_large = torch.randn(1000, 1000, self.config.hidden_size)
        batch_size = 10
        outputs = []
        
        for i in range(0, len(x_large), batch_size):
            batch = x_large[i:i+batch_size].to(self.device)
            with torch.no_grad():
                output = model(batch)
            outputs.append(output.cpu())
            
        final_output = torch.cat(outputs, dim=0)
        self.assertEqual(final_output.shape[0], len(x_large))
        
    def test_gradient_edge_cases(self):
        """测试梯度边界情况"""
        model = MSRAModel(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # 测试梯度裁剪
        x = torch.randn(2, 32, self.config.hidden_size).to(self.device)
        y = torch.randn(2, 32, self.config.hidden_size).to(self.device)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # 应用极端梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
        optimizer.step()
        
        # 验证参数更新
        for p in model.parameters():
            if p.requires_grad:
                self.assertTrue(torch.isfinite(p).all())
                
        # 测试梯度累积
        optimizer.zero_grad()
        for _ in range(10):
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss = loss / 10  # 缩小梯度
            loss.backward()
            
        # 验证累积梯度
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
                
if __name__ == '__main__':
    unittest.main() 