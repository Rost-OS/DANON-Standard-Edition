import unittest
import torch
import torch.nn as nn
from danon.parallel.distributed.data.amp import (
    MixedPrecisionTraining,
    DynamicLossScaler,
    AdaptivePrecisionManager,
    EnhancedMixedPrecisionTraining
)

class TestMixedPrecisionTraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建简单模型
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def test_mixed_precision_training(self):
        amp = MixedPrecisionTraining(enabled=True)
        
        # 创建测试数据
        x = torch.randn(16, 32).to(self.device)
        y = torch.randn(16, 1).to(self.device)
        
        # 测试前向传播
        with amp.autocast():
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            
        # 测试反向传播
        scaled_loss = amp.scale_loss(loss)
        scaled_loss.backward()
        
        # 测试优化器步骤
        amp.step(self.optimizer)
        amp.update()
        
        # 检查缩放因子
        scale = amp.get_scale()
        self.assertGreater(scale, 0)
        
    def test_dynamic_loss_scaler(self):
        scaler = DynamicLossScaler()
        
        # 测试正常更新
        scaler.update_scale(has_inf_nan=False)
        self.assertGreater(scaler.get_scale(), 65536.0)
        
        # 测试溢出处理
        scaler.update_scale(has_inf_nan=True)
        self.assertLess(scaler.get_scale(), 65536.0)
        
        # 测试统计信息
        stats = scaler.get_stats()
        self.assertIn('current_scale', stats)
        self.assertIn('num_bad_steps', stats)
        self.assertIn('total_steps', stats)
        
    def test_adaptive_precision_manager(self):
        manager = AdaptivePrecisionManager()
        
        # 测试初始精度
        dtype = manager.get_dtype()
        self.assertEqual(dtype, torch.float16)
        
        # 测试精度调整
        for _ in range(100):
            manager.update_stats(
                loss=1.0,
                grad_norm=1.0,
                overflow=False
            )
            
        # 检查统计信息
        stats = manager.get_stats()
        self.assertIn('current_bits', stats)
        self.assertIn('avg_loss', stats)
        self.assertIn('avg_grad_norm', stats)
        self.assertIn('overflow_rate', stats)
        
    def test_enhanced_mixed_precision_training(self):
        amp = EnhancedMixedPrecisionTraining(enabled=True)
        
        # 创建测试数据
        x = torch.randn(16, 32).to(self.device)
        y = torch.randn(16, 1).to(self.device)
        
        # 测试训练步骤
        with amp.autocast():
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            
        scaled_loss = amp.scale_loss(loss)
        scaled_loss.backward()
        
        # 计算梯度范数
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm() for p in self.model.parameters() if p.grad is not None
            ])
        )
        
        # 执行优化器步骤
        amp.step(self.optimizer, loss, grad_norm)
        
        # 检查性能统计
        stats = amp.get_performance_stats()
        self.assertIn('precision', stats)
        self.assertIn('throughput', stats)
        self.assertIn('memory', stats)
        self.assertIn('convergence', stats)
        
if __name__ == '__main__':
    unittest.main() 