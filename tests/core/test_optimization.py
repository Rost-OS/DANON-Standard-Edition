"""
测试优化器和调度器功能
"""

import unittest
import torch
import torch.nn as nn
from danon.core.optimization import DynamicAdamW, DynamicLRScheduler

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def test_dynamic_adamw(self):
        """测试动态AdamW优化器"""
        optimizer = DynamicAdamW(
            self.model.parameters(),
            lr=0.001,
            adaptive_momentum=True,
            dynamic_weight_decay=True
        )
        
        # 模拟训练步骤
        x = torch.randn(32, 10).to(self.device)
        y = torch.randn(32, 1).to(self.device)
        
        for _ in range(5):
            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()
            optimizer.step()
            
        # 检查参数是否更新
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)
            
    def test_dynamic_lr_scheduler(self):
        """测试动态学习率调度器"""
        optimizer = DynamicAdamW(self.model.parameters(), lr=0.1)
        scheduler = DynamicLRScheduler(
            optimizer,
            warmup_steps=10,
            max_steps=100,
            adaptive_warmup=True,
            dynamic_decay=True
        )
        
        # 模拟训练循环
        initial_lr = optimizer.param_groups[0]['lr']
        
        # 预热阶段
        for _ in range(5):
            scheduler.step({'loss': 1.0, 'grad_norm': 1.0})
            current_lr = optimizer.param_groups[0]['lr']
            self.assertLess(current_lr, initial_lr)
            
        # 衰减阶段
        for _ in range(50):
            scheduler.step({'loss': 0.5, 'grad_norm': 0.5})
            
        final_lr = optimizer.param_groups[0]['lr']
        self.assertLess(final_lr, initial_lr)
        
    def test_scheduler_state_dict(self):
        """测试调度器状态保存和加载"""
        optimizer = DynamicAdamW(self.model.parameters(), lr=0.1)
        scheduler = DynamicLRScheduler(optimizer, warmup_steps=10)
        
        # 执行几个步骤
        for _ in range(5):
            scheduler.step()
            
        # 保存状态
        state_dict = scheduler.state_dict()
        
        # 创建新的调度器
        new_scheduler = DynamicLRScheduler(optimizer, warmup_steps=10)
        new_scheduler.load_state_dict(state_dict)
        
        # 验证状态是否正确恢复
        self.assertEqual(scheduler.last_epoch, new_scheduler.last_epoch)
        self.assertEqual(len(scheduler.lr_history), len(new_scheduler.lr_history))
        
    def test_adaptive_features(self):
        """测试自适应特性"""
        optimizer = DynamicAdamW(
            self.model.parameters(),
            lr=0.001,
            adaptive_momentum=True,
            dynamic_weight_decay=True
        )
        scheduler = DynamicLRScheduler(
            optimizer,
            warmup_steps=10,
            adaptive_warmup=True,
            dynamic_decay=True
        )
        
        # 模拟不同的训练情况
        scenarios = [
            {'loss': 1.0, 'grad_norm': 2.0},  # 高梯度范数
            {'loss': 0.5, 'grad_norm': 0.1},  # 低梯度范数
            {'loss': 0.3, 'grad_norm': 0.05}  # 非常低的梯度范数
        ]
        
        for metrics in scenarios:
            scheduler.step(metrics)
            
        # 验证学习率历史
        self.assertTrue(len(scheduler.lr_history) > 0)
        self.assertTrue(len(scheduler.grad_norm_history) > 0)

if __name__ == '__main__':
    unittest.main() 