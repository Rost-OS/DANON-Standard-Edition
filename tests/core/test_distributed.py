"""
测试分布式训练功能
"""

import unittest
import torch
import torch.nn as nn
import torch.distributed as dist
from danon.core.distributed import (
    DistributedManager,
    GradientAccumulator,
    MixedPrecisionTrainer
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.train_batch_size = 32
        
    def forward(self, x):
        return self.linear(x)

class TestDistributed(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_distributed_manager(self):
        """测试分布式管理器"""
        manager = DistributedManager(
            backend='gloo',  # 使用gloo后端进行CPU测试
            world_size=1,
            rank=0
        )
        
        # 初始化
        manager.setup()
        self.assertTrue(manager.initialized)
        
        # 准备模型
        model = manager.prepare_model(self.model)
        self.assertIsInstance(model, nn.Module)
        
        # 准备优化器
        optimizer = torch.optim.Adam(model.parameters())
        optimizer = manager.prepare_optimizer(optimizer)
        
        # 测试梯度同步
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        manager.sync_gradients()
        
        # 清理
        manager.cleanup()
        self.assertFalse(manager.initialized)
        
    def test_gradient_accumulator(self):
        """测试梯度累积器"""
        accumulator = GradientAccumulator(
            self.model,
            accumulation_steps=4,
            clip_grad_norm=1.0,
            dynamic_accumulation=True
        )
        
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # 模拟训练步骤
        for i in range(8):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            
            accumulator.backward(loss)
            
            if (i + 1) % 4 == 0:
                avg_loss = accumulator.step(optimizer)
                self.assertIsNotNone(avg_loss)
            else:
                avg_loss = accumulator.step(optimizer)
                self.assertIsNone(avg_loss)
                
        # 检查有效批量大小
        self.assertEqual(
            accumulator.get_effective_batch_size(),
            self.model.train_batch_size * accumulator.accumulation_steps
        )
        
    @unittest.skipIf(not torch.cuda.is_available(), "需要CUDA支持")
    def test_mixed_precision_trainer(self):
        """测试混合精度训练器"""
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = MixedPrecisionTrainer(
            model,
            optimizer,
            dynamic_loss_scale=True
        )
        
        # 模拟训练步骤
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        for _ in range(5):
            output = model(x)
            loss = nn.MSELoss()(output, y)
            success = trainer.step(loss)
            self.assertTrue(success)
            
        # 检查状态
        self.assertGreater(trainer.get_scale(), 0)
        self.assertEqual(trainer.get_overflow_rate(), 0)
        
        # 测试状态保存和加载
        state = trainer.state_dict()
        new_trainer = MixedPrecisionTrainer(model, optimizer)
        new_trainer.load_state_dict(state)
        
        self.assertEqual(trainer.get_scale(), new_trainer.get_scale())
        self.assertEqual(trainer.get_overflow_rate(), new_trainer.get_overflow_rate())
        
    def test_combined_features(self):
        """测试特性组合"""
        # 创建分布式环境
        manager = DistributedManager(backend='gloo', world_size=1, rank=0)
        manager.setup()
        
        # 准备模型和优化器
        model = manager.prepare_model(self.model)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer = manager.prepare_optimizer(optimizer)
        
        # 设置梯度累积
        accumulator = GradientAccumulator(
            model,
            accumulation_steps=2,
            dynamic_accumulation=True
        )
        
        # 如果可用，添加混合精度训练
        if torch.cuda.is_available():
            trainer = MixedPrecisionTrainer(model, optimizer)
        else:
            trainer = None
            
        # 模拟训练循环
        for i in range(4):
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            if trainer is not None:
                trainer.step(loss)
            else:
                accumulator.backward(loss)
                if (i + 1) % 2 == 0:
                    accumulator.step(optimizer)
                    
            # 同步梯度
            manager.sync_gradients()
            
        # 清理
        manager.cleanup()

if __name__ == '__main__':
    unittest.main() 