import unittest
import torch
import torch.nn as nn
import torch.distributed as dist
from danon.core.attention import (
    MSRAConfig,
    MSRAModel
)
from danon.core.routing import EnhancedDynamicRouter
from danon.parallel.distributed.data.loader import EnhancedDistributedDataLoader
from danon.parallel.distributed.data.amp import EnhancedMixedPrecisionTraining
from torch.utils.data import TensorDataset

class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.msra = MSRAModel(config)
        self.router = EnhancedDynamicRouter(
            dim=config.hidden_size,
            num_routes=4
        )
        self.output_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSRA处理
        msra_output = self.msra(x)[0]
        
        # 动态路由
        compute_functions = [
            nn.Linear(self.msra.config.hidden_size, self.msra.config.hidden_size).to(x.device)
            for _ in range(self.router.num_routes)
        ]
        routed_output, _ = self.router(msra_output, compute_functions)
        
        # 输出层
        return self.output_layer(routed_output)

class TestTrainingPipeline(unittest.TestCase):
    """训练流程集成测试"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = MSRAConfig(
            hidden_size=256,
            num_levels=3,
            chunk_size=64,
            num_layers=4
        )
        self.model = SimpleModel(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def create_dummy_data(self, num_samples: int = 1000):
        """创建测试数据"""
        x = torch.randn(num_samples, 32, self.config.hidden_size)
        y = torch.randn(num_samples, 32, self.config.hidden_size)
        return TensorDataset(x, y)
        
    def test_single_gpu_training(self):
        """测试单GPU训练流程"""
        # 创建数据集和加载器
        dataset = self.create_dummy_data()
        loader = EnhancedDistributedDataLoader(
            dataset=dataset,
            batch_size=16,
            num_replicas=1,
            rank=0
        )
        
        # 创建混合精度训练器
        amp = EnhancedMixedPrecisionTraining(enabled=True)
        
        # 训练循环
        self.model.train()
        for epoch in range(2):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                with amp.autocast():
                    output = self.model(data)
                    loss = nn.MSELoss()(output, target)
                    
                # 反向传播
                scaled_loss = amp.scale_loss(loss)
                scaled_loss.backward()
                
                # 计算梯度范数
                grad_norm = torch.norm(
                    torch.stack([
                        p.grad.norm()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ])
                )
                
                # 优化器步骤
                amp.step(self.optimizer, loss, grad_norm)
                self.optimizer.zero_grad()
                
                if batch_idx == 0:
                    # 检查基本训练功能
                    self.assertGreater(loss.item(), 0)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(
                        torch.isfinite(p.grad).all()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ))
                    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        # 保存模型
        state_dict = self.model.state_dict()
        torch.save(state_dict, 'test_model.pt')
        
        # 创建新模型并加载
        new_model = SimpleModel(self.config).to(self.device)
        new_model.load_state_dict(torch.load('test_model.pt'))
        
        # 比较模型参数
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
            
    def test_gradient_checkpointing(self):
        """测试梯度检查点功能"""
        # 启用梯度检查点
        self.model.msra.enable_gradient_checkpointing()
        
        # 创建测试数据
        x = torch.randn(8, 32, self.config.hidden_size).to(self.device)
        y = torch.randn(8, 32, self.config.hidden_size).to(self.device)
        
        # 前向和反向传播
        output = self.model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # 检查梯度
        self.assertTrue(all(
            p.grad is not None
            for p in self.model.parameters()
            if p.requires_grad
        ))
        
    def test_distributed_data_parallel(self):
        """测试分布式数据并行训练"""
        if not torch.cuda.is_available():
            self.skipTest("需要CUDA支持")
            
        try:
            # 初始化分布式环境
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:23456',
                world_size=1,
                rank=0
            )
            
            # 创建DistributedDataParallel模型
            model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[0]
            )
            
            # 创建数据集和加载器
            dataset = self.create_dummy_data()
            loader = EnhancedDistributedDataLoader(
                dataset=dataset,
                batch_size=16,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank()
            )
            
            # 训练一个批次
            data, target = next(iter(loader))
            data, target = data.to(self.device), target.to(self.device)
            
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # 检查梯度同步
            for p in model.parameters():
                if p.requires_grad:
                    self.assertTrue(p.grad is not None)
                    
        finally:
            # 清理分布式环境
            if dist.is_initialized():
                dist.destroy_process_group()
                
    def test_mixed_precision_stability(self):
        """测试混合精度训练的稳定性"""
        amp = EnhancedMixedPrecisionTraining(enabled=True)
        
        # 创建测试数据
        x = torch.randn(8, 32, self.config.hidden_size).to(self.device)
        y = torch.randn(8, 32, self.config.hidden_size).to(self.device)
        
        # 多次训练步骤
        losses = []
        for _ in range(10):
            with amp.autocast():
                output = self.model(x)
                loss = nn.MSELoss()(output, y)
                
            scaled_loss = amp.scale_loss(loss)
            scaled_loss.backward()
            
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.norm()
                    for p in self.model.parameters()
                    if p.grad is not None
                ])
            )
            
            amp.step(self.optimizer, loss, grad_norm)
            self.optimizer.zero_grad()
            
            losses.append(loss.item())
            
        # 检查损失的稳定性
        loss_std = torch.tensor(losses).std()
        self.assertLess(loss_std, 1.0)
        
if __name__ == '__main__':
    unittest.main() 