"""
DANON测试框架
提供全面的单元测试、集成测试和性能测试功能
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import time
from typing import Dict, List, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import pytest
from danon.core.attention import MSRAModel, DALAModel
from danon.core.distributed import DistributedMSRA
from danon.core.profiler import PerformanceProfiler

class BaseTestCase(unittest.TestCase):
    """基础测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.setup_logging()
        
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        import shutil
        shutil.rmtree(cls.test_dir)
        
    @classmethod
    def setup_logging(cls):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(cls.__name__)
        
    def setUp(self):
        """每个测试用例初始化"""
        torch.manual_seed(42)
        np.random.seed(42)
        
    def assertTensorEqual(self, x: torch.Tensor, y: torch.Tensor, msg: str = None):
        """张量相等性检查"""
        self.assertTrue(
            torch.allclose(x, y, rtol=1e-5, atol=1e-8),
            msg or f"Tensors not equal: max diff = {(x - y).abs().max().item()}"
        )
        
    def assertTensorShape(self, x: torch.Tensor, shape: tuple, msg: str = None):
        """张量形状检查"""
        self.assertEqual(
            tuple(x.shape),
            shape,
            msg or f"Expected shape {shape}, got {tuple(x.shape)}"
        )
        
    def assertGradientExists(self, tensor: torch.Tensor, msg: str = None):
        """梯度存在性检查"""
        self.assertIsNotNone(
            tensor.grad,
            msg or "Gradient does not exist"
        )
        
class ModelTestMixin:
    """模型测试混入类"""
    
    def create_dummy_input(self, batch_size: int = 4, seq_len: int = 128, hidden_size: int = 768):
        """创建测试输入"""
        return torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        
    def verify_output(self, model: nn.Module, input_tensor: torch.Tensor):
        """验证模型输出"""
        output = model(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.device, input_tensor.device)
        self.assertEqual(output.dtype, input_tensor.dtype)
        return output
        
    def verify_backward(self, output: torch.Tensor, input_tensor: torch.Tensor):
        """验证反向传播"""
        loss = output.mean()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad)
        
    def verify_model_save_load(self, model: nn.Module, input_tensor: torch.Tensor):
        """验证模型保存和加载"""
        # 保存模型
        save_path = self.test_dir / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # 加载模型
        loaded_model = type(model)(*model.__init_args__, **model.__init_kwargs__)
        loaded_model.load_state_dict(torch.load(save_path))
        loaded_model.to(self.device)
        
        # 验证输出一致性
        with torch.no_grad():
            original_output = model(input_tensor)
            loaded_output = loaded_model(input_tensor)
            self.assertTensorEqual(original_output, loaded_output)
            
class PerformanceTestMixin:
    """性能测试混入类"""
    
    def measure_time(self, func, *args, **kwargs) -> float:
        """测量函数执行时间"""
        start_time = time.time()
        func(*args, **kwargs)
        return time.time() - start_time
        
    def profile_memory(self, func, *args, **kwargs) -> Dict[str, float]:
        """分析内存使用"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        torch.cuda.reset_peak_memory_stats()
        func(*args, **kwargs)
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'peak_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
            'reserved': torch.cuda.memory_reserved() / 1024**2,  # MB
            'peak_reserved': torch.cuda.max_memory_reserved() / 1024**2   # MB
        }
        
    def benchmark_throughput(self, model: nn.Module, batch_size: int, seq_len: int,
                           num_iterations: int = 100) -> Dict[str, float]:
        """基准测试吞吐量"""
        model.eval()
        input_tensor = self.create_dummy_input(batch_size, seq_len)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                model(input_tensor)
                
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                model(input_tensor)
                torch.cuda.synchronize()
                times.append(time.time() - start_time)
                
        avg_time = np.mean(times)
        throughput = batch_size * seq_len / avg_time
        
        return {
            'avg_time': avg_time,
            'throughput': throughput,
            'latency': avg_time * 1000  # ms
        }
        
class TestMSRAModel(BaseTestCase, ModelTestMixin, PerformanceTestMixin):
    """MSRA模型测试"""
    
    def setUp(self):
        super().setUp()
        self.model = MSRAModel(
            hidden_size=768,
            num_levels=3
        ).to(self.device)
        
    def test_forward(self):
        """测试前向传播"""
        input_tensor = self.create_dummy_input()
        output = self.verify_output(self.model, input_tensor)
        self.assertTensorShape(output, input_tensor.shape)
        
    def test_backward(self):
        """测试反向传播"""
        input_tensor = self.create_dummy_input()
        output = self.model(input_tensor)
        self.verify_backward(output, input_tensor)
        
    def test_save_load(self):
        """测试模型保存加载"""
        input_tensor = self.create_dummy_input()
        self.verify_model_save_load(self.model, input_tensor)
        
    def test_long_sequence(self):
        """测试长序列处理"""
        input_tensor = self.create_dummy_input(seq_len=2048)
        output = self.verify_output(self.model, input_tensor)
        self.assertTensorShape(output, input_tensor.shape)
        
    def test_attention_weights(self):
        """测试注意力权重"""
        input_tensor = self.create_dummy_input()
        attention_weights = self.model.compute_attention_weights(input_tensor)
        
        self.assertIsInstance(attention_weights, dict)
        for level, weights in attention_weights.items():
            self.assertIsInstance(weights, torch.Tensor)
            self.assertEqual(weights.dim(), 4)  # [batch, heads, seq, seq]
            
    def test_compression(self):
        """测试压缩机制"""
        input_tensor = self.create_dummy_input()
        compression_stats = self.model.get_compression_stats(input_tensor)
        
        self.assertIsInstance(compression_stats, dict)
        self.assertIn('compression_ratio', compression_stats)
        self.assertIn('information_retention', compression_stats)
        
    @pytest.mark.performance
    def test_performance(self):
        """性能测试"""
        batch_sizes = [1, 4, 16]
        seq_lengths = [128, 512, 2048]
        
        results = {}
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                key = f"b{batch_size}_s{seq_len}"
                results[key] = self.benchmark_throughput(
                    self.model, batch_size, seq_len
                )
                
        # 保存性能测试结果
        with open(self.test_dir / "performance_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    @pytest.mark.memory
    def test_memory_efficiency(self):
        """内存效率测试"""
        batch_size = 4
        seq_len = 1024
        input_tensor = self.create_dummy_input(batch_size, seq_len)
        
        memory_stats = self.profile_memory(
            lambda: self.model(input_tensor)
        )
        
        # 验证内存使用是否在合理范围内
        self.assertLess(
            memory_stats['peak_allocated'],
            1024,  # 1GB
            "Peak memory usage too high"
        )
        
class TestDALAModel(BaseTestCase, ModelTestMixin, PerformanceTestMixin):
    """DALA模型测试"""
    
    def setUp(self):
        super().setUp()
        self.model = DALAModel(
            hidden_size=768,
            max_sequence_length=8192
        ).to(self.device)
        
    def test_infinite_attention(self):
        """测试无限长度注意力"""
        # 测试不同长度的序列
        seq_lengths = [1024, 2048, 4096, 8192]
        for seq_len in seq_lengths:
            input_tensor = self.create_dummy_input(seq_len=seq_len)
            output = self.verify_output(self.model, input_tensor)
            self.assertTensorShape(output, input_tensor.shape)
            
    def test_adaptive_mechanism(self):
        """测试自适应机制"""
        input_tensor = self.create_dummy_input()
        
        # 记录初始状态
        initial_patterns = self.model.get_attention_patterns()
        
        # 运行多次前向传播
        for _ in range(5):
            self.model(input_tensor)
            
        # 验证注意力模式已适应
        adapted_patterns = self.model.get_attention_patterns()
        self.assertNotEqual(initial_patterns, adapted_patterns)
        
    def test_complexity(self):
        """测试计算复杂度"""
        seq_lengths = [128, 256, 512, 1024]
        times = []
        
        for seq_len in seq_lengths:
            input_tensor = self.create_dummy_input(seq_len=seq_len)
            time_taken = self.measure_time(
                lambda: self.model(input_tensor)
            )
            times.append(time_taken)
            
        # 验证复杂度是否接近线性
        ratios = [t2/t1 for t1, t2 in zip(times[:-1], times[1:])]
        avg_ratio = np.mean(ratios)
        self.assertLess(avg_ratio, 2.5, "Complexity might not be near-linear")
        
class TestDistributedMSRA(BaseTestCase):
    """分布式MSRA测试"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
            
        cls.world_size = min(torch.cuda.device_count(), 4)
        if cls.world_size < 2:
            raise unittest.SkipTest("Need at least 2 GPUs")
            
    def setUp(self):
        super().setUp()
        self.model = DistributedMSRA(
            world_size=self.world_size,
            backend="nccl"
        )
        
    def test_distributed_forward(self):
        """测试分布式前向传播"""
        input_tensor = torch.randn(16, 128, 768, device=self.device)
        output = self.model(input_tensor)
        self.assertTensorShape(output, input_tensor.shape)
        
    def test_gradient_sync(self):
        """测试梯度同步"""
        input_tensor = torch.randn(16, 128, 768, device=self.device)
        output = self.model(input_tensor)
        loss = output.mean()
        loss.backward()
        
        # 验证所有GPU上的梯度是否同步
        for param in self.model.parameters():
            if param.requires_grad:
                grads = [param.grad.clone() for _ in range(self.world_size)]
                for g1, g2 in zip(grads[:-1], grads[1:]):
                    self.assertTensorEqual(g1, g2)
                    
    def test_performance_scaling(self):
        """测试性能扩展性"""
        batch_size = 32
        seq_len = 256
        input_tensor = torch.randn(batch_size, seq_len, 768, device=self.device)
        
        # 测量单GPU性能
        single_gpu_time = self.measure_time(
            lambda: self.model.forward_single_gpu(input_tensor)
        )
        
        # 测量分布式性能
        distributed_time = self.measure_time(
            lambda: self.model(input_tensor)
        )
        
        # 计算扩展效率
        scaling_efficiency = (single_gpu_time / distributed_time) / self.world_size
        self.assertGreater(
            scaling_efficiency,
            0.7,  # 期望至少70%的扩展效率
            f"Poor scaling efficiency: {scaling_efficiency:.2f}"
        )
        
def run_tests():
    """运行所有测试"""
    unittest.main(verbosity=2) 