import unittest
import torch
import torch.nn as nn
import time
from typing import Dict, Any
import numpy as np
from danon.core.attention import (
    MSRAConfig,
    MSRAModel,
    EnhancedSelfCalibratingAttention
)
from danon.core.routing import EnhancedDynamicRouter
from danon.parallel.distributed.data.loader import EnhancedDistributedDataLoader
from danon.parallel.distributed.data.amp import EnhancedMixedPrecisionTraining

class PerformanceBenchmark:
    """性能基准测试基类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def measure_time(self, func, *args, **kwargs) -> float:
        """测量函数执行时间"""
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start
        
    def measure_memory(self, func, *args, **kwargs) -> float:
        """测量内存使用"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            func(*args, **kwargs)
            return torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        return 0
        
    def measure_throughput(self, func, batch_size: int, *args, **kwargs) -> float:
        """测量吞吐量"""
        time_taken = self.measure_time(func, *args, **kwargs)
        return batch_size / time_taken if time_taken > 0 else 0
        
    def get_results(self) -> Dict[str, Any]:
        """获取测试结果"""
        return self.results

class AttentionBenchmark(PerformanceBenchmark):
    """注意力机制性能测试"""
    
    def __init__(self):
        super().__init__()
        self.config = MSRAConfig(
            hidden_size=256,
            num_levels=3,
            chunk_size=64
        )
        
    def test_attention_forward(self, batch_size: int = 32, seq_len: int = 512):
        """测试注意力前向传播性能"""
        model = EnhancedSelfCalibratingAttention(self.config).to(self.device)
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        def forward_pass():
            with torch.no_grad():
                model(x)
                
        time_cost = self.measure_time(forward_pass)
        memory_usage = self.measure_memory(forward_pass)
        throughput = self.measure_throughput(forward_pass, batch_size)
        
        self.results['attention_forward'] = {
            'time_ms': time_cost * 1000,
            'memory_mb': memory_usage,
            'throughput': throughput,
            'batch_size': batch_size,
            'seq_len': seq_len
        }
        
    def test_attention_backward(self, batch_size: int = 32, seq_len: int = 512):
        """测试注意力反向传播性能"""
        model = EnhancedSelfCalibratingAttention(self.config).to(self.device)
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        def backward_pass():
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            
        time_cost = self.measure_time(backward_pass)
        memory_usage = self.measure_memory(backward_pass)
        throughput = self.measure_throughput(backward_pass, batch_size)
        
        self.results['attention_backward'] = {
            'time_ms': time_cost * 1000,
            'memory_mb': memory_usage,
            'throughput': throughput,
            'batch_size': batch_size,
            'seq_len': seq_len
        }

class RoutingBenchmark(PerformanceBenchmark):
    """动态路由性能测试"""
    
    def __init__(self):
        super().__init__()
        self.dim = 256
        self.num_routes = 4
        
    def test_routing_decision(self, batch_size: int = 32, seq_len: int = 512):
        """测试路由决策性能"""
        model = EnhancedDynamicRouter(
            dim=self.dim,
            num_routes=self.num_routes
        ).to(self.device)
        
        x = torch.randn(batch_size, seq_len, self.dim).to(self.device)
        compute_functions = [
            nn.Linear(self.dim, self.dim).to(self.device)
            for _ in range(self.num_routes)
        ]
        
        def routing_pass():
            with torch.no_grad():
                model(x, compute_functions)
                
        time_cost = self.measure_time(routing_pass)
        memory_usage = self.measure_memory(routing_pass)
        throughput = self.measure_throughput(routing_pass, batch_size)
        
        self.results['routing_decision'] = {
            'time_ms': time_cost * 1000,
            'memory_mb': memory_usage,
            'throughput': throughput,
            'batch_size': batch_size,
            'seq_len': seq_len
        }

class DataLoaderBenchmark(PerformanceBenchmark):
    """数据加载性能测试"""
    
    def __init__(self):
        super().__init__()
        self.feature_dim = 256
        
    def test_data_loading(self, num_samples: int = 10000, batch_size: int = 32):
        """测试数据加载性能"""
        # 创建测试数据集
        x = torch.randn(num_samples, self.feature_dim)
        y = torch.randint(0, 2, (num_samples,))
        dataset = torch.utils.data.TensorDataset(x, y)
        
        loader = EnhancedDistributedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=1,
            rank=0
        )
        
        def load_data():
            for _ in loader:
                pass
                
        time_cost = self.measure_time(load_data)
        memory_usage = self.measure_memory(load_data)
        throughput = self.measure_throughput(load_data, num_samples)
        
        self.results['data_loading'] = {
            'time_ms': time_cost * 1000,
            'memory_mb': memory_usage,
            'throughput': throughput,
            'num_samples': num_samples,
            'batch_size': batch_size
        }

class MixedPrecisionBenchmark(PerformanceBenchmark):
    """混合精度训练性能测试"""
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def test_mixed_precision(self, batch_size: int = 32):
        """测试混合精度训练性能"""
        amp = EnhancedMixedPrecisionTraining(enabled=True)
        x = torch.randn(batch_size, 256).to(self.device)
        y = torch.randn(batch_size, 256).to(self.device)
        
        def training_step():
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
            
        time_cost = self.measure_time(training_step)
        memory_usage = self.measure_memory(training_step)
        throughput = self.measure_throughput(training_step, batch_size)
        
        self.results['mixed_precision'] = {
            'time_ms': time_cost * 1000,
            'memory_mb': memory_usage,
            'throughput': throughput,
            'batch_size': batch_size
        }

def run_benchmarks():
    """运行所有基准测试"""
    results = {}
    
    # 注意力机制测试
    attention_bench = AttentionBenchmark()
    attention_bench.test_attention_forward()
    attention_bench.test_attention_backward()
    results['attention'] = attention_bench.get_results()
    
    # 动态路由测试
    routing_bench = RoutingBenchmark()
    routing_bench.test_routing_decision()
    results['routing'] = routing_bench.get_results()
    
    # 数据加载测试
    loader_bench = DataLoaderBenchmark()
    loader_bench.test_data_loading()
    results['data_loading'] = loader_bench.get_results()
    
    # 混合精度训练测试
    amp_bench = MixedPrecisionBenchmark()
    amp_bench.test_mixed_precision()
    results['mixed_precision'] = amp_bench.get_results()
    
    return results

if __name__ == '__main__':
    results = run_benchmarks()
    
    # 打印结果
    for component, metrics in results.items():
        print(f"\n{component} Benchmark Results:")
        for test_name, values in metrics.items():
            print(f"\n{test_name}:")
            for metric, value in values.items():
                print(f"  {metric}: {value}") 