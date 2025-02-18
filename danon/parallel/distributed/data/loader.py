"""
分布式数据加载器
实现了高效的数据分片和并行加载机制
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Optional, Any, Iterator, List, TypeVar, Generic, Sized, Dict
import math

T_co = TypeVar('T_co', covariant=True)

class DistributedSampler(Sampler[T_co]):
    """分布式采样器，负责数据分片"""
    
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("分布式包不可用，请检查安装")
            num_replicas = dist.get_world_size()
            
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("分布式包不可用，请检查安装")
            rank = dist.get_rank()
            
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"无效的rank {rank}, rank应该在0到{num_replicas-1}之间")
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        if isinstance(dataset, Sized):
            self.num_samples = self._get_num_samples()
        else:
            raise ValueError("数据集必须实现__len__方法")
            
    def _get_num_samples(self) -> int:
        """计算当前进程应该处理的样本数"""
        total_size = len(self.dataset)
        if self.drop_last:
            return math.ceil(
                (total_size - self.rank - 1) / self.num_replicas
            )
        else:
            return math.ceil(total_size / self.num_replicas)
            
    def __iter__(self) -> Iterator[T_co]:
        """返回数据索引迭代器"""
        if self.shuffle:
            # 使用epoch和seed生成随机序列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            
        if not self.drop_last:
            # 填充以确保可以整除
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # 移除多余的数据
            indices = indices[:self.total_size]
            
        # 分发数据到各个进程
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
        
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于随机种子"""
        self.epoch = epoch

class AdaptiveBatchSampler(Sampler[T_co]):
    """自适应批处理采样器"""
    
    def __init__(
        self,
        dataset: Dataset,
        initial_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("分布式包不可用，请检查安装")
            num_replicas = dist.get_world_size()
            
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("分布式包不可用，请检查安装")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        
        # 性能监控
        self.batch_times = []
        self.memory_usage = []
        self.gradient_norms = []
        
        # 自适应调整参数
        self.adjustment_window = 100
        self.adjustment_threshold = 0.1
        self.momentum = 0.9
        
    def update_metrics(
        self,
        batch_time: float,
        memory_used: float,
        gradient_norm: float
    ) -> None:
        """更新性能指标"""
        self.batch_times.append(batch_time)
        self.memory_usage.append(memory_used)
        self.gradient_norms.append(gradient_norm)
        
        # 保持固定窗口大小
        if len(self.batch_times) > self.adjustment_window:
            self.batch_times.pop(0)
            self.memory_usage.pop(0)
            self.gradient_norms.pop(0)
            
        # 调整批处理大小
        if len(self.batch_times) == self.adjustment_window:
            self._adjust_batch_size()
            
    def _adjust_batch_size(self) -> None:
        """调整批处理大小"""
        # 计算性能指标
        avg_time = sum(self.batch_times) / len(self.batch_times)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        avg_grad_norm = sum(self.gradient_norms) / len(self.gradient_norms)
        
        # 计算调整因子
        time_factor = 1.0
        if avg_time > self.adjustment_threshold:
            time_factor = 0.9
        elif avg_time < self.adjustment_threshold / 2:
            time_factor = 1.1
            
        memory_factor = 1.0
        if avg_memory > 0.9:  # 90% 内存使用率
            memory_factor = 0.9
            
        grad_factor = 1.0
        if avg_grad_norm > 10.0:
            grad_factor = 0.9
        elif avg_grad_norm < 1.0:
            grad_factor = 1.1
            
        # 综合调整因子
        adjustment = time_factor * memory_factor * grad_factor
        
        # 应用动量
        adjustment = self.momentum + (1 - self.momentum) * adjustment
        
        # 调整批处理大小
        new_batch_size = int(self.current_batch_size * adjustment)
        new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
        
        # 更新批处理大小
        self.current_batch_size = new_batch_size
        
    def __iter__(self) -> Iterator[T_co]:
        """返回数据索引迭代器"""
        if self.shuffle:
            # 使用epoch和seed生成随机序列
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
            
        # 计算每个进程的数据量
        total_size = len(indices)
        num_samples = self.current_batch_size * (total_size // (self.current_batch_size * self.num_replicas))
        
        # 分发数据到各个进程
        indices = indices[self.rank:total_size:self.num_replicas]
        
        # 按批处理大小分组
        batches = []
        for i in range(0, len(indices), self.current_batch_size):
            batch = indices[i:i + self.current_batch_size]
            if len(batch) == self.current_batch_size or not self.drop_last:
                batches.append(batch)
                
        return iter(batches)
        
    def __len__(self) -> int:
        """返回批次数量"""
        if self.drop_last:
            return len(self.dataset) // (self.current_batch_size * self.num_replicas)
        else:
            return (len(self.dataset) + self.current_batch_size - 1) // (self.current_batch_size * self.num_replicas)
            
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch"""
        self.epoch = epoch

class EnhancedDistributedDataLoader(DataLoader):
    """增强型分布式数据加载器"""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        min_batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        **kwargs: Any
    ):
        # 设置批处理大小范围
        self.initial_batch_size = batch_size
        self.min_batch_size = min_batch_size or max(1, batch_size // 2)
        self.max_batch_size = max_batch_size or batch_size * 2
        
        # 创建自适应采样器
        self.adaptive_sampler = AdaptiveBatchSampler(
            dataset=dataset,
            initial_batch_size=batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        
        # 初始化基类
        super().__init__(
            dataset=dataset,
            batch_sampler=self.adaptive_sampler,
            **kwargs
        )
        
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch"""
        self.adaptive_sampler.set_epoch(epoch)
        
    def update_batch_metrics(
        self,
        batch_time: float,
        memory_used: float,
        gradient_norm: float
    ) -> None:
        """更新批处理性能指标"""
        self.adaptive_sampler.update_metrics(batch_time, memory_used, gradient_norm)
        
    def get_current_batch_size(self) -> int:
        """获取当前批处理大小"""
        return self.adaptive_sampler.current_batch_size
        
    def get_batch_size_stats(self) -> Dict[str, Any]:
        """获取批处理大小统计信息"""
        return {
            'current_batch_size': self.adaptive_sampler.current_batch_size,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'initial_batch_size': self.initial_batch_size,
            'avg_batch_time': sum(self.adaptive_sampler.batch_times) / len(self.adaptive_sampler.batch_times) if self.adaptive_sampler.batch_times else 0,
            'avg_memory_usage': sum(self.adaptive_sampler.memory_usage) / len(self.adaptive_sampler.memory_usage) if self.adaptive_sampler.memory_usage else 0,
            'avg_gradient_norm': sum(self.adaptive_sampler.gradient_norms) / len(self.adaptive_sampler.gradient_norms) if self.adaptive_sampler.gradient_norms else 0
        } 