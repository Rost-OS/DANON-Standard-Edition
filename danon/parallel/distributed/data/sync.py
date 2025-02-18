"""
分布式同步策略
实现高效的参数同步和梯度聚合机制
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, List, Optional, Any, Iterator, Tuple
from threading import Lock
import time
from enum import Enum
import numpy as np
import math

class SyncStrategy(Enum):
    """同步策略"""
    FULL = "full"  # 完全同步
    PERIODIC = "periodic"  # 周期性同步
    ADAPTIVE = "adaptive"  # 自适应同步

class GradientCompressionType(Enum):
    """梯度压缩类型"""
    NONE = "none"  # 不压缩
    QUANTIZE = "quantize"  # 量化
    SPARSIFY = "sparsify"  # 稀疏化

class SyncConfig:
    """同步配置"""
    def __init__(
        self,
        sync_interval: int = 1,
        warmup_steps: int = 100,
        gradient_compression: bool = True,
        compression_ratio: float = 0.01,
        timeout: float = 30.0,
        max_retries: int = 3,
        enable_adaptive_sync: bool = True,
        enable_fault_tolerance: bool = True
    ):
        self.sync_interval = sync_interval
        self.warmup_steps = warmup_steps
        self.gradient_compression = gradient_compression
        self.compression_ratio = compression_ratio
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_adaptive_sync = enable_adaptive_sync
        self.enable_fault_tolerance = enable_fault_tolerance

class GradientCompressor:
    """梯度压缩器：实现高效的梯度压缩和解压缩"""
    def __init__(self, config: SyncConfig):
        self.config = config
        self.compression_ratio = config.compression_ratio
        self.residuals = {}
        
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """压缩梯度"""
        # 添加残差
        if tensor in self.residuals:
            tensor = tensor + self.residuals[tensor]
            
        # 计算阈值
        threshold = torch.quantile(
            torch.abs(tensor),
            1 - self.compression_ratio
        )
        
        # 生成掩码
        mask = torch.abs(tensor) >= threshold
        
        # 更新残差
        self.residuals[tensor] = tensor * (~mask)
        
        # 压缩
        compressed = tensor * mask
        return compressed, mask
        
    def decompress(self, compressed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """解压缩梯度"""
        return compressed

class AdaptiveSynchronizer:
    """自适应同步器：动态调整同步策略"""
    def __init__(self, config: SyncConfig):
        self.config = config
        self.step_count = 0
        self.sync_history = []
        self.bandwidth_history = []
        
    def should_sync(self, gradient_norm: float) -> bool:
        """决定是否需要同步"""
        if not self.config.enable_adaptive_sync:
            return self.step_count % self.config.sync_interval == 0
            
        # 预热阶段
        if self.step_count < self.config.warmup_steps:
            self.step_count += 1
            return True
            
        # 根据梯度范数和带宽历史决定
        if len(self.bandwidth_history) > 0:
            avg_bandwidth = sum(self.bandwidth_history[-10:]) / len(self.bandwidth_history[-10:])
            if gradient_norm > 2.0 * avg_bandwidth:
                return True
                
        self.step_count += 1
        return self.step_count % self.config.sync_interval == 0
        
    def update_stats(self, sync_time: float, data_size: int):
        """更新同步统计信息"""
        self.sync_history.append(sync_time)
        bandwidth = data_size / sync_time if sync_time > 0 else float('inf')
        self.bandwidth_history.append(bandwidth)
        
        # 保持历史记录在合理范围内
        if len(self.sync_history) > 100:
            self.sync_history.pop(0)
            self.bandwidth_history.pop(0)

class FaultTolerantSyncer:
    """容错同步器：处理同步过程中的错误"""
    def __init__(self, config: SyncConfig):
        self.config = config
        self.error_history = []
        
    def sync_with_retry(self, func, *args, **kwargs) -> Any:
        """带重试的同步操作"""
        if not self.config.enable_fault_tolerance:
            return func(*args, **kwargs)
            
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                sync_time = time.time() - start_time
                self.error_history.append(None)  # 成功
                return result, sync_time
            except Exception as e:
                self.error_history.append(e)
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 10))  # 指数退避
                
        return None, 0.0
        
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_errors = len([e for e in self.error_history if e is not None])
        return {
            'total_attempts': len(self.error_history),
            'error_rate': total_errors / len(self.error_history) if self.error_history else 0,
            'recent_errors': [str(e) for e in self.error_history[-5:] if e is not None]
        }

class DistributedSynchronizer:
    """分布式同步器：管理参数同步和梯度聚合"""
    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        self.compressor = GradientCompressor(self.config)
        self.adaptive_sync = AdaptiveSynchronizer(self.config)
        self.fault_tolerant = FaultTolerantSyncer(self.config)
        
    def sync_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """同步梯度"""
        synced_grads = {}
        total_data_size = 0
        
        for name, grad in gradients.items():
            # 计算梯度范数
            grad_norm = torch.norm(grad)
            
            # 决定是否需要同步
            if not self.adaptive_sync.should_sync(grad_norm):
                synced_grads[name] = grad
                continue
                
            # 压缩梯度
            if self.config.gradient_compression:
                compressed, mask = self.compressor.compress(grad)
                total_data_size += compressed.numel() * compressed.element_size()
                
                # 同步压缩后的梯度
                def sync_func():
                    dist.all_reduce(compressed)
                    return self.compressor.decompress(compressed, mask)
                    
                synced_grad, sync_time = self.fault_tolerant.sync_with_retry(sync_func)
            else:
                total_data_size += grad.numel() * grad.element_size()
                
                # 直接同步原始梯度
                def sync_func():
                    dist.all_reduce(grad)
                    return grad
                    
                synced_grad, sync_time = self.fault_tolerant.sync_with_retry(sync_func)
                
            if synced_grad is not None:
                synced_grads[name] = synced_grad
                self.adaptive_sync.update_stats(sync_time, total_data_size)
                
        return synced_grads
        
    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor], src: int = 0) -> Dict[str, torch.Tensor]:
        """广播模型参数"""
        broadcasted_params = {}
        
        for name, param in parameters.items():
            def sync_func():
                dist.broadcast(param, src=src)
                return param
                
            synced_param, _ = self.fault_tolerant.sync_with_retry(sync_func)
            if synced_param is not None:
                broadcasted_params[name] = synced_param
                
        return broadcasted_params
        
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        return {
            'adaptive_sync': {
                'step_count': self.adaptive_sync.step_count,
                'avg_sync_time': sum(self.adaptive_sync.sync_history) / len(self.adaptive_sync.sync_history) if self.adaptive_sync.sync_history else 0,
                'avg_bandwidth': sum(self.adaptive_sync.bandwidth_history) / len(self.adaptive_sync.bandwidth_history) if self.adaptive_sync.bandwidth_history else 0
            },
            'fault_tolerance': self.fault_tolerant.get_error_stats()
        }

class ElasticTraining:
    """弹性训练管理器"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        min_workers: int = 1,
        max_workers: int = 8,
        timeout: float = 60.0,
        sync_interval: int = 10
    ):
        self.model = model
        self.optimizer = optimizer
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.timeout = timeout
        self.sync_interval = sync_interval
        
        # 工作节点状态
        self.active_workers = {}
        self.worker_metrics = {}
        self.step_counter = 0
        
        # 性能监控
        self.performance_metrics = {
            'compute_efficiency': [],
            'communication_overhead': [],
            'worker_utilization': []
        }
        
    def save_checkpoint(self) -> Dict[str, Any]:
        """保存检查点"""
        return {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'worker_metrics': self.worker_metrics,
            'performance_metrics': self.performance_metrics
        }
        
    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """加载检查点"""
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.step_counter = checkpoint['step_counter']
        self.worker_metrics = checkpoint['worker_metrics']
        self.performance_metrics = checkpoint['performance_metrics']
        
    def handle_worker_join(self, worker_id: int, worker_info: Dict[str, Any] = None):
        """处理工作节点加入"""
        if len(self.active_workers) >= self.max_workers:
            return False
            
        self.active_workers[worker_id] = {
            'join_time': time.time(),
            'last_heartbeat': time.time(),
            'status': 'active',
            'compute_capacity': worker_info.get('compute_capacity', 1.0),
            'memory_available': worker_info.get('memory_available', 0),
            'network_bandwidth': worker_info.get('network_bandwidth', 0)
        }
        
        self.worker_metrics[worker_id] = {
            'compute_efficiency': 0.0,
            'communication_overhead': 0.0,
            'utilization': 0.0,
            'training_progress': 0.0
        }
        
        return True
        
    def handle_worker_leave(self, worker_id: int):
        """处理工作节点离开"""
        if worker_id in self.active_workers:
            # 记录性能指标
            metrics = self.worker_metrics[worker_id]
            for key, value in metrics.items():
                if key in self.performance_metrics:
                    self.performance_metrics[key].append(value)
                    
            del self.active_workers[worker_id]
            del self.worker_metrics[worker_id]
            
    def update_worker_status(self, worker_id: int, metrics: Dict[str, float] = None):
        """更新工作节点状态"""
        if worker_id in self.active_workers:
            self.active_workers[worker_id]['last_heartbeat'] = time.time()
            if metrics:
                self.worker_metrics[worker_id].update(metrics)
                
    def check_timeouts(self) -> List[int]:
        """检查超时节点"""
        current_time = time.time()
        timed_out = []
        
        for worker_id, info in self.active_workers.items():
            if current_time - info['last_heartbeat'] > self.timeout:
                timed_out.append(worker_id)
                
        return timed_out
        
    def should_sync(self) -> bool:
        """判断是否需要同步"""
        return self.step_counter % self.sync_interval == 0
        
    def step(self):
        """执行训练步骤"""
        self.step_counter += 1
        
        # 检查超时节点
        timed_out = self.check_timeouts()
        for worker_id in timed_out:
            self.handle_worker_leave(worker_id)
            
        # 更新性能指标
        self._update_performance_metrics()
        
    def _update_performance_metrics(self):
        """更新性能指标"""
        total_workers = len(self.active_workers)
        if total_workers == 0:
            return
            
        # 计算平均指标
        avg_metrics = {
            'compute_efficiency': 0.0,
            'communication_overhead': 0.0,
            'utilization': 0.0
        }
        
        for metrics in self.worker_metrics.values():
            for key in avg_metrics:
                avg_metrics[key] += metrics[key]
                
        for key in avg_metrics:
            avg_metrics[key] /= total_workers
            self.performance_metrics[key].append(avg_metrics[key])
            
    def get_worker_metrics(self) -> Dict[str, Dict[str, float]]:
        """获取工作节点指标"""
        return self.worker_metrics
        
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'active_workers': len(self.active_workers),
            'step_counter': self.step_counter,
            'performance_metrics': {
                k: sum(v[-10:]) / len(v[-10:]) if v else 0  # 最近10步的平均值
                for k, v in self.performance_metrics.items()
            },
            'worker_status': {
                worker_id: {
                    'status': info['status'],
                    'uptime': time.time() - info['join_time'],
                    'compute_capacity': info['compute_capacity']
                }
                for worker_id, info in self.active_workers.items()
            }
        }

class CommunicationOptimizer:
    """通信优化器"""
    
    def __init__(
        self,
        compression_type: GradientCompressionType = GradientCompressionType.NONE,
        compression_ratio: float = 0.1,
        overlap_communication: bool = True
    ):
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio
        self.overlap_communication = overlap_communication
        
        # 压缩统计
        self.compression_stats = {
            'total_bytes_original': 0,
            'total_bytes_compressed': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }
        
    def compress_tensor(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩张量"""
        start_time = time.time()
        original_size = tensor.nelement() * tensor.element_size()
        
        if self.compression_type == GradientCompressionType.QUANTIZE:
            # 量化压缩
            scale = tensor.abs().max()
            compressed = torch.round(tensor / scale * 255).to(torch.uint8)
            meta = {'scale': scale, 'original_dtype': tensor.dtype}
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            # 稀疏化压缩
            threshold = self._calculate_threshold(tensor)
            mask = tensor.abs() > threshold
            indices = mask.nonzero(as_tuple=True)
            values = tensor[indices]
            compressed = (indices, values)
            meta = {
                'shape': tensor.shape,
                'threshold': threshold,
                'nnz': len(values)
            }
            
        else:
            # 不压缩
            compressed = tensor
            meta = {}
            
        # 更新统计信息
        compressed_size = sum(
            t.nelement() * t.element_size()
            for t in (compressed if isinstance(compressed, tuple) else [compressed])
        )
        self.compression_stats['total_bytes_original'] += original_size
        self.compression_stats['total_bytes_compressed'] += compressed_size
        self.compression_stats['compression_time'] += time.time() - start_time
        
        return compressed, meta
        
    def decompress_tensor(
        self,
        compressed: torch.Tensor,
        meta: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩张量"""
        start_time = time.time()
        
        if self.compression_type == GradientCompressionType.QUANTIZE:
            # 反量化
            tensor = compressed.to(torch.float32) * meta['scale'] / 255
            tensor = tensor.to(meta['original_dtype'])
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            # 重建稀疏张量
            indices, values = compressed
            tensor = torch.zeros(meta['shape'], dtype=values.dtype)
            tensor[indices] = values
            
        else:
            # 无需解压缩
            tensor = compressed
            
        self.compression_stats['decompression_time'] += time.time() - start_time
        return tensor
        
    def optimize_communication(
        self,
        tensor: torch.Tensor,
        group: Optional[dist.ProcessGroup] = None
    ) -> torch.Tensor:
        """优化通信过程"""
        # 压缩
        compressed, meta = self.compress_tensor(tensor)
        
        # 异步通信
        if self.overlap_communication:
            # 创建通信缓冲区
            buffer = [
                torch.zeros_like(t)
                for t in (compressed if isinstance(compressed, tuple) else [compressed])
            ]
            
            # 启动异步通信
            handles = []
            for src, dst in zip(compressed if isinstance(compressed, tuple) else [compressed],
                              buffer):
                handle = dist.all_reduce(dst, group=group, async_op=True)
                handles.append(handle)
                
            # 等待通信完成
            for handle in handles:
                handle.wait()
                
            # 解压缩
            result = self.decompress_tensor(
                tuple(buffer) if isinstance(compressed, tuple) else buffer[0],
                meta
            )
            
        else:
            # 同步通信
            if isinstance(compressed, tuple):
                for t in compressed:
                    dist.all_reduce(t, group=group)
            else:
                dist.all_reduce(compressed, group=group)
                
            # 解压缩
            result = self.decompress_tensor(compressed, meta)
            
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        total_bytes_original = self.compression_stats['total_bytes_original']
        total_bytes_compressed = self.compression_stats['total_bytes_compressed']
        
        return {
            'compression_ratio': total_bytes_compressed / total_bytes_original
                if total_bytes_original > 0 else 1.0,
            'total_bytes_saved': total_bytes_original - total_bytes_compressed,
            'average_compression_time': self.compression_stats['compression_time'] /
                (self.compression_stats['total_bytes_original'] / 1e6)
                if self.compression_stats['total_bytes_original'] > 0 else 0.0,
            'average_decompression_time': self.compression_stats['decompression_time'] /
                (self.compression_stats['total_bytes_compressed'] / 1e6)
                if self.compression_stats['total_bytes_compressed'] > 0 else 0.0
        }

class AdaptiveGradientCompressor:
    """自适应梯度压缩器"""
    
    def __init__(
        self,
        initial_compression_ratio: float = 0.1,
        min_compression_ratio: float = 0.01,
        max_compression_ratio: float = 0.5,
        adjustment_interval: int = 100
    ):
        self.compression_ratio = initial_compression_ratio
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.adjustment_interval = adjustment_interval
        
        self.step_counter = 0
        self.compression_errors = []
        self.communication_times = []
        self.gradient_norms = []
        
        self.error_feedback = {}
        self.momentum = 0.9
        
    def compress(
        self,
        tensor: torch.Tensor,
        name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """压缩梯度"""
        # 添加误差反馈
        if name in self.error_feedback:
            tensor = tensor + self.error_feedback[name]
            
        # 计算梯度范数
        grad_norm = torch.norm(tensor)
        self.gradient_norms.append(grad_norm.item())
        
        # 计算压缩阈值
        k = max(1, int(tensor.numel() * self.compression_ratio))
        values, indices = torch.topk(tensor.abs().view(-1), k)
        threshold = values[-1]
        
        # 生成掩码
        mask = tensor.abs() >= threshold
        
        # 压缩
        compressed = tensor * mask
        
        # 更新误差反馈
        error = tensor - compressed
        self.error_feedback[name] = error
        
        # 计算压缩误差
        compression_error = torch.norm(error) / (grad_norm + 1e-8)
        self.compression_errors.append(compression_error.item())
        
        return compressed, mask
        
    def decompress(
        self,
        compressed: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """解压梯度"""
        return compressed
        
    def update_stats(
        self,
        communication_time: float
    ) -> None:
        """更新统计信息"""
        self.communication_times.append(communication_time)
        self.step_counter += 1
        
        # 调整压缩率
        if self.step_counter % self.adjustment_interval == 0:
            self._adjust_compression_ratio()
            
    def _adjust_compression_ratio(self) -> None:
        """调整压缩率"""
        if len(self.compression_errors) == 0:
            return
            
        # 计算平均指标
        avg_error = sum(self.compression_errors[-self.adjustment_interval:]) / self.adjustment_interval
        avg_time = sum(self.communication_times[-self.adjustment_interval:]) / self.adjustment_interval
        avg_norm = sum(self.gradient_norms[-self.adjustment_interval:]) / self.adjustment_interval
        
        # 计算调整因子
        error_factor = 1.0
        if avg_error > 0.1:  # 误差过大
            error_factor = 0.9
        elif avg_error < 0.01:  # 误差较小
            error_factor = 1.1
            
        time_factor = 1.0
        if avg_time > 0.1:  # 通信时间过长
            time_factor = 0.9
        elif avg_time < 0.01:  # 通信时间较短
            time_factor = 1.1
            
        norm_factor = 1.0
        if avg_norm > 10.0:  # 梯度范数过大
            norm_factor = 0.9
        elif avg_norm < 0.1:  # 梯度范数较小
            norm_factor = 1.1
            
        # 综合调整因子
        adjustment = error_factor * time_factor * norm_factor
        
        # 应用动量
        adjustment = self.momentum + (1 - self.momentum) * adjustment
        
        # 更新压缩率
        new_ratio = self.compression_ratio * adjustment
        self.compression_ratio = max(
            self.min_compression_ratio,
            min(self.max_compression_ratio, new_ratio)
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'compression_ratio': self.compression_ratio,
            'avg_error': sum(self.compression_errors) / len(self.compression_errors) if self.compression_errors else 0,
            'avg_time': sum(self.communication_times) / len(self.communication_times) if self.communication_times else 0,
            'avg_norm': sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0
        }

class DynamicCommunicationScheduler:
    """动态通信调度器"""
    
    def __init__(
        self,
        world_size: int,
        bandwidth_threshold: float = 0.8,
        latency_threshold: float = 0.1
    ):
        self.world_size = world_size
        self.bandwidth_threshold = bandwidth_threshold
        self.latency_threshold = latency_threshold
        
        self.bandwidth_usage = []
        self.communication_latency = []
        self.pending_operations = []
        
        self.ring_groups = self._create_ring_groups()
        self.tree_groups = self._create_tree_groups()
        
    def _create_ring_groups(self) -> List[List[int]]:
        """创建环形通信组"""
        groups = []
        for i in range(self.world_size):
            group = [(i + j) % self.world_size for j in range(self.world_size)]
            groups.append(group)
        return groups
        
    def _create_tree_groups(self) -> List[List[int]]:
        """创建树形通信组"""
        groups = []
        levels = int(math.log2(self.world_size))
        for level in range(levels):
            stride = 2 ** level
            group = []
            for i in range(0, self.world_size, stride):
                group.append(list(range(i, min(i + stride, self.world_size))))
            groups.append(group)
        return groups
        
    def schedule_communication(
        self,
        tensor_sizes: List[int],
        priorities: List[float]
    ) -> List[Tuple[int, int, int]]:
        """调度通信操作"""
        # 计算当前带宽使用率
        current_bandwidth = sum(self.bandwidth_usage[-10:]) / 10 if self.bandwidth_usage else 0
        
        # 选择通信模式
        if current_bandwidth > self.bandwidth_threshold:
            # 带宽受限，使用树形通信
            schedule = self._schedule_tree(tensor_sizes, priorities)
        else:
            # 带宽充足，使用环形通信
            schedule = self._schedule_ring(tensor_sizes, priorities)
            
        return schedule
        
    def _schedule_ring(
        self,
        tensor_sizes: List[int],
        priorities: List[float]
    ) -> List[Tuple[int, int, int]]:
        """环形通信调度"""
        schedule = []
        
        # 按优先级排序
        indices = list(range(len(tensor_sizes)))
        indices.sort(key=lambda i: priorities[i], reverse=True)
        
        # 分配到不同的环
        for i, idx in enumerate(indices):
            ring_idx = i % len(self.ring_groups)
            ring = self.ring_groups[ring_idx]
            
            # 添加通信操作
            for j in range(len(ring) - 1):
                schedule.append((ring[j], ring[j + 1], idx))
                
        return schedule
        
    def _schedule_tree(
        self,
        tensor_sizes: List[int],
        priorities: List[float]
    ) -> List[Tuple[int, int, int]]:
        """树形通信调度"""
        schedule = []
        
        # 按优先级排序
        indices = list(range(len(tensor_sizes)))
        indices.sort(key=lambda i: priorities[i], reverse=True)
        
        # 分配到不同的树
        for i, idx in enumerate(indices):
            level_idx = i % len(self.tree_groups)
            level_groups = self.tree_groups[level_idx]
            
            # 添加通信操作
            for group in level_groups:
                if len(group) > 1:
                    for j in range(1, len(group)):
                        schedule.append((group[0], group[j], idx))
                        
        return schedule
        
    def update_stats(
        self,
        bandwidth: float,
        latency: float
    ) -> None:
        """更新统计信息"""
        self.bandwidth_usage.append(bandwidth)
        self.communication_latency.append(latency)
        
        # 保持固定窗口大小
        if len(self.bandwidth_usage) > 100:
            self.bandwidth_usage.pop(0)
        if len(self.communication_latency) > 100:
            self.communication_latency.pop(0)
            
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'avg_bandwidth': sum(self.bandwidth_usage) / len(self.bandwidth_usage) if self.bandwidth_usage else 0,
            'avg_latency': sum(self.communication_latency) / len(self.communication_latency) if self.communication_latency else 0,
            'pending_operations': len(self.pending_operations)
        }

class EnhancedDistributedSynchronizer(DistributedSynchronizer):
    """增强型分布式同步器"""
    
    def __init__(self, config: SyncConfig = None):
        super().__init__(config)
        
        # 自适应压缩器
        self.adaptive_compressor = AdaptiveGradientCompressor(
            initial_compression_ratio=0.1,
            min_compression_ratio=0.01,
            max_compression_ratio=0.5
        )
        
        # 动态通信调度器
        self.comm_scheduler = DynamicCommunicationScheduler(
            world_size=dist.get_world_size()
        )
        
    def sync_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """同步梯度"""
        sync_start = time.time()
        synced_grads = {}
        
        # 准备通信计划
        tensor_sizes = [grad.numel() * grad.element_size() for grad in gradients.values()]
        grad_norms = [torch.norm(grad).item() for grad in gradients.values()]
        priorities = [norm / size for norm, size in zip(grad_norms, tensor_sizes)]
        
        # 获取通信调度
        schedule = self.comm_scheduler.schedule_communication(tensor_sizes, priorities)
        
        # 执行通信
        for src, dst, grad_idx in schedule:
            if dist.get_rank() in (src, dst):
                name = list(gradients.keys())[grad_idx]
                grad = gradients[name]
                
                # 压缩梯度
                compressed, mask = self.adaptive_compressor.compress(grad, name)
                
                # 同步压缩后的梯度
                if dist.get_rank() == src:
                    dist.send(compressed, dst)
                else:
                    dist.recv(compressed, src)
                    
                # 解压梯度
                synced_grad = self.adaptive_compressor.decompress(compressed, mask)
                synced_grads[name] = synced_grad
                
        # 更新统计信息
        sync_time = time.time() - sync_start
        total_bytes = sum(tensor_sizes)
        bandwidth = total_bytes / (sync_time + 1e-8)
        
        self.adaptive_compressor.update_stats(sync_time)
        self.comm_scheduler.update_stats(bandwidth, sync_time)
        
        return synced_grads
        
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        return {
            'compressor': self.adaptive_compressor.get_stats(),
            'scheduler': self.comm_scheduler.get_stats()
        }

class ParameterSync:
    """参数同步器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sync_strategy: SyncStrategy = SyncStrategy.FULL,
        sync_period: int = 10,
        broadcast_buffers: bool = True,
        compression_type: GradientCompressionType = GradientCompressionType.NONE,
        compression_ratio: float = 0.1,
        enable_elastic: bool = True,
        min_workers: int = 1,
        max_workers: int = 8
    ):
        self.model = model
        self.device = device
        self.sync_strategy = sync_strategy
        self.sync_period = sync_period
        self.broadcast_buffers = broadcast_buffers
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio
        
        self.sync_lock = Lock()
        self.step_counter = 0
        self.last_sync_time = time.time()
        
        # 通信优化
        self.param_groups = self._group_parameters()
        self.compression_buffers = {}
        self.error_feedback = {}
        
        # 添加弹性训练支持
        self.elastic_training = ElasticTraining(
            model=model,
            optimizer=None,  # 需要在训练时设置
            min_workers=min_workers,
            max_workers=max_workers
        )
        
        # 添加通信优化器
        self.comm_optimizer = CommunicationOptimizer(
            compression_type=compression_type,
            compression_ratio=compression_ratio,
            overlap_communication=True
        )
        
    def _group_parameters(self) -> List[List[torch.nn.Parameter]]:
        """将参数分组以优化通信"""
        groups = []
        current_group = []
        current_size = 0
        target_size = 1024 * 1024  # 1MB
        
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
                
            param_size = param.numel() * param.element_size()
            if current_size + param_size > target_size and current_group:
                groups.append(current_group)
                current_group = []
                current_size = 0
                
            current_group.append(param)
            current_size += param_size
            
        if current_group:
            groups.append(current_group)
            
        return groups
        
    def _compress_gradients(
        self,
        gradients: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩梯度"""
        if self.compression_type == GradientCompressionType.NONE:
            return gradients, {}
            
        elif self.compression_type == GradientCompressionType.QUANTIZE:
            # 实现梯度量化
            max_val = torch.max(torch.abs(gradients))
            scale = max_val / 127.0
            compressed = torch.round(gradients / scale).to(torch.int8)
            return compressed, {'scale': scale}
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            # 实现梯度稀疏化
            k = int(gradients.numel() * self.compression_ratio)
            values, indices = torch.topk(torch.abs(gradients).view(-1), k)
            compressed = torch.zeros_like(gradients.view(-1))
            compressed[indices] = gradients.view(-1)[indices]
            return compressed.view_as(gradients), {'k': k}
            
    def _decompress_gradients(
        self,
        compressed: torch.Tensor,
        meta: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩梯度"""
        if self.compression_type == GradientCompressionType.NONE:
            return compressed
            
        elif self.compression_type == GradientCompressionType.QUANTIZE:
            return compressed.to(torch.float32) * meta['scale']
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            return compressed
            
    def should_sync(self) -> bool:
        """判断是否应该进行同步"""
        if self.sync_strategy == SyncStrategy.FULL:
            return True
            
        elif self.sync_strategy == SyncStrategy.PERIODIC:
            return self.step_counter % self.sync_period == 0
            
        elif self.sync_strategy == SyncStrategy.ADAPTIVE:
            # 基于时间和梯度变化自适应决定是否同步
            current_time = time.time()
            time_elapsed = current_time - self.last_sync_time
            
            if time_elapsed > 60:  # 至少每分钟同步一次
                return True
                
            # 检查梯度变化
            grad_change = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_change += torch.norm(param.grad).item()
                    
            return grad_change > 1.0  # 可调整的阈值
            
    def broadcast_parameters(self) -> None:
        """广播模型参数到所有进程"""
        with self.sync_lock:
            for group in self.param_groups:
                # 将组内参数打包到连续内存
                sizes = [p.numel() for p in group]
                flat_params = torch.cat([p.data.view(-1) for p in group])
                
                # 广播参数
                dist.broadcast(flat_params, src=0)
                
                # 更新各个参数
                offset = 0
                for param, size in zip(group, sizes):
                    param.data.copy_(
                        flat_params[offset:offset + size].view_as(param)
                    )
                    offset += size
                    
            if self.broadcast_buffers:
                for buf in self.model.buffers():
                    dist.broadcast(buf.data, src=0)
                    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """设置优化器"""
        self.elastic_training.optimizer = optimizer
        
    def sync_gradients(self) -> None:
        """同步所有进程的梯度"""
        if not self.should_sync():
            return
            
        with self.sync_lock:
            for group in self.param_groups:
                # 将组内梯度打包
                sizes = [p.grad.numel() for p in group if p.grad is not None]
                flat_grads = torch.cat([
                    p.grad.data.view(-1) for p in group if p.grad is not None
                ])
                
                # 使用通信优化器进行梯度同步
                optimized_grads = self.comm_optimizer.optimize_communication(flat_grads)
                
                # 更新各个参数的梯度
                offset = 0
                for param, size in zip(group, sizes):
                    if param.grad is not None:
                        param.grad.data.copy_(
                            optimized_grads[offset:offset + size].view_as(param.grad)
                        )
                        offset += size
                        
        self.step_counter += 1
        self.last_sync_time = time.time()
        
    def handle_worker_change(self, worker_id: int, is_joining: bool):
        """处理工作节点变化"""
        if is_joining:
            checkpoint = self.elastic_training.handle_worker_join(worker_id)
            if checkpoint is not None:
                self.load_state_dict(checkpoint)
        else:
            self.elastic_training.handle_worker_leave(worker_id)
            
    def check_worker_status(self):
        """检查工作节点状态"""
        self.elastic_training.check_timeouts()
        
    def save_state_dict(self) -> Dict[str, Any]:
        """保存状态字典"""
        return {
            'model_state': self.model.state_dict(),
            'elastic_state': self.elastic_training.save_checkpoint(),
            'comm_stats': self.comm_optimizer.get_stats()
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.model.load_state_dict(state_dict['model_state'])
        self.elastic_training.load_checkpoint(state_dict['elastic_state'])
        
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        return {
            'sync_strategy': self.sync_strategy.value,
            'compression_type': self.compression_type.value,
            'steps_since_last_sync': self.step_counter % self.sync_period if self.sync_strategy == SyncStrategy.PERIODIC else 0,
            'time_since_last_sync': time.time() - self.last_sync_time
        }

class GradientAllReduce:
    """梯度聚合器"""
    
    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: float = 25,
        overlap_comm: bool = True,
        compression_type: GradientCompressionType = GradientCompressionType.NONE,
        compression_ratio: float = 0.1
    ):
        self.model = model
        self.bucket_size_mb = bucket_size_mb
        self.overlap_comm = overlap_comm
        self.compression_type = compression_type
        self.compression_ratio = compression_ratio
        
        # 初始化梯度桶
        self.buckets: List[List[torch.Tensor]] = []
        self.bucket_sizes: List[int] = []
        self._initialize_buckets()
        
        # 用于通信重叠的缓冲区
        self.grad_buffers: Dict[int, torch.Tensor] = {}
        self.comm_lock = Lock()
        
        # 压缩相关
        self.compression_buffers = {}
        self.error_feedback = {}
        
    def _initialize_buckets(self) -> None:
        """初始化梯度桶"""
        current_bucket: List[torch.Tensor] = []
        current_size = 0
        bucket_size = self.bucket_size_mb * 1024 * 1024  # 转换为字节
        
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
                
            param_size = param.numel() * param.element_size()
            if current_size + param_size > bucket_size and current_bucket:
                self.buckets.append(current_bucket)
                self.bucket_sizes.append(current_size)
                current_bucket = []
                current_size = 0
                
            current_bucket.append(param)
            current_size += param_size
            
        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_sizes.append(current_size)
            
    def _compress_bucket(
        self,
        bucket: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩梯度桶"""
        if self.compression_type == GradientCompressionType.NONE:
            return bucket, {}
            
        elif self.compression_type == GradientCompressionType.QUANTIZE:
            max_val = torch.max(torch.abs(bucket))
            scale = max_val / 127.0
            compressed = torch.round(bucket / scale).to(torch.int8)
            return compressed, {'scale': scale}
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            k = int(bucket.numel() * self.compression_ratio)
            values, indices = torch.topk(torch.abs(bucket).view(-1), k)
            compressed = torch.zeros_like(bucket.view(-1))
            compressed[indices] = bucket.view(-1)[indices]
            return compressed.view_as(bucket), {'k': k}
            
    def _decompress_bucket(
        self,
        compressed: torch.Tensor,
        meta: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩梯度桶"""
        if self.compression_type == GradientCompressionType.NONE:
            return compressed
            
        elif self.compression_type == GradientCompressionType.QUANTIZE:
            return compressed.to(torch.float32) * meta['scale']
            
        elif self.compression_type == GradientCompressionType.SPARSIFY:
            return compressed
            
    def _allreduce_bucket(
        self,
        bucket: List[torch.Tensor],
        async_op: bool = False
    ) -> Optional[dist.Work]:
        """对一个梯度桶进行AllReduce操作"""
        # 将梯度打包到连续内存
        bucket_size = sum(param.grad.numel() for param in bucket)
        buffer = self.grad_buffers.get(bucket_size)
        
        if buffer is None:
            buffer = torch.empty(
                bucket_size,
                dtype=bucket[0].grad.dtype,
                device=bucket[0].grad.device
            )
            self.grad_buffers[bucket_size] = buffer
            
        # 复制梯度到缓冲区
        offset = 0
        for param in bucket:
            grad_data = param.grad.data
            numel = grad_data.numel()
            buffer[offset:offset + numel].copy_(grad_data.view(-1))
            offset += numel
            
        # 压缩梯度
        compressed_buffer, meta = self._compress_bucket(buffer)
        
        # 执行AllReduce
        with self.comm_lock:
            work = dist.all_reduce(
                compressed_buffer,
                op=dist.ReduceOp.SUM,
                async_op=async_op
            )
            
        if not async_op:
            # 解压缩并复制回参数
            decompressed = self._decompress_bucket(compressed_buffer, meta)
            decompressed.div_(dist.get_world_size())
            
            offset = 0
            for param in bucket:
                grad_data = param.grad.data
                numel = grad_data.numel()
                grad_data.copy_(
                    decompressed[offset:offset + numel].view_as(grad_data)
                )
                offset += numel
                
        return work if async_op else None
        
    def reduce_gradients(self) -> None:
        """执行梯度聚合"""
        if not self.overlap_comm:
            # 同步模式
            for bucket in self.buckets:
                self._allreduce_bucket(bucket)
        else:
            # 异步模式，使用通信重叠
            works: List[dist.Work] = []
            for i, bucket in enumerate(self.buckets):
                # 启动异步AllReduce
                work = self._allreduce_bucket(bucket, async_op=True)
                if work is not None:
                    works.append(work)
                    
                # 等待前一个桶完成并更新梯度
                if i > 0 and works:
                    prev_work = works[-2]
                    prev_work.wait()
                    
                    # 解压缩并更新前一个桶的梯度
                    prev_bucket = self.buckets[i-1]
                    buffer = self.grad_buffers[sum(p.grad.numel() for p in prev_bucket)]
                    compressed_buffer = buffer
                    meta = {}  # 需要保存压缩元数据
                    decompressed = self._decompress_bucket(compressed_buffer, meta)
                    decompressed.div_(dist.get_world_size())
                    
                    offset = 0
                    for param in prev_bucket:
                        grad_data = param.grad.data
                        numel = grad_data.numel()
                        grad_data.copy_(
                            decompressed[offset:offset + numel].view_as(grad_data)
                        )
                        offset += numel
                        
            # 等待最后一个桶完成
            if works:
                works[-1].wait()
                last_bucket = self.buckets[-1]
                buffer = self.grad_buffers[sum(p.grad.numel() for p in last_bucket)]
                compressed_buffer = buffer
                meta = {}
                decompressed = self._decompress_bucket(compressed_buffer, meta)
                decompressed.div_(dist.get_world_size())
                
                offset = 0
                for param in last_bucket:
                    grad_data = param.grad.data
                    numel = grad_data.numel()
                    grad_data.copy_(
                        decompressed[offset:offset + numel].view_as(grad_data)
                    )
                    offset += numel 