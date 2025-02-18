"""
性能分析工具
实现了系统性能监控和分析功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import time
import psutil
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from enum import Enum

class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"  # 延迟
    MEMORY = "memory"    # 内存
    COMPUTE = "compute"  # 计算
    IO = "io"           # 输入输出
    NETWORK = "network" # 网络

@dataclass
class ProfilePoint:
    """性能分析点"""
    name: str
    metric_type: MetricType
    timestamp: float
    value: float
    metadata: Dict[str, Any]

class SystemProfiler:
    """系统性能分析器"""
    
    def __init__(
        self,
        enabled: bool = True,
        log_dir: Optional[str] = None,
        sampling_interval: float = 0.1
    ):
        self.enabled = enabled
        self.log_dir = log_dir
        self.sampling_interval = sampling_interval
        
        self.profile_points: List[ProfilePoint] = []
        self.start_time = time.time()
        self.current_context: List[str] = []
        
        # 初始化日志
        if log_dir:
            logging.basicConfig(
                filename=f"{log_dir}/profiler.log",
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
    @contextmanager
    def profile_section(self, name: str):
        """性能分析上下文管理器"""
        if not self.enabled:
            yield
            return
            
        self.current_context.append(name)
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # 记录延迟
            self.add_profile_point(
                name=name,
                metric_type=MetricType.LATENCY,
                value=end_time - start_time,
                metadata={'context': '/'.join(self.current_context)}
            )
            
            # 记录内存变化
            self.add_profile_point(
                name=name,
                metric_type=MetricType.MEMORY,
                value=end_memory - start_memory,
                metadata={'context': '/'.join(self.current_context)}
            )
            
            self.current_context.pop()
            
    def add_profile_point(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加性能分析点"""
        if not self.enabled:
            return
            
        point = ProfilePoint(
            name=name,
            metric_type=metric_type,
            timestamp=time.time() - self.start_time,
            value=value,
            metadata=metadata or {}
        )
        
        self.profile_points.append(point)
        
        # 记录日志
        if self.log_dir:
            logging.info(
                f"Profile point: {name} - {metric_type.value} - {value:.4f} - "
                f"metadata: {metadata}"
            )
            
    def get_statistics(
        self,
        metric_type: Optional[MetricType] = None,
        name_filter: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        stats = {}
        
        # 过滤数据点
        points = self.profile_points
        if metric_type:
            points = [p for p in points if p.metric_type == metric_type]
        if name_filter:
            points = [p for p in points if name_filter in p.name]
            
        # 按名称分组
        grouped_points = {}
        for point in points:
            if point.name not in grouped_points:
                grouped_points[point.name] = []
            grouped_points[point.name].append(point)
            
        # 计算统计信息
        for name, group in grouped_points.items():
            values = [p.value for p in group]
            stats[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
            
        return stats
        
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        bottlenecks = []
        
        # 分析延迟瓶颈
        latency_stats = self.get_statistics(metric_type=MetricType.LATENCY)
        for name, stats in latency_stats.items():
            if stats['mean'] > 0.1:  # 100ms阈值
                bottlenecks.append({
                    'type': 'latency',
                    'name': name,
                    'severity': 'high' if stats['mean'] > 1.0 else 'medium',
                    'stats': stats
                })
                
        # 分析内存瓶颈
        memory_stats = self.get_statistics(metric_type=MetricType.MEMORY)
        total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        
        for name, stats in memory_stats.items():
            if total_memory and stats['max'] > total_memory * 0.1:  # 10%阈值
                bottlenecks.append({
                    'type': 'memory',
                    'name': name,
                    'severity': 'high' if stats['max'] > total_memory * 0.5 else 'medium',
                    'stats': stats
                })
                
        return bottlenecks
        
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        report = {
            'summary': {
                'total_duration': time.time() - self.start_time,
                'total_points': len(self.profile_points),
                'unique_sections': len(set(p.name for p in self.profile_points))
            },
            'statistics': {},
            'bottlenecks': self.analyze_bottlenecks()
        }
        
        # 添加各类型的统计信息
        for metric_type in MetricType:
            report['statistics'][metric_type.value] = self.get_statistics(metric_type)
            
        return report
        
    def reset(self) -> None:
        """重置分析器"""
        self.profile_points.clear()
        self.start_time = time.time()
        self.current_context.clear()
        
class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_stats = []
        
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """分析内存使用情况"""
        stats = {
            'model_size': sum(p.numel() * p.element_size() for p in self.model.parameters()),
            'buffer_size': sum(b.numel() * b.element_size() for b in self.model.buffers()),
            'parameter_count': sum(p.numel() for p in self.model.parameters()),
            'buffer_count': sum(b.numel() for b in self.model.buffers()),
            'cuda_memory': {
                'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            },
            'system_memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used
            }
        }
        
        self.memory_stats.append(stats)
        return stats
        
    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """获取内存使用时间线"""
        return self.memory_stats
        
class ComputeProfiler:
    """计算分析器"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.compute_stats = []
        
    def analyze_compute_intensity(self) -> Dict[str, Any]:
        """分析计算密集度"""
        stats = {
            'flops': self._estimate_flops(),
            'memory_access': self._estimate_memory_access(),
            'arithmetic_intensity': self._calculate_arithmetic_intensity()
        }
        
        self.compute_stats.append(stats)
        return stats
        
    def _estimate_flops(self) -> int:
        """估算浮点运算数"""
        total_flops = 0
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 简单估算线性层和卷积层的FLOPs
                if isinstance(module, nn.Linear):
                    total_flops += module.in_features * module.out_features * 2
                else:  # Conv2d
                    total_flops += (
                        module.in_channels
                        * module.out_channels
                        * module.kernel_size[0]
                        * module.kernel_size[1]
                        * 2  # 乘加运算
                    )
        return total_flops
        
    def _estimate_memory_access(self) -> int:
        """估算内存访问量"""
        total_access = 0
        for param in self.model.parameters():
            total_access += param.numel() * param.element_size() * 2  # 读写各一次
        return total_access
        
    def _calculate_arithmetic_intensity(self) -> float:
        """计算算术密度"""
        flops = self._estimate_flops()
        memory_access = self._estimate_memory_access()
        return flops / max(1, memory_access)  # 避免除零
        
    def get_compute_timeline(self) -> List[Dict[str, Any]]:
        """获取计算使用时间线"""
        return self.compute_stats

class DistributedProfiler:
    """分布式性能分析器"""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        enabled: bool = True
    ):
        self.world_size = world_size
        self.rank = rank
        self.enabled = enabled
        self.communication_stats = []
        
    def record_communication(
        self,
        operation: str,
        size: int,
        duration: float
    ) -> None:
        """记录通信操作"""
        if not self.enabled:
            return
            
        stats = {
            'operation': operation,
            'size': size,
            'duration': duration,
            'bandwidth': size / max(duration, 1e-6),
            'timestamp': time.time()
        }
        
        self.communication_stats.append(stats)
        
    def analyze_communication(self) -> Dict[str, Any]:
        """分析通信性能"""
        if not self.communication_stats:
            return {}
            
        analysis = {
            'total_communication': sum(s['size'] for s in self.communication_stats),
            'total_duration': sum(s['duration'] for s in self.communication_stats),
            'average_bandwidth': np.mean([s['bandwidth'] for s in self.communication_stats]),
            'operation_breakdown': {}
        }
        
        # 按操作类型分析
        for op in set(s['operation'] for s in self.communication_stats):
            op_stats = [s for s in self.communication_stats if s['operation'] == op]
            analysis['operation_breakdown'][op] = {
                'count': len(op_stats),
                'total_size': sum(s['size'] for s in op_stats),
                'average_duration': np.mean([s['duration'] for s in op_stats]),
                'average_bandwidth': np.mean([s['bandwidth'] for s in op_stats])
            }
            
        return analysis
        
    def get_communication_timeline(self) -> List[Dict[str, Any]]:
        """获取通信时间线"""
        return self.communication_stats
        
    def reset(self) -> None:
        """重置分析器"""
        self.communication_stats.clear() 