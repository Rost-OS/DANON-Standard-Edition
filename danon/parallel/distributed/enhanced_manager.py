"""
增强型分布式训练管理器
整合了原有的分布式管理功能，并添加了新的特性
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple, Union, Set
from collections import deque
import time
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import psutil
from datetime import datetime
import math

from danon.core.error_handler import ErrorHandler, DANONError
from danon.core.security import SecurityManager
from danon.parallel.distributed.heartbeat import HeartbeatManager
from danon.parallel.distributed.load_balancer import LoadBalancer
from danon.parallel.distributed.communicator import AsyncCommunicator

class NodeRole:
    MASTER = "master"
    WORKER = "worker"
    BACKUP_MASTER = "backup_master"

class NodeStatus:
    ACTIVE = "active"
    INACTIVE = "inactive"
    RECOVERING = "recovering"

class NodeResources:
    def __init__(
        self,
        gpu_count: int,
        gpu_memory: List[int],
        cpu_count: int,
        memory_total: int,
        network_bandwidth: float
    ):
        self.gpu_count = gpu_count
        self.gpu_memory = gpu_memory
        self.cpu_count = cpu_count
        self.memory_total = memory_total
        self.network_bandwidth = network_bandwidth

class NodeInfo:
    def __init__(
        self,
        node_id: str,
        rank: int,
        role: str,
        status: str,
        resources: NodeResources,
        last_heartbeat: float,
        performance_metrics: Dict[str, Any],
        load_stats: Dict[str, Any]
    ):
        self.node_id = node_id
        self.rank = rank
        self.role = role
        self.status = status
        self.resources = resources
        self.last_heartbeat = last_heartbeat
        self.performance_metrics = performance_metrics
        self.load_stats = load_stats

class EnhancedDistributedManager:
    """增强型分布式训练管理器"""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        backend: str = 'nccl',
        timeout: float = 30.0,
        max_retries: int = 3,
        checkpoint_freq: int = 100,
        enable_backup_masters: bool = True,
        encryption_key: Optional[bytes] = None,
        enable_fault_tolerance: bool = True,
        enable_adaptive_compression: bool = True,
        enable_performance_monitoring: bool = True
    ):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.timeout = timeout
        self.max_retries = max_retries
        self.checkpoint_freq = checkpoint_freq
        self.enable_backup_masters = enable_backup_masters
        self.enable_fault_tolerance = enable_fault_tolerance
        self.enable_adaptive_compression = enable_adaptive_compression
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # 初始化组件
        self.error_handler = ErrorHandler(self)
        self.security = SecurityManager(encryption_key)
        self.heartbeat = HeartbeatManager(world_size)
        self.load_balancer = LoadBalancer(world_size)
        self.communicator = AsyncCommunicator(self.security)
        
        # 节点状态
        self.node_states: Dict[int, NodeInfo] = {}
        self.backup_masters: Set[int] = set()
        
        # 性能监控
        self.performance_metrics = {
            'throughput': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'gpu_util': deque(maxlen=1000),
            'memory_util': deque(maxlen=1000),
            'network_bandwidth': deque(maxlen=1000),
            'compression_ratio': deque(maxlen=1000)
        }
        
        # 故障恢复状态
        self.recovery_state = {
            'in_recovery': False,
            'failed_nodes': set(),
            'recovery_start': None,
            'checkpoint_path': None
        }
        
        # 通信优化
        self.comm_stats = {
            'total_bytes_sent': 0,
            'total_time': 0.0,
            'num_syncs': 0,
            'compression_savings': 0.0
        }
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 初始化分布式环境
        self._init_distributed()
        
    def _init_distributed(self):
        """初始化分布式训练环境"""
        try:
            # 初始化进程组
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
                timeout=datetime.timedelta(seconds=self.timeout)
            )
            
            # 设置节点角色
            self._setup_node_roles()
            
            # 启动监控
            if self.enable_performance_monitoring:
                self._start_monitoring()
            
            # 如果是主节点，启动备份主节点选举
            if self.rank == 0 and self.enable_backup_masters:
                self._elect_backup_masters()
                
        except Exception as e:
            self.error_handler.handle(DANONError(
                "INIT_ERROR",
                str(e),
                severity="critical"
            ))
            
    def _setup_node_roles(self):
        """设置节点角色"""
        if self.rank == 0:
            role = NodeRole.MASTER
        else:
            role = NodeRole.WORKER
            
        # 初始化节点信息
        self.node_states[self.rank] = NodeInfo(
            node_id=f"node_{self.rank}",
            rank=self.rank,
            role=role,
            status=NodeStatus.ACTIVE,
            resources=self._get_node_resources(),
            last_heartbeat=time.time(),
            performance_metrics={},
            load_stats={}
        )
        
    def _get_node_resources(self) -> NodeResources:
        """获取节点资源信息"""
        gpu_count = torch.cuda.device_count()
        gpu_memory = [
            torch.cuda.get_device_properties(i).total_memory
            for i in range(gpu_count)
        ]
        
        return NodeResources(
            gpu_count=gpu_count,
            gpu_memory=gpu_memory,
            cpu_count=os.cpu_count() or 1,
            memory_total=psutil.virtual_memory().total,
            network_bandwidth=self._measure_network_bandwidth()
        )
        
    def _measure_network_bandwidth(self) -> float:
        """测量网络带宽"""
        if self.rank == 0:
            # 创建测试数据
            data = torch.randn(1024, 1024, device='cuda')
            start_time = time.time()
            
            # 广播数据
            dist.broadcast(data, src=0)
            
            # 计算带宽
            elapsed = time.time() - start_time
            bytes_sent = data.element_size() * data.nelement()
            return bytes_sent / elapsed
        else:
            # 接收数据
            data = torch.empty(1024, 1024, device='cuda')
            dist.broadcast(data, src=0)
            return 0.0
            
    def _start_monitoring(self):
        """启动性能监控"""
        def monitor_loop():
            while True:
                try:
                    self._collect_metrics()
                    time.sleep(1)
                except Exception as e:
                    self.error_handler.handle(DANONError(
                        "MONITORING_ERROR",
                        str(e),
                        severity="warning"
                    ))
                    
        self.thread_pool.submit(monitor_loop)
        
    def _collect_metrics(self):
        """收集性能指标"""
        metrics = {}
        
        # GPU 利用率
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f'gpu_{i}_util'] = torch.cuda.utilization(i)
                
        # 内存使用
        memory = psutil.virtual_memory()
        metrics['memory_util'] = memory.percent
        
        # 更新性能指标
        node_info = self.node_states[self.rank]
        node_info.performance_metrics.update(metrics)
        
    def _elect_backup_masters(self):
        """选举备份主节点"""
        if not self.enable_backup_masters:
            return
            
        # 根据节点性能选择备份主节点
        available_nodes = [
            rank for rank, info in self.node_states.items()
            if rank != 0 and info.status == NodeStatus.ACTIVE
        ]
        
        if not available_nodes:
            return
            
        # 选择性能最好的节点作为备份主节点
        backup_masters = sorted(
            available_nodes,
            key=lambda r: self._calculate_node_score(self.node_states[r]),
            reverse=True
        )[:2]  # 选择2个备份节点
        
        self.backup_masters = set(backup_masters)
        
        # 通知被选中的节点
        for rank in backup_masters:
            self.node_states[rank].role = NodeRole.BACKUP_MASTER
            
    def _calculate_node_score(self, node_info: NodeInfo) -> float:
        """计算节点得分"""
        score = 0.0
        
        # 考虑 GPU 数量和显存
        score += node_info.resources.gpu_count * 10
        score += sum(node_info.resources.gpu_memory) / (1024 * 1024 * 1024)  # 转换为 GB
        
        # 考虑 CPU 和内存
        score += node_info.resources.cpu_count
        score += node_info.resources.memory_total / (1024 * 1024 * 1024)  # 转换为 GB
        
        # 考虑网络带宽
        score += node_info.resources.network_bandwidth / 1000  # 转换为 GB/s
        
        # 考虑负载情况
        if 'cpu_util' in node_info.load_stats:
            score *= (1 - node_info.load_stats['cpu_util'] / 100)
            
        return score
        
    def sync_gradients(self, model: nn.Module):
        """同步梯度"""
        if self.world_size <= 1:
            return
            
        start_time = time.time()
        total_bytes = 0
        
        try:
            for param in model.parameters():
                if param.grad is not None:
                    # 决定是否压缩
                    should_compress = (
                        self.enable_adaptive_compression and 
                        param.grad.numel() > 1e6
                    )
                    
                    if should_compress:
                        # 压缩梯度
                        compressed, meta = self._compress_gradients(param.grad)
                        
                        # 同步压缩后的梯度
                        dist.all_reduce(compressed)
                        
                        # 解压缩
                        param.grad.copy_(self._decompress_gradients(compressed, meta))
                        
                        # 更新统计信息
                        total_bytes += compressed.nelement() * compressed.element_size()
                        self.comm_stats['compression_savings'] += (
                            param.grad.nelement() * param.grad.element_size() -
                            compressed.nelement() * compressed.element_size()
                        )
                    else:
                        # 直接同步
                        dist.all_reduce(param.grad)
                        total_bytes += param.grad.nelement() * param.grad.element_size()
                        
                    # 平均梯度
                    param.grad.div_(self.world_size)
                    
        except Exception as e:
            self.error_handler.handle(DANONError(
                "SYNC_ERROR",
                str(e),
                severity="error"
            ))
            
        # 更新统计信息
        end_time = time.time()
        self.comm_stats['total_time'] += end_time - start_time
        self.comm_stats['total_bytes_sent'] += total_bytes
        self.comm_stats['num_syncs'] += 1
        
    def _compress_gradients(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩梯度"""
        # 计算绝对值的 top-k
        k = max(1, int(tensor.numel() * 0.01))  # 保留 1% 的值
        values, indices = torch.topk(tensor.abs().flatten(), k)
        
        # 创建稀疏张量
        compressed = torch.zeros(k * 2, dtype=torch.float32, device=tensor.device)
        compressed[0::2] = values
        compressed[1::2] = indices.float()
        
        return compressed, {
            'shape': tensor.shape,
            'original_dtype': tensor.dtype,
            'k': k
        }
        
    def _decompress_gradients(
        self,
        compressed: torch.Tensor,
        meta: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩梯度"""
        # 重建原始张量
        decompressed = torch.zeros(
            meta['shape'].numel(),
            dtype=meta['original_dtype'],
            device=compressed.device
        )
        
        values = compressed[0::2]
        indices = compressed[1::2].long()
        decompressed[indices] = values
        
        return decompressed.view(meta['shape'])
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {
            'communication': {
                'total_bytes_sent': self.comm_stats['total_bytes_sent'],
                'total_time': self.comm_stats['total_time'],
                'average_bandwidth': (
                    self.comm_stats['total_bytes_sent'] /
                    max(self.comm_stats['total_time'], 1e-6)
                ),
                'compression_ratio': (
                    self.comm_stats['compression_savings'] /
                    max(self.comm_stats['total_bytes_sent'], 1)
                )
            },
            'node_status': {
                rank: {
                    'role': info.role,
                    'status': info.status,
                    'performance': info.performance_metrics
                }
                for rank, info in self.node_states.items()
            }
        }
        
        if self.enable_performance_monitoring:
            stats['monitoring'] = {
                'throughput': list(self.performance_metrics['throughput']),
                'latency': list(self.performance_metrics['latency']),
                'gpu_util': list(self.performance_metrics['gpu_util']),
                'memory_util': list(self.performance_metrics['memory_util'])
            }
            
        return stats
        
    def shutdown(self):
        """关闭分布式训练环境"""
        try:
            # 停止监控
            if self.enable_performance_monitoring:
                self.thread_pool.shutdown(wait=True)
                
            # 清理通信组件
            if self.communicator:
                self.communicator.close()
                
            # 关闭分布式环境
            if dist.is_initialized():
                dist.destroy_process_group()
                
        except Exception as e:
            self.error_handler.handle(DANONError(
                "SHUTDOWN_ERROR",
                str(e),
                severity="warning"
            )) 