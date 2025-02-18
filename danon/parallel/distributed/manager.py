"""
分布式训练管理器
实现了分布式训练的核心功能,包括节点管理、通信优化和训练控制
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Tuple, Set
import threading
import time
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os
import nvidia_smi
from collections import deque
import logging
import json
from pathlib import Path
import asyncio
import aiohttp
from cryptography.fernet import Fernet
from concurrent.futures import ThreadPoolExecutor
import socket
import struct
import hashlib
from datetime import datetime, timedelta

class NodeRole(Enum):
    """节点角色"""
    MASTER = "master"
    WORKER = "worker"
    BACKUP_MASTER = "backup_master"

class NodeStatus(Enum):
    """节点状态"""
    ACTIVE = "active"
    IDLE = "idle"
    FAILED = "failed"
    RECOVERING = "recovering"
    SYNCING = "syncing"

@dataclass
class NodeResources:
    """节点资源信息"""
    gpu_count: int
    gpu_memory: List[int]
    cpu_count: int
    memory_total: int
    network_bandwidth: float

@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    rank: int
    role: NodeRole
    status: NodeStatus
    resources: NodeResources
    last_heartbeat: float
    performance_metrics: Dict[str, float]
    load_stats: Dict[str, float]

class SecurityManager:
    """安全管理器"""
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def encrypt_message(self, message: bytes) -> bytes:
        """加密消息"""
        return self.cipher_suite.encrypt(message)
        
    def decrypt_message(self, encrypted_message: bytes) -> bytes:
        """解密消息"""
        return self.cipher_suite.decrypt(encrypted_message)
        
    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """验证消息签名"""
        expected_signature = hashlib.sha256(message + self.key).digest()
        return signature == expected_signature

class LoadBalancer:
    """负载均衡器"""
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.node_loads: Dict[int, float] = {}
        self.resource_usage: Dict[int, Dict[str, float]] = {}
        self.performance_history: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(world_size)
        }
        
    def update_node_load(self, rank: int, metrics: Dict[str, float]):
        """更新节点负载信息"""
        # 计算综合负载分数
        gpu_util = metrics.get('gpu_utilization', 0)
        memory_util = metrics.get('memory_utilization', 0)
        cpu_util = metrics.get('cpu_utilization', 0)
        network_util = metrics.get('network_utilization', 0)
        
        load_score = (
            0.4 * gpu_util +
            0.3 * memory_util +
            0.2 * cpu_util +
            0.1 * network_util
        )
        
        self.node_loads[rank] = load_score
        self.resource_usage[rank] = metrics
        self.performance_history[rank].append(load_score)
        
    def get_optimal_target(self, data_size: int) -> int:
        """获取最优的目标节点"""
        min_load = float('inf')
        target_rank = -1
        
        for rank, load in self.node_loads.items():
            if load < min_load:
                min_load = load
                target_rank = rank
                
        return target_rank
        
    def should_rebalance(self) -> bool:
        """判断是否需要重新平衡负载"""
        if not self.node_loads:
            return False
            
        max_load = max(self.node_loads.values())
        min_load = min(self.node_loads.values())
        
        return max_load - min_load > 0.3  # 负载差异超过30%时触发重平衡
        
    def get_rebalance_plan(self) -> List[Tuple[int, int]]:
        """生成负载重平衡方案"""
        if not self.should_rebalance():
            return []
            
        plan = []
        sorted_nodes = sorted(
            self.node_loads.items(),
            key=lambda x: x[1]
        )
        
        # 从负载最高的节点向负载最低的节点迁移
        while len(sorted_nodes) >= 2:
            high_rank, high_load = sorted_nodes.pop()
            low_rank, low_load = sorted_nodes[0]
            
            if high_load - low_load <= 0.3:
                break
                
            # 计算迁移量
            transfer_load = (high_load - low_load) / 2
            plan.append((high_rank, low_rank))
            
            # 更新负载
            new_low_load = low_load + transfer_load
            sorted_nodes[0] = (low_rank, new_low_load)
            sorted_nodes.sort(key=lambda x: x[1])
            
        return plan

class HeartbeatManager:
    """心跳管理器"""
    def __init__(
        self,
        world_size: int,
        heartbeat_interval: float = 1.0,
        timeout: float = 5.0
    ):
        self.world_size = world_size
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.last_heartbeats: Dict[int, float] = {}
        self.missed_heartbeats: Dict[int, int] = {}
        self._lock = threading.Lock()
        
    def record_heartbeat(self, rank: int):
        """记录心跳"""
        with self._lock:
            current_time = time.time()
            self.last_heartbeats[rank] = current_time
            self.missed_heartbeats[rank] = 0
            
    def check_timeouts(self) -> Set[int]:
        """检查超时节点"""
        current_time = time.time()
        timed_out = set()
        
        with self._lock:
            for rank in range(self.world_size):
                if rank not in self.last_heartbeats:
                    continue
                    
                time_since_last = current_time - self.last_heartbeats[rank]
                if time_since_last > self.timeout:
                    self.missed_heartbeats[rank] = self.missed_heartbeats.get(rank, 0) + 1
                    timed_out.add(rank)
                    
        return timed_out

class AsyncCommunicator:
    """异步通信管理器"""
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: Dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()
        
    async def send_message(
        self,
        target_rank: int,
        message: Any,
        timeout: float = 5.0
    ) -> Any:
        """异步发送消息"""
        message_id = str(uuid.uuid4())
        message_data = {
            'id': message_id,
            'sender_rank': dist.get_rank(),
            'content': message
        }
        
        # 加密消息
        encoded = json.dumps(message_data).encode()
        encrypted = self.security.encrypt_message(encoded)
        
        # 创建响应future
        response_future = asyncio.Future()
        with self._lock:
            self.response_futures[message_id] = response_future
            
        try:
            # 发送消息
            await self.message_queue.put((target_rank, encrypted))
            
            # 等待响应
            response = await asyncio.wait_for(response_future, timeout)
            return response
        finally:
            with self._lock:
                self.response_futures.pop(message_id, None)
                
    async def handle_message(self, rank: int, encrypted_message: bytes) -> None:
        """处理接收到的消息"""
        try:
            # 解密消息
            decrypted = self.security.decrypt_message(encrypted_message)
            message_data = json.loads(decrypted.decode())
            
            # 处理响应
            if 'response_to' in message_data:
                message_id = message_data['response_to']
                with self._lock:
                    if message_id in self.response_futures:
                        self.response_futures[message_id].set_result(
                            message_data['content']
                        )
                return
                
            # 处理新消息
            response = await self._process_message(message_data['content'])
            
            # 发送响应
            response_data = {
                'response_to': message_data['id'],
                'content': response
            }
            
            # 加密响应
            encoded = json.dumps(response_data).encode()
            encrypted = self.security.encrypt_message(encoded)
            
            await self.message_queue.put((rank, encrypted))
            
        except Exception as e:
            logging.error(f"消息处理错误: {str(e)}")
            
    async def _process_message(self, message: Any) -> Any:
        """处理消息内容"""
        # 在子类中实现具体的消息处理逻辑
        raise NotImplementedError

class DistributedManager:
    """分布式训练管理器"""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        backend: str = 'nccl',
        timeout: float = 30.0,
        max_retries: int = 3,
        checkpoint_freq: int = 100,
        enable_backup_masters: bool = True,
        encryption_key: Optional[bytes] = None
    ):
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.timeout = timeout
        self.max_retries = max_retries
        self.checkpoint_freq = checkpoint_freq
        self.enable_backup_masters = enable_backup_masters
        
        # 初始化组件
        self.security = SecurityManager()
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
            'memory_util': deque(maxlen=1000)
        }
        
        # 故障恢复状态
        self.recovery_state = {
            'in_recovery': False,
            'failed_nodes': set(),
            'recovery_start': None,
            'checkpoint_path': None
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
            self._start_monitoring()
            
            # 如果是主节点，启动备份主节点选举
            if self.rank == 0 and self.enable_backup_masters:
                self._elect_backup_masters()
                
        except Exception as e:
            self._handle_init_failure(e)
            
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
            dist.broadcast(data, 0)
            
            # 计算带宽
            elapsed = time.time() - start_time
            bytes_sent = data.numel() * data.element_size()
            return bytes_sent / elapsed
        else:
            data = torch.empty(1024, 1024, device='cuda')
            dist.broadcast(data, 0)
            return 0.0
            
    def _elect_backup_masters(self):
        """选举备份主节点"""
        # 选择资源最丰富的节点作为备份主节点
        candidates = []
        for rank, info in self.node_states.items():
            if rank == 0:  # 跳过主节点
                continue
            candidates.append((rank, info.resources))
            
        # 按资源丰富程度排序
        candidates.sort(
            key=lambda x: (
                x[1].gpu_count,
                sum(x[1].gpu_memory),
                x[1].cpu_count,
                x[1].memory_total
            ),
            reverse=True
        )
        
        # 选择前两个节点作为备份主节点
        self.backup_masters = {
            candidates[i][0] for i in range(min(2, len(candidates)))
        }
        
        # 更新节点角色
        for rank in self.backup_masters:
            self.node_states[rank].role = NodeRole.BACKUP_MASTER
            
    async def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                # 检查节点状态
                self._check_node_health()
                
                # 更新负载均衡
                self._update_load_balance()
                
                # 检查是否需要重新平衡
                if self.load_balancer.should_rebalance():
                    await self._rebalance_load()
                    
                # 更新性能指标
                self._update_performance_metrics()
                
                # 如果是主节点，额外检查备份主节点状态
                if self.rank == 0:
                    self._check_backup_masters()
                    
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"监控异常: {str(e)}")
                
    def _check_node_health(self):
        """检查节点健康状态"""
        # 检查心跳超时
        timed_out = self.heartbeat.check_timeouts()
        
        for rank in timed_out:
            if rank in self.node_states:
                node = self.node_states[rank]
                if node.status == NodeStatus.ACTIVE:
                    self._handle_node_failure(rank)
                    
    async def _rebalance_load(self):
        """重新平衡负载"""
        if self.rank != 0:  # 只有主节点可以触发重平衡
            return
            
        # 获取重平衡计划
        plan = self.load_balancer.get_rebalance_plan()
        
        for source, target in plan:
            try:
                # 通知源节点和目标节点
                await asyncio.gather(
                    self.communicator.send_message(
                        source,
                        {
                            'type': 'prepare_transfer',
                            'target': target
                        }
                    ),
                    self.communicator.send_message(
                        target,
                        {
                            'type': 'prepare_receive',
                            'source': source
                        }
                    )
                )
                
                # 等待传输完成
                await asyncio.gather(
                    self.communicator.send_message(
                        source,
                        {'type': 'confirm_transfer'}
                    ),
                    self.communicator.send_message(
                        target,
                        {'type': 'confirm_receive'}
                    )
                )
                
            except Exception as e:
                logging.error(f"负载重平衡失败: {str(e)}")
                
    def _handle_node_failure(self, rank: int):
        """处理节点故障"""
        node = self.node_states[rank]
        node.status = NodeStatus.FAILED
        
        # 如果是备份主节点失败，需要重新选举
        if node.role == NodeRole.BACKUP_MASTER:
            self.backup_masters.remove(rank)
            if self.rank == 0:
                self._elect_backup_masters()
                
        # 将失败节点的工作重新分配
        self._redistribute_work(rank)
        
        # 启动恢复流程
        self.recovery_state['failed_nodes'].add(rank)
        if not self.recovery_state['in_recovery']:
            self._initiate_recovery(rank)
            
    def _redistribute_work(self, failed_rank: int):
        """重新分配失败节点的工作"""
        # 获取活跃节点
        active_nodes = [
            rank for rank, info in self.node_states.items()
            if info.status == NodeStatus.ACTIVE and rank != failed_rank
        ]
        
        if not active_nodes:
            raise RuntimeError("没有可用的活跃节点")
            
        # 根据负载情况分配工作
        target_rank = self.load_balancer.get_optimal_target(0)  # TODO: 计算工作量
        
        # 通知目标节点接管工作
        self.thread_pool.submit(
            self._transfer_work,
            failed_rank,
            target_rank
        )
        
    async def _transfer_work(self, source_rank: int, target_rank: int):
        """转移工作负载"""
        try:
            # 通知目标节点准备接收
            await self.communicator.send_message(
                target_rank,
                {
                    'type': 'take_over',
                    'source_rank': source_rank
                }
            )
            
            # 等待确认
            response = await self.communicator.send_message(
                target_rank,
                {'type': 'confirm_takeover'}
            )
            
            if response.get('status') == 'success':
                logging.info(f"成功将节点 {source_rank} 的工作转移到节点 {target_rank}")
            else:
                raise RuntimeError(f"工作转移失败: {response.get('error')}")
                
        except Exception as e:
            logging.error(f"工作转移失败: {str(e)}")
            
    def _initiate_recovery(self, failed_rank: int):
        """启动故障恢复流程"""
        self.recovery_state['in_recovery'] = True
        self.recovery_state['recovery_start'] = time.time()
        
        # 如果是主节点失败，启动主节点切换
        if failed_rank == 0:
            self._switch_master()
        else:
            # 否则尝试恢复节点
            self.thread_pool.submit(self._recover_node, failed_rank)
            
    def _switch_master(self):
        """切换主节点"""
        if not self.backup_masters:
            raise RuntimeError("没有可用的备份主节点")
            
        # 选择第一个备份主节点作为新主节点
        new_master = min(self.backup_masters)
        
        try:
            # 通知所有节点切换主节点
            for rank in range(self.world_size):
                if rank != new_master and self.node_states[rank].status == NodeStatus.ACTIVE:
                    dist.rpc_async(
                        f"worker_{rank}",
                        self._notify_master_switch,
                        args=(new_master,)
                    )
                    
            # 更新节点角色
            self.node_states[new_master].role = NodeRole.MASTER
            self.backup_masters.remove(new_master)
            
            # 重新选举备份主节点
            self._elect_backup_masters()
            
        except Exception as e:
            logging.error(f"主节点切换失败: {str(e)}")
            
    async def _recover_node(self, rank: int):
        """恢复失败的节点"""
        try:
            # 检查节点是否可以重连
            if await self._check_node_connectivity(rank):
                # 同步状态
                await self._sync_node_state(rank)
                
                # 恢复节点状态
                self.node_states[rank].status = NodeStatus.ACTIVE
                self.recovery_state['failed_nodes'].remove(rank)
                
                logging.info(f"节点 {rank} 恢复成功")
            else:
                logging.error(f"节点 {rank} 无法恢复")
                
        except Exception as e:
            logging.error(f"节点恢复失败: {str(e)}")
            
        finally:
            if not self.recovery_state['failed_nodes']:
                self.recovery_state['in_recovery'] = False
                
    async def _check_node_connectivity(self, rank: int) -> bool:
        """检查节点连接性"""
        try:
            # 尝试建立连接
            response = await self.communicator.send_message(
                rank,
                {'type': 'connectivity_check'},
                timeout=5.0
            )
            return response.get('status') == 'ok'
        except Exception:
            return False
            
    async def _sync_node_state(self, rank: int):
        """同步节点状态"""
        try:
            # 获取最新的全局状态
            global_state = self._get_global_state()
            
            # 发送同步请求
            await self.communicator.send_message(
                rank,
                {
                    'type': 'sync_state',
                    'state': global_state
                }
            )
            
            # 等待同步完成
            response = await self.communicator.send_message(
                rank,
                {'type': 'confirm_sync'}
            )
            
            if response.get('status') != 'success':
                raise RuntimeError(f"状态同步失败: {response.get('error')}")
                
        except Exception as e:
            logging.error(f"状态同步失败: {str(e)}")
            raise
            
    def _get_global_state(self) -> Dict[str, Any]:
        """获取全局状态"""
        return {
            'node_states': self.node_states,
            'backup_masters': list(self.backup_masters),
            'performance_metrics': self.performance_metrics,
            'load_balancer_state': {
                'node_loads': self.load_balancer.node_loads,
                'resource_usage': self.load_balancer.resource_usage
            }
        }
        
    def get_status(self) -> Dict[str, Any]:
        """获取分布式系统状态"""
        return {
            'node_states': {
                rank: {
                    'role': info.role.value,
                    'status': info.status.value,
                    'last_heartbeat': info.last_heartbeat,
                    'performance_metrics': info.performance_metrics
                }
                for rank, info in self.node_states.items()
            },
            'backup_masters': list(self.backup_masters),
            'recovery_state': self.recovery_state,
            'performance_metrics': {
                k: list(v) for k, v in self.performance_metrics.items()
            }
        }
        
    def __del__(self):
        """清理资源"""
        self.thread_pool.shutdown()

def create_distributed_manager(world_size: int) -> DistributedManager:
    """创建分布式训练管理器"""
    return DistributedManager(world_size) 