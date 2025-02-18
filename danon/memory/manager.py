"""
智能内存管理系统
实现高效的内存分配、回收和优化策略
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from collections import OrderedDict
import weakref
import gc
import psutil
import numpy as np
from threading import Lock
import logging
import time
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .cache import FastCache
from .storage import LongTermStorage
from .working import WorkingMemory

class MemoryPriority(Enum):
    """内存优先级"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class MemoryBlock:
    """内存块"""
    tensor: torch.Tensor
    priority: MemoryPriority
    last_access: float
    access_count: int
    is_pinned: bool = False
    metadata: Dict[str, Any] = None

class MemoryConfig:
    """内存管理配置"""
    def __init__(
        self,
        small_size_threshold: int = 1024 * 1024,  # 1MB
        medium_size_threshold: int = 10 * 1024 * 1024,  # 10MB
        pool_size: int = 100,
        cleanup_threshold: float = 0.8,
        defrag_threshold: float = 0.3,
        enable_auto_optimization: bool = True,
        cache_size: int = 1000,
        device: str = "cuda",
        enable_cache_compression: bool = True,
        cache_dtype: torch.dtype = torch.float32,
        optimize_memory_usage: bool = True,
        use_gradient_checkpointing: bool = True,
        use_mixed_precision: bool = True,
        enable_tensor_fusion: bool = True
    ):
        self.small_size_threshold = small_size_threshold
        self.medium_size_threshold = medium_size_threshold
        self.pool_size = pool_size
        self.cleanup_threshold = cleanup_threshold
        self.defrag_threshold = defrag_threshold
        self.enable_auto_optimization = enable_auto_optimization
        self.cache_size = cache_size
        self.device = device
        self.enable_cache_compression = enable_cache_compression
        self.cache_dtype = cache_dtype
        self.optimize_memory_usage = optimize_memory_usage
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.enable_tensor_fusion = enable_tensor_fusion

class TensorPool:
    """智能张量池：管理不同大小的张量"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pools = {
            'small': defaultdict(list),   # <1MB
            'medium': defaultdict(list),  # 1-10MB
            'large': defaultdict(list)    # >10MB
        }
        self.stats = defaultdict(int)
        self.references = weakref.WeakKeyDictionary()
        
    def _get_size_category(self, tensor: torch.Tensor) -> str:
        """确定张量的大小类别"""
        size = tensor.numel() * tensor.element_size()
        if size < self.config.small_size_threshold:
            return 'small'
        elif size < self.config.medium_size_threshold:
            return 'medium'
        return 'large'
        
    def _get_tensor_key(self, tensor: torch.Tensor) -> tuple:
        """生成张量的唯一键"""
        return (tensor.shape, tensor.dtype, tensor.device)
        
    def acquire(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        """从池中获取张量"""
        key = (shape, dtype, device)
        mock_tensor = torch.empty(shape, dtype=dtype, device=device)
        category = self._get_size_category(mock_tensor)
        
        if self.pools[category][key]:
            tensor = self.pools[category][key].pop()
            self.stats['reuse_count'] += 1
            return tensor
            
        self.stats['new_alloc_count'] += 1
        return None
        
    def release(self, tensor: torch.Tensor):
        """将张量返回到池中"""
        if tensor is None:
            return
            
        key = self._get_tensor_key(tensor)
        category = self._get_size_category(tensor)
        
        # 检查池大小限制
        if len(self.pools[category][key]) < self.config.pool_size:
            # 清除计算图
            tensor.detach_()
            # 重置内存
            tensor.zero_()
            self.pools[category][key].append(tensor)
            self.stats['release_count'] += 1
            
    def cleanup(self):
        """清理未使用的张量"""
        for category in self.pools:
            for key in list(self.pools[category].keys()):
                pool = self.pools[category][key]
                # 仅保留部分张量
                keep_count = int(len(pool) * (1 - self.config.cleanup_threshold))
                if keep_count < len(pool):
                    self.pools[category][key] = pool[:keep_count]
                    self.stats['cleanup_count'] += len(pool) - keep_count
                    
    def get_stats(self) -> Dict[str, int]:
        """获取使用统计"""
        return dict(self.stats)

class MemoryOptimizer:
    """内存优化器：实现内存使用优化策略"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.fragmentation_history = []
        self.peak_memory_history = []
        
    def analyze_fragmentation(self) -> float:
        """分析内存碎片率"""
        allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        if reserved == 0:
            return 0.0
        return 1.0 - (allocated / reserved)
        
    def should_defrag(self) -> bool:
        """判断是否需要进行碎片整理"""
        frag_rate = self.analyze_fragmentation()
        self.fragmentation_history.append(frag_rate)
        
        # 保持历史记录在合理范围内
        if len(self.fragmentation_history) > 100:
            self.fragmentation_history.pop(0)
            
        # 当碎片率超过阈值时进行整理
        return frag_rate > self.config.defrag_threshold
        
    def defrag(self):
        """执行内存碎片整理"""
        if torch.cuda.is_available():
            # 强制进行垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            
        # 记录峰值内存使用
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        self.peak_memory_history.append(peak_memory)
        
        # 保持历史记录在合理范围内
        if len(self.peak_memory_history) > 100:
            self.peak_memory_history.pop(0)
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'avg_fragmentation': sum(self.fragmentation_history) / len(self.fragmentation_history) if self.fragmentation_history else 0,
            'peak_memory': max(self.peak_memory_history) if self.peak_memory_history else 0,
            'defrag_count': len(self.fragmentation_history)
        }

class TensorCache:
    """高效的张量缓存系统"""
    def __init__(self, max_size: int = 1000, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self._cache: OrderedDict = OrderedDict()
        self._size = 0
        self._lock = Lock()
        
    def __len__(self) -> int:
        return len(self._cache)
        
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._size = 0
            
    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取缓存的张量"""
        with self._lock:
            if key in self._cache:
                tensor = self._cache.pop(key)
                self._cache[key] = tensor  # 移动到最新位置
                return tensor
        return None
        
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """存储张量到缓存"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 移除最旧的项
                _, old_tensor = self._cache.popitem(last=False)
                self._size -= old_tensor.numel() * old_tensor.element_size()
                
            self._cache[key] = tensor
            self._size += tensor.numel() * tensor.element_size()

class SmartCache:
    """智能缓存系统"""
    def __init__(self, capacity: int, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.cache: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.priority_queues: Dict[MemoryPriority, Set[str]] = {
            priority: set() for priority in MemoryPriority
        }
        self._lock = Lock()
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取缓存项"""
        with self._lock:
            if key in self.cache:
                block = self.cache[key]
                block.last_access = time.time()
                block.access_count += 1
                self.cache.move_to_end(key)
                return block.tensor
        return None
        
    def put(self, key: str, tensor: torch.Tensor, priority: MemoryPriority = MemoryPriority.MEDIUM):
        """存入缓存项"""
        with self._lock:
            if key in self.cache:
                old_block = self.cache[key]
                self.priority_queues[old_block.priority].remove(key)
                
            # 如果缓存已满，移除低优先级项
            while len(self.cache) >= self.capacity:
                self._evict_one()
                
            block = MemoryBlock(
                tensor=tensor,
                priority=priority,
                last_access=time.time(),
                access_count=1
            )
            
            self.cache[key] = block
            self.priority_queues[priority].add(key)
            
    def _evict_one(self) -> None:
        """驱逐一个缓存项"""
        # 从最低优先级开始查找可驱逐的项
        for priority in MemoryPriority:
            candidates = self.priority_queues[priority]
            if not candidates:
                continue
                
            # 找出最久未使用的非固定项
            lru_key = None
            lru_time = float('inf')
            
            for key in candidates:
                block = self.cache[key]
                if not block.is_pinned and block.last_access < lru_time:
                    lru_key = key
                    lru_time = block.last_access
                    
            if lru_key:
                del self.cache[lru_key]
                candidates.remove(lru_key)
                return
                
        # 如果所有项都被固定，抛出异常
        raise RuntimeError("无法驱逐缓存项：所有项都被固定")

class MemoryAnalyzer:
    """内存分析器"""
    def __init__(self):
        self.allocation_history: List[Dict[str, Any]] = []
        self.peak_memory = 0
        self.fragmentation_ratio = 0.0
        self.tensor_sizes: Dict[str, int] = {}
        self._lock = Lock()
        
    def record_allocation(self, tensor: torch.Tensor, source: str):
        """记录内存分配"""
        with self._lock:
            size = tensor.numel() * tensor.element_size()
            self.allocation_history.append({
                'time': time.time(),
                'size': size,
                'source': source,
                'shape': tensor.shape,
                'dtype': tensor.dtype
            })
            
            self.tensor_sizes[id(tensor)] = size
            self.peak_memory = max(self.peak_memory, self._get_current_memory())
            
    def record_deallocation(self, tensor_id: int):
        """记录内存释放"""
        with self._lock:
            if tensor_id in self.tensor_sizes:
                del self.tensor_sizes[tensor_id]
                
    def analyze_fragmentation(self) -> float:
        """分析内存碎片率"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved == 0:
                return 0.0
            self.fragmentation_ratio = 1.0 - (allocated / reserved)
            return self.fragmentation_ratio
        return 0.0
        
    def get_memory_profile(self) -> Dict[str, Any]:
        """获取内存使用分析"""
        with self._lock:
            current_memory = self._get_current_memory()
            allocation_patterns = self._analyze_allocation_patterns()
            
            return {
                'current_memory': current_memory,
                'peak_memory': self.peak_memory,
                'fragmentation_ratio': self.fragmentation_ratio,
                'allocation_patterns': allocation_patterns,
                'tensor_count': len(self.tensor_sizes),
                'allocation_history': self.allocation_history[-100:]  # 最近100条记录
            }
            
    def _get_current_memory(self) -> int:
        """获取当前内存使用量"""
        return sum(self.tensor_sizes.values())
        
    def _analyze_allocation_patterns(self) -> Dict[str, Any]:
        """分析内存分配模式"""
        if not self.allocation_history:
            return {}
            
        sizes = [record['size'] for record in self.allocation_history]
        return {
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'max_size': max(sizes),
            'min_size': min(sizes),
            'total_allocations': len(self.allocation_history)
        }

class EnhancedMemoryManager:
    """增强型内存管理器"""
    def __init__(
        self,
        cache_size: int = 1000,
        enable_monitoring: bool = True,
        device: str = "cuda",
        num_workers: int = 4
    ):
        self.device = device
        self.cache = SmartCache(cache_size, device)
        self.analyzer = MemoryAnalyzer()
        self.enable_monitoring = enable_monitoring
        
        # 线程池用于异步操作
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        # 监控线程
        self._monitor_thread = None
        self._should_monitor = False
        self._lock = Lock()
        
        if enable_monitoring:
            self.start_monitoring()
            
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        name: Optional[str] = None
    ) -> torch.Tensor:
        """分配内存"""
        with self._lock:
            # 检查缓存
            if name and (cached := self.cache.get(name)) is not None:
                return cached
                
            # 分配新张量
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            
            # 记录分配
            self.analyzer.record_allocation(tensor, name or 'unnamed')
            
            # 如果指定了名称，加入缓存
            if name:
                self.cache.put(name, tensor, priority)
                
            return tensor
            
    def free(self, tensor: torch.Tensor):
        """释放内存"""
        self.analyzer.record_deallocation(id(tensor))
        del tensor
        
    def start_monitoring(self):
        """启动监控"""
        if self.enable_monitoring and not self._monitor_thread:
            self._should_monitor = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.start()
            
    def stop_monitoring(self):
        """停止监控"""
        self._should_monitor = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
            
    def _monitor_loop(self):
        """监控循环"""
        while self._should_monitor:
            try:
                # 分析内存使用
                self.analyzer.analyze_fragmentation()
                
                # 如果碎片率过高，触发整理
                if self.analyzer.fragmentation_ratio > 0.3:
                    self.thread_pool.submit(self._defragment)
                    
                # 检查内存压力
                if self._check_memory_pressure():
                    self.thread_pool.submit(self._handle_memory_pressure)
                    
                time.sleep(1)  # 监控间隔
            except Exception as e:
                logging.error(f"监控异常: {str(e)}")
                
    def _check_memory_pressure(self) -> bool:
        """检查内存压力"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total > 0.9
        return False
        
    def _handle_memory_pressure(self):
        """处理内存压力"""
        # 清理缓存中的低优先级项
        with self._lock:
            for priority in [MemoryPriority.LOW, MemoryPriority.MEDIUM]:
                for key in list(self.cache.priority_queues[priority]):
                    self.cache._evict_one()
                    
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _defragment(self):
        """内存碎片整理"""
        with self._lock:
            # 获取所有缓存的张量
            tensors = [(key, block.tensor) for key, block in self.cache.cache.items()]
            
            # 清空缓存
            self.cache.cache.clear()
            for queue in self.cache.priority_queues.values():
                queue.clear()
                
            # 重新分配张量
            for key, tensor in tensors:
                new_tensor = torch.empty_like(tensor)
                new_tensor.copy_(tensor)
                self.cache.put(key, new_tensor)
                
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return {
            'memory_profile': self.analyzer.get_memory_profile(),
            'cache_size': len(self.cache.cache),
            'device': self.device
        }
        
    def __del__(self):
        """清理资源"""
        self.stop_monitoring()
        self.thread_pool.shutdown()

class MemoryManager:
    """内存管理器"""
    def __init__(self, config: Any):
        self.config = config
        self.tensor_pool = TensorPool(self.config)
        self.optimizer = MemoryOptimizer(self.config)
        self.allocation_hooks = []
        self.tensor_cache = TensorCache(
            max_size=config.cache_size,
            device=config.device
        )
        self._temp_tensors: Dict[int, weakref.ref] = {}
        self._peak_memory = 0
        self._current_memory = 0
        
    def register_allocation_hook(self, hook):
        """注册内存分配钩子"""
        self.allocation_hooks.append(hook)
        
    def allocate(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """分配张量"""
        # 尝试从池中获取
        tensor = self.tensor_pool.acquire(shape, dtype, device)
        
        if tensor is None:
            # 创建新张量
            tensor = torch.empty(shape, dtype=dtype, device=device)
            
        # 调用分配钩子
        for hook in self.allocation_hooks:
            hook(tensor)
            
        return tensor
        
    def free(self, tensor: torch.Tensor):
        """释放张量"""
        self.tensor_pool.release(tensor)
        
        # 检查是否需要优化
        if self.config.enable_auto_optimization:
            if self.optimizer.should_defrag():
                self.optimizer.defrag()
                
    def cleanup(self):
        """清理未使用的内存"""
        self.tensor_pool.cleanup()
        self.optimizer.defrag()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        stats = {
            'cache_size': len(self.tensor_cache),
            'peak_memory': self._peak_memory,
            'current_memory': self._current_memory
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated': torch.cuda.memory_allocated(),
                'gpu_cached': torch.cuda.memory_reserved(),
                'gpu_max_allocated': torch.cuda.max_memory_allocated()
            })
            
        # 系统内存统计
        process = psutil.Process()
        stats['system_memory'] = process.memory_info().rss
        
        return stats
        
    def check_memory_pressure(self) -> bool:
        """检查是否存在内存压力"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total > 0.9
        return False
        
    def apply_memory_optimizations(self, model: nn.Module) -> None:
        """应用内存优化策略"""
        if not self.config.optimize_memory_usage:
            return
            
        # 启用梯度检查点
        if self.config.use_gradient_checkpointing:
            model.enable_gradient_checkpointing()
            
        # 使用混合精度训练
        if self.config.use_mixed_precision:
            model = model.to(dtype=torch.float16)
            
        # 启用张量融合
        if self.config.enable_tensor_fusion:
            self._fuse_model_tensors(model)
            
    def _fuse_model_tensors(self, model: nn.Module) -> None:
        """融合模型中的张量以减少内存碎片"""
        fusible_tensors = []
        
        # 收集可融合的张量
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                fusible_tensors.append((name, param))
                
        if not fusible_tensors:
            return
            
        # 按形状分组
        shape_groups: Dict[Tuple[int, ...], List[torch.Tensor]] = {}
        for _, param in fusible_tensors:
            shape = tuple(param.shape)
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append(param)
            
        # 对每组进行融合
        for shape, tensors in shape_groups.items():
            if len(tensors) > 1:
                fused = torch.cat(tensors, dim=0)
                # 将融合后的张量分配回原始参数
                offset = 0
                for tensor in tensors:
                    tensor.data = fused[offset:offset + tensor.size(0)]
                    offset += tensor.size(0)
                    
    def __del__(self):
        """清理资源"""
        self.optimize_memory()

class MemoryManager(nn.Module):
    """记忆管理器，协调不同层次的记忆系统"""
    
    def __init__(
        self,
        dim: int = 512,
        cache_size: int = 1024,
        storage_size: int = 1000000,
        working_memory_size: int = 100,
        storage_path: Optional[str] = None,
        enable_adaptive_optimization: bool = True,
        min_cache_hit_rate: float = 0.6,
        max_memory_pressure: float = 0.85
    ):
        super().__init__()
        self.dim = dim
        self.enable_adaptive_optimization = enable_adaptive_optimization
        self.min_cache_hit_rate = min_cache_hit_rate
        self.max_memory_pressure = max_memory_pressure
        
        # 初始化各个记忆层
        self.fast_cache = FastCache(
            cache_size=cache_size,
            dim=dim
        )
        
        self.long_term_storage = LongTermStorage(
            dim=dim,
            storage_size=storage_size,
            storage_path=storage_path
        )
        
        self.working_memory = WorkingMemory(
            dim=dim,
            max_items=working_memory_size
        )
        
        # 记忆路由网络
        self.memory_router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 3),  # 3个记忆层的路由概率
            nn.Softmax(dim=-1)
        )
        
        # 记忆融合网络
        self.memory_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

        # 自适应优化器
        self.adaptive_optimizer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),  # 优化策略选择
            nn.Softmax(dim=-1)
        )
        
        # 性能监控
        self.performance_stats = {
            'cache_hit_rate': [],
            'memory_pressure': [],
            'optimization_decisions': []
        }
        
    def _optimize_memory_allocation(self) -> None:
        """优化内存分配策略"""
        if not self.enable_adaptive_optimization:
            return
            
        # 计算当前性能指标
        cache_stats = self.fast_cache.get_stats()
        current_hit_rate = cache_stats.get('hit_rate', 0)
        memory_stats = self.get_memory_stats()
        current_memory_pressure = memory_stats.get('memory_pressure', 0)
        
        # 根据性能指标调整策略
        if current_hit_rate < self.min_cache_hit_rate:
            # 增加缓存大小
            self.fast_cache.resize(int(self.fast_cache.cache_size * 1.2))
            
        if current_memory_pressure > self.max_memory_pressure:
            # 触发内存优化
            self.optimize_memory_usage()
            
        # 记录性能数据
        self.performance_stats['cache_hit_rate'].append(current_hit_rate)
        self.performance_stats['memory_pressure'].append(current_memory_pressure)
        
    def optimize_memory_usage(self) -> None:
        """优化内存使用"""
        # 清理不活跃的缓存项
        self.fast_cache.cleanup_inactive()
        
        # 将不常用的工作记忆项移动到长期存储
        working_items = self.working_memory.get_all_memories()
        for item in working_items:
            if item.importance < 0.3:  # 重要性阈值
                self.long_term_storage.store(
                    item.key,
                    item.value,
                    item.metadata
                )
                self.working_memory.remove_item(item)
                
        # 整理长期存储
        self.long_term_storage.optimize()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存系统的统计信息"""
        stats = {
            'cache': self.fast_cache.get_stats(),
            'storage': self.long_term_storage.get_stats(),
            'working_memory': self.working_memory.get_memory_state(),
            'performance_history': self.performance_stats
        }
        
        # 计算内存压力
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            stats['memory_pressure'] = allocated_memory / total_memory
        else:
            stats['memory_pressure'] = psutil.virtual_memory().percent / 100
            
        return stats
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，包含自适应优化"""
        # 运行内存优化
        self._optimize_memory_allocation()
        
        # 原有的前向传播逻辑
        route_probs = self._route_memory(x)
        
        # 从各个记忆层获取结果
        cache_result = self.fast_cache(x)
        storage_result = self.long_term_storage(x)
        working_result = self.working_memory(x)
        
        # 融合结果
        combined = torch.cat([
            cache_result * route_probs[:, 0:1],
            storage_result * route_probs[:, 1:2],
            working_result * route_probs[:, 2:3]
        ], dim=-1)
        
        return self.memory_fusion(combined)
        
    def _route_memory(self, x: torch.Tensor) -> torch.Tensor:
        """决定记忆应该存储在哪个层次"""
        return self.memory_router(x)
        
    def store(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        存储记忆到适当的层次
        
        Args:
            x: 输入张量
            value: 要存储的值
            metadata: 可选的元数据
            
        Returns:
            info: 存储操作的相关信息
        """
        # 获取路由概率
        route_probs = self._route_memory(x)
        
        storage_info = {}
        
        # 存储到各个层次
        if route_probs[0] > 0.3:  # 快速缓存阈值
            cache_key = self.fast_cache.store(x, value, metadata)
            storage_info['cache_key'] = cache_key
            
        if route_probs[1] > 0.3:  # 长期存储阈值
            storage_idx = self.long_term_storage.store(x, value, metadata)
            storage_info['storage_index'] = storage_idx
            
        if route_probs[2] > 0.3:  # 工作记忆阈值
            self.working_memory.add_memory(x, value, metadata)
            
        storage_info['route_probabilities'] = route_probs.tolist()
        return storage_info
        
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        从所有记忆层检索相关信息
        
        Args:
            query: 查询向量
            k: 每层返回的结果数量
            threshold: 相似度阈值
            
        Returns:
            merged: 融合后的记忆表示
            info: 检索操作的相关信息
        """
        results = []
        retrieval_info = {}
        
        # 从快速缓存检索
        cache_value, cache_hit = self.fast_cache.retrieve(query)
        if cache_hit:
            results.append(cache_value)
            retrieval_info['cache_hit'] = True
            
        # 从长期存储检索
        storage_values, similarities, storage_metadata = self.long_term_storage.retrieve(
            query, k=k, threshold=threshold
        )
        if len(storage_values) > 0:
            results.append(storage_values.mean(0))
            retrieval_info['storage_results'] = len(storage_values)
            
        # 从工作记忆检索
        working_results = self.working_memory.query_memory(
            query, k=k, threshold=threshold
        )
        if working_results:
            working_values = torch.stack([r[0] for r in working_results])
            results.append(working_values.mean(0))
            retrieval_info['working_memory_results'] = len(working_results)
            
        # 如果没有找到任何结果
        if not results:
            return query, {'found': False}
            
        # 融合不同层次的记忆
        if len(results) == 1:
            merged = results[0]
        else:
            # 填充到相同维度
            while len(results) < 3:
                results.append(torch.zeros_like(results[0]))
            merged = self.memory_fusion(torch.cat(results, dim=-1))
            
        retrieval_info['found'] = True
        retrieval_info['num_sources'] = len(results)
        
        return merged, retrieval_info
        
    def update(self) -> None:
        """更新记忆系统的状态"""
        # 更新工作记忆的重要性分数
        self.working_memory.update_importance()
        
        # 将长期未使用的缓存项移动到长期存储
        # TODO: 实现缓存到长期存储的迁移策略
        
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统的统计信息"""
        return {
            'cache': self.fast_cache.get_stats(),
            'storage': self.long_term_storage.get_stats(),
            'working_memory': self.working_memory.get_memory_state()
        }
        
    def save_state(self) -> None:
        """保存记忆系统的状态"""
        self.long_term_storage.save_storage()
        
    def load_state(self) -> None:
        """加载记忆系统的状态"""
        self.long_term_storage.load_storage()
        
    def clear(self) -> None:
        """清空所有记忆"""
        self.fast_cache.clear()
        self.long_term_storage.clear()
        self.working_memory.clear() 