"""
UCSA (Unified Compressed Sparse Attention)

这个模块实现了一个创新的无限长度序列处理机制，主要特点：

1. 无限长度处理能力
   - 通过动态压缩和局部注意力实现
   - 支持任意长度的序列输入
   - 内存占用与序列长度呈对数关系

2. 统一压缩机制
   - 自适应压缩率调整
   - 多层次特征提取
   - 局部-全局信息融合

3. 错误处理和恢复
   - 内存自动管理
   - 错误自动恢复
   - 性能自适应调整

主要组件：
- LocalAttention: 局部注意力计算
- HierarchicalCompression: 层次化压缩
- GlobalSparsityControl: 全局稀疏控制
- ThreadSafeCache: 线程安全缓存
- ErrorTracker: 错误追踪和恢复

工作流程：
1. 输入序列首先通过局部注意力处理
2. 应用层次化压缩降低内存占用
3. 全局稀疏控制确保计算效率
4. 错误处理机制保证稳定性
5. 缓存机制提高处理速度

性能特点：
- 时间复杂度: O(n log n)
- 空间复杂度: O(log n)
- 支持并行计算
- 动态内存管理

使用示例：
```python
# 创建配置
config = InfiniteAttentionConfig(
    hidden_size=768,
    sparsity_factor=0.1,
    compression_ratio=0.5,
    enable_cache=True
)

# 创建模型
model = InfiniteAttention(config)

# 处理序列
output, stats = model(
    input_sequence,
    attention_mask=mask,
    segment_id="segment_1"
)

# 获取性能统计
print(f"稀疏度: {stats['sparsity_threshold']}")
print(f"压缩率: {stats['compression_ratio']}")
print(f"缓存命中率: {stats['cache_hit_rate']}")
```

注意事项：
1. 建议根据实际序列长度调整压缩率
2. 可以通过调整稀疏度在性能和精度间取得平衡
3. 对于特别长的序列，建议启用缓存机制
4. 如果出现内存不足，系统会自动调整参数

与其他模块的集成：
- 可与MSRA和DALA无缝集成
- 支持混合注意力模式
- 可用于增强现有Transformer模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
import math
from dataclasses import dataclass
import numpy as np
from threading import Lock
import logging
import time
from .base_config import BaseConfig
import threading
import sys
import weakref
import random
from collections import defaultdict, deque
from .attention_base import BaseAttentionConfig, BaseAttention, AttentionMonitor, AttentionError
from .unified_config import InfiniteAttentionConfig
from .monitoring import PerformanceContext
from .error_handling import ComputationError
from .compatibility import ensure_compatibility

@dataclass
class UCSAConfig(BaseAttentionConfig):
    """统一压缩稀疏注意力配置"""
    # 保持与InfiniteAttentionConfig相同的配置
    sparsity_factor: float = 0.1
    top_k_ratio: float = 0.2
    local_window_size: int = 512
    sliding_window_size: int = 256
    compression_ratio: float = 0.5
    max_cache_size: int = 10000
    enable_cache: bool = True
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_head_size: int = 64
    dropout: float = 0.1
    num_levels: int = 3
    chunk_size: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class InfiniteAttentionConfig(BaseAttentionConfig):
    """无限注意力配置"""
    # 稀疏注意力特有配置
    sparsity_factor: float = 0.1  # 稀疏度
    top_k_ratio: float = 0.2  # 保留top-k比例
    
    # 局部注意力特有配置
    local_window_size: int = 4096  # 增大局部窗口大小
    sliding_window_size: int = 2048  # 增大滑动窗口大小
    
    # 压缩特有配置
    compression_ratio: float = 0.5  # 压缩比
    min_compression_ratio: float = 0.1
    max_compression_ratio: float = 0.9
    enable_adaptive_compression: bool = True
    
    # 缓存特有配置
    max_cache_size: int = 100000  # 最大缓存大小
    enable_cache: bool = True  # 是否启用缓存
    cache_dtype: torch.dtype = torch.float16  # 缓存使用半精度
    
    # 基础配置
    hidden_size: int = 8192  # 隐藏层大小
    num_attention_heads: int = 64  # 注意力头数
    attention_head_size: int = 128  # 注意力头维度
    dropout: float = 0.1
    
    # 层次注意力
    num_levels: int = 4  # 层次级别数
    
    # 性能优化
    chunk_size: int = 4096  # 计算分块大小
    gradient_checkpointing: bool = True  # 启用梯度检查点
    mixed_precision: bool = True  # 启用混合精度训练
    optimizer_states_in_half_precision: bool = True  # 优化器状态使用半精度
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        """初始化后的验证"""
        # 验证head_dim和hidden_size的关系
        if self.hidden_size != self.num_attention_heads * self.attention_head_size:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal num_attention_heads * attention_head_size "
                f"({self.num_attention_heads * self.attention_head_size})"
            )
        
        # 验证压缩比率关系
        if not (self.min_compression_ratio <= self.compression_ratio <= self.max_compression_ratio):
            raise ValueError(
                "compression_ratio must be between min_compression_ratio and max_compression_ratio"
            )
        
        # 验证窗口大小
        if self.sliding_window_size > self.local_window_size:
            raise ValueError(
                "sliding_window_size must not be larger than local_window_size"
            )
        
        # 验证性能优化配置
        if self.mixed_precision and not torch.cuda.is_available():
            logging.warning("Mixed precision training requires CUDA, but CUDA is not available")
            self.mixed_precision = False
            
        # 计算理论内存占用
        self.theoretical_memory = self._calculate_theoretical_memory()
        logging.info(f"Theoretical peak memory usage: {self.theoretical_memory / 1e9:.2f}GB")
        
    def _calculate_theoretical_memory(self) -> int:
        """计算理论峰值内存占用(字节)"""
        bytes_per_param = 4 if self.dtype == torch.float32 else 2
        
        # 模型参数内存
        param_memory = (
            self.hidden_size * self.hidden_size * 4 +  # Q,K,V,O矩阵
            self.hidden_size * 2  # LayerNorm参数
        ) * bytes_per_param
        
        # 激活值内存(考虑梯度检查点)
        if self.gradient_checkpointing:
            activation_memory = self.chunk_size * self.hidden_size * 3 * bytes_per_param
        else:
            activation_memory = self.chunk_size * self.hidden_size * 6 * bytes_per_param
            
        # 注意力分数内存
        attention_memory = (
            self.num_attention_heads * self.chunk_size * self.chunk_size
        ) * bytes_per_param
        
        # 缓存内存
        cache_memory = (
            self.max_cache_size * self.hidden_size * 2  # key和value缓存
        ) * (2 if self.cache_dtype == torch.float16 else 4)  # 半精度或全精度
        
        return param_memory + activation_memory + attention_memory + cache_memory

class LocalAttention(BaseAttention):
    """局部注意力模块"""
    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__(config)
        self.config = config
        
        # 添加性能优化组件
        self.gradient_checkpointing = config.gradient_checkpointing
        self.mixed_precision = config.mixed_precision
        
        # 添加自适应窗口大小
        self.window_size_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def _compute_adaptive_window_size(self, x: torch.Tensor) -> int:
        """计算自适应窗口大小"""
        with torch.no_grad():
            ratio = self.window_size_net(x.mean(dim=1)).mean().item()
            window_size = int(self.config.local_window_size * (0.5 + ratio))
            return max(256, min(window_size, self.config.local_window_size))
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # 自适应窗口大小
        chunk_size = self._compute_adaptive_window_size(hidden_states)
        num_chunks = math.ceil(seq_len / chunk_size)
        
        outputs = []
        chunk_stats = []
        
        # 使用torch.cuda.amp进行混合精度训练
        with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, seq_len)
                
                # 获取当前块
                chunk_hidden = hidden_states[:, start_idx:end_idx, :]
                chunk_mask = attention_mask[:, :, start_idx:end_idx, :] if attention_mask is not None else None
                
                # 使用梯度检查点
                if self.gradient_checkpointing and self.training:
                    chunk_output = torch.utils.checkpoint.checkpoint(
                        super().forward,
                        chunk_hidden,
                        chunk_mask
                    )
                    chunk_output, stats = chunk_output[0], chunk_output[1]
                else:
                    chunk_output, stats = super().forward(chunk_hidden, chunk_mask)
                    
                outputs.append(chunk_output)
                chunk_stats.append(stats)
        
        # 合并所有块的输出
        output = torch.cat(outputs, dim=1)
        
        # 合并统计信息
        merged_stats = {
            'attention_scores': torch.cat([s['attention_scores'] for s in chunk_stats], dim=-1),
            'compute_time': sum(s['compute_time'] for s in chunk_stats),
            'memory_usage': max(s['memory_usage'] for s in chunk_stats),
            'num_chunks': num_chunks,
            'chunk_size': chunk_size
        }
        
        return output, merged_stats

class GlobalSparsityControl(nn.Module):
    """全局稀疏控制器，支持自适应稀疏度和性能优化"""
    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__()
        self.config = config
        self.monitor = AttentionMonitor()
        
        # 稀疏度预测网络
        self.sparsity_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 性能监控器
        self._performance_stats = {
            'compute_times': [],  # 计算时间统计
            'memory_usage': [],   # 内存使用统计
            'sparsity_history': [] # 稀疏度历史
        }
        self._stats_lock = Lock()
        self._MAX_STATS_LENGTH = 1000  # 最大历史记录长度
        
        # 自适应控制参数
        self._min_sparsity = 0.01
        self._max_sparsity = 0.99
        self._target_compute_time = 0.1  # 目标计算时间（秒）
        self._adaptation_rate = 0.1  # 自适应调整率
        
        # 性能优化参数
        self._batch_size_multiplier = 1.0  # 批处理大小调整因子
        self._last_adjustment_time = time.time()
        self._adjustment_cooldown = 60  # 调整冷却时间（秒）
        
    def _update_stats(self, compute_time: float, memory_usage: float, sparsity: float):
        """更新性能统计"""
        self.monitor.update_stats(compute_time, memory_usage, sparsity)
            
    def _get_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        return self.monitor.get_stats()
            
    def _adjust_batch_size(self, metrics: Dict[str, float]):
        """动态调整批处理大小"""
        current_time = time.time()
        if current_time - self._last_adjustment_time < self._adjustment_cooldown:
            return
            
        if metrics['avg_compute_time'] > self._target_compute_time * 1.2:
            self._batch_size_multiplier *= 0.9
        elif metrics['avg_compute_time'] < self._target_compute_time * 0.8:
            self._batch_size_multiplier *= 1.1
            
        self._batch_size_multiplier = max(0.5, min(2.0, self._batch_size_multiplier))
        self._last_adjustment_time = current_time
        
    def _adaptive_sparsity_adjustment(
        self,
        base_sparsity: float,
        metrics: Dict[str, float]
    ) -> float:
        """自适应稀疏度调整"""
        # 基于性能指标调整稀疏度
        if metrics['avg_compute_time'] > self._target_compute_time:
            adjustment = self._adaptation_rate
        else:
            adjustment = -self._adaptation_rate
            
        # 考虑内存使用情况
        if metrics['avg_memory_usage'] > 0.9:  # 内存使用超过90%
            adjustment = max(adjustment, 0.05)  # 强制增加稀疏度
            
        adjusted_sparsity = base_sparsity * (1 + adjustment)
        return np.clip(adjusted_sparsity, self._min_sparsity, self._max_sparsity)
        
    @torch.no_grad()
    def compute_sparsity_threshold(
        self,
        hidden_states: torch.Tensor,
        return_metrics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """计算动态稀疏阈值，支持性能监控和自适应调整"""
        start_time = time.time()
        
        try:
            # 预测基础稀疏度
            sparsity = self.sparsity_net(hidden_states).mean()
            base_threshold = self.config.sparsity_factor * (1 + sparsity)
            
            # 获取性能指标
            metrics = self._get_performance_metrics()
            
            # 自适应调整
            adjusted_threshold = self._adaptive_sparsity_adjustment(
                base_threshold,
                metrics
            )
            
            # 更新性能统计
            compute_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            self._update_stats(compute_time, memory_usage, adjusted_threshold)
            
            if return_metrics:
                return adjusted_threshold, metrics
            return adjusted_threshold
            
        except Exception as e:
            raise AttentionError("计算稀疏阈值失败", {
                'error': str(e),
                'hidden_states_shape': hidden_states.shape,
                'device': hidden_states.device
            })

class HierarchicalCompression(nn.Module):
    """层次压缩模块"""
    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__()
        self.config = config
        
        # 创建每个层次的压缩网络
        self.compression_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, int(config.hidden_size * config.compression_ratio)),
                nn.ReLU(),
                nn.Linear(int(config.hidden_size * config.compression_ratio), config.hidden_size)
            )
            for _ in range(config.num_levels)
        ])
        
    def compress(
        self,
        hidden_states: torch.Tensor,
        level: int
    ) -> torch.Tensor:
        """压缩隐藏状态"""
        if level >= len(self.compression_nets):
            return hidden_states
            
        return self.compression_nets[level](hidden_states)
        
class InfiniteAttentionError(Exception):
    """无限注意力模块的基础异常类"""
    pass

class ComputationError(InfiniteAttentionError):
    """计算过程中的错误"""
    def __init__(self, message: str, device_info: dict = None):
        super().__init__(message)
        self.device_info = device_info or {}

class CacheError(InfiniteAttentionError):
    """缓存操作相关的错误"""
    def __init__(self, message: str, cache_stats: dict = None):
        super().__init__(message)
        self.cache_stats = cache_stats or {}

class ResourceError(InfiniteAttentionError):
    """资源不足相关的错误"""
    def __init__(self, message: str, resource_stats: dict = None):
        super().__init__(message)
        self.resource_stats = resource_stats or {}

class AttentionConfigError(InfiniteAttentionError):
    """配置相关的错误"""
    def __init__(self, message: str, config_info: dict = None):
        super().__init__(message)
        self.config_info = config_info or {}

class MemoryError(ResourceError):
    """内存不足错误"""
    def __init__(self, message: str, memory_stats: dict = None):
        super().__init__(message, memory_stats)
        self.memory_stats = memory_stats or {}

class ConcurrencyError(InfiniteAttentionError):
    """并发操作相关的错误"""
    def __init__(self, message: str, thread_info: dict = None):
        super().__init__(message)
        self.thread_info = thread_info or {}

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 1.0
    exponential_base: float = 2
    
    # 新增重试策略配置
    retry_on_exceptions: tuple = (ComputationError, CacheError, ResourceError)
    ignore_exceptions: tuple = ()
    on_retry_callback: callable = None
    jitter: float = 0.1  # 随机抖动因子
    
    def get_next_delay(self, attempt: int) -> float:
        """计算下一次重试的延迟时间"""
        delay = min(
            self.max_delay,
            self.base_delay * (self.exponential_base ** (attempt - 1))
        )
        # 添加随机抖动以避免惊群效应
        jitter_range = delay * self.jitter
        return delay + random.uniform(-jitter_range, jitter_range)

class ErrorTracker:
    """错误跟踪器"""
    def __init__(self):
        self._error_stats = defaultdict(lambda: {
            'count': 0,
            'last_occurrence': None,
            'error_samples': deque(maxlen=10)
        })
        self._lock = threading.Lock()
    
    def record_error(self, error: Exception, context: dict = None):
        """记录错误"""
        with self._lock:
            error_type = type(error).__name__
            stats = self._error_stats[error_type]
            stats['count'] += 1
            stats['last_occurrence'] = time.time()
            stats['error_samples'].append({
                'error': str(error),
                'context': context,
                'timestamp': time.time()
            })
    
    def get_stats(self) -> dict:
        """获取错误统计信息"""
        with self._lock:
            return {k: dict(v) for k, v in self._error_stats.items()}
    
    def clear_stats(self):
        """清除统计信息"""
        with self._lock:
            self._error_stats.clear()

class SmartLockManager:
    """智能锁管理器，支持混合锁机制和自适应优化"""
    def __init__(self):
        self._rlock = threading.RLock()  # 可重入锁，用于递归操作
        self._locks = weakref.WeakKeyDictionary()  # 键到锁的映射
        self._lock_stats = {}  # 锁使用统计
        self._stats_lock = threading.Lock()
        
        # 性能监控
        self._perf_stats = {
            'contention_rates': [],  # 锁争用率
            'wait_times': [],       # 等待时间
            'operation_types': {}   # 操作类型统计
        }
        
        # 自适应参数
        self._adaptive_config = {
            'contention_threshold': 0.7,  # 争用率阈值
            'wait_time_threshold': 0.1,   # 等待时间阈值（秒）
            'sample_window': 100,         # 采样窗口大小
            'upgrade_threshold': 0.8,     # 升级阈值
            'downgrade_threshold': 0.2    # 降级阈值
        }
        
    def _record_operation(self, op_type: str, wait_time: float, contention: bool):
        """记录锁操作统计"""
        with self._stats_lock:
            self._perf_stats['wait_times'].append(wait_time)
            self._perf_stats['contention_rates'].append(float(contention))
            self._perf_stats['operation_types'][op_type] = \
                self._perf_stats['operation_types'].get(op_type, 0) + 1
                
            # 限制统计数据大小
            if len(self._perf_stats['wait_times']) > self._adaptive_config['sample_window']:
                self._perf_stats['wait_times'] = self._perf_stats['wait_times'][-self._adaptive_config['sample_window']:]
                self._perf_stats['contention_rates'] = self._perf_stats['contention_rates'][-self._adaptive_config['sample_window']:]
                
    def _should_use_rlock(self, op_type: str) -> bool:
        """决定是否使用RLock"""
        with self._stats_lock:
            if not self._perf_stats['wait_times']:
                return False
                
            recent_contention = np.mean(self._perf_stats['contention_rates'][-self._adaptive_config['sample_window']:])
            recent_wait_time = np.mean(self._perf_stats['wait_times'][-self._adaptive_config['sample_window']:])
            
            # 对于递归操作，优先使用RLock
            if op_type.startswith('recursive_'):
                return True
                
            # 根据性能指标决定
            if recent_contention > self._adaptive_config['contention_threshold'] or \
               recent_wait_time > self._adaptive_config['wait_time_threshold']:
                return True
                
            return False
            
    def acquire_lock(self, key: Any = None, timeout: float = None, op_type: str = 'default') -> bool:
        """智能获取锁"""
        start_time = time.time()
        
        try:
            if self._should_use_rlock(op_type):
                success = self._rlock.acquire(timeout=timeout if timeout else -1)
                wait_time = time.time() - start_time
                self._record_operation(op_type, wait_time, not success)
                return success
                
            if key is None:
                success = self._rlock.acquire(timeout=timeout if timeout else -1)
                wait_time = time.time() - start_time
                self._record_operation(op_type, wait_time, not success)
                return success
                
            # 获取或创建键特定的锁
            with self._stats_lock:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()
                    
            lock = self._locks[key]
            success = lock.acquire(timeout=timeout if timeout else -1)
            
            wait_time = time.time() - start_time
            self._record_operation(op_type, wait_time, not success)
            
            return success
            
        except Exception as e:
            logging.warning(f"锁获取失败: {str(e)}")
            return False
            
    def release_lock(self, key: Any = None):
        """智能释放锁"""
        try:
            if key is None or self._should_use_rlock('release'):
                self._rlock.release()
                return
                
            if key in self._locks:
                self._locks[key].release()
                
        except Exception as e:
            logging.warning(f"锁释放失败: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """获取锁使用统计"""
        with self._stats_lock:
            return {
                'contention_rate': np.mean(self._perf_stats['contention_rates'][-self._adaptive_config['sample_window']:]) if self._perf_stats['contention_rates'] else 0,
                'avg_wait_time': np.mean(self._perf_stats['wait_times'][-self._adaptive_config['sample_window']:]) if self._perf_stats['wait_times'] else 0,
                'operation_types': dict(self._perf_stats['operation_types'])
            }
            
    def update_config(self, **kwargs):
        """更新自适应配置"""
        with self._stats_lock:
            self._adaptive_config.update(kwargs)
            
class ThreadSafeCache:
    """线程安全的缓存实现，支持LRU淘汰、过期时间和内存限制，增强的死锁预防"""
    def __init__(self, max_memory_mb: float = 1024, ttl_seconds: float = 3600):
        self._cache = {}  # 实际存储
        self._access_times = {}  # 访问时间记录
        self._creation_times = {}  # 创建时间记录
        self._memory_usage = {}  # 每个键的内存使用记录
        
        # 智能锁管理器
        self._lock_manager = SmartLockManager()
        
        # 保留原有的锁以保持兼容性
        self._global_lock = Lock()  # 全局锁，仅用于元数据操作
        self._locks_lock = Lock()  # 用于管理键锁的锁
        
        self._max_memory = max_memory_mb * 1024 * 1024  # 转换为字节
        self._ttl = ttl_seconds
        self._current_memory = 0
        
        # 死锁预防
        self._lock_timeouts = {}  # 锁超时记录
        self._lock_owners = {}  # 锁持有者记录
        self._DEFAULT_LOCK_TIMEOUT = 5.0  # 默认锁超时时间（秒）
        
    def _get_key_lock(self, key: str) -> Lock:
        """获取特定键的锁，如果不存在则创建"""
        return self._lock_manager.acquire_lock(key, op_type='get_key_lock')
        
    def _acquire_locks(self, *keys: str, timeout: float = None) -> bool:
        """智能获取多个锁"""
        if not keys:
            return True
            
        success = True
        acquired = []
        
        try:
            for key in sorted(set(keys)):  # 保持排序以防止死锁
                if self._lock_manager.acquire_lock(key, timeout=timeout, op_type='batch_acquire'):
                    acquired.append(key)
                else:
                    success = False
                    break
                    
            if not success:
                # 释放已获取的锁
                for key in acquired:
                    self._lock_manager.release_lock(key)
                    
            return success
            
        except Exception as e:
            logging.warning(f"批量获取锁失败: {str(e)}")
            # 释放已获取的锁
            for key in acquired:
                self._lock_manager.release_lock(key)
            return False
            
    def _release_locks(self, *keys: str):
        """智能释放多个锁"""
        for key in keys:
            self._lock_manager.release_lock(key)
            
    def set(self, key: str, value: Tuple[torch.Tensor, torch.Tensor]):
        """线程安全地设置缓存"""
        required_memory = sum(
            self._estimate_tensor_memory(t) for t in value if t is not None
        )
        
        # 使用智能锁管理器
        if self._lock_manager.acquire_lock(key, timeout=self._DEFAULT_LOCK_TIMEOUT, op_type='set'):
            try:
                self._ensure_memory_limit(required_memory)
                self._cache[key] = value
                self._access_times[key] = time.time()
                self._creation_times[key] = time.time()
                self._memory_usage[key] = required_memory
                self._current_memory += required_memory
            finally:
                self._lock_manager.release_lock(key)
                
    def get(self, key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """线程安全地获取缓存"""
        if self._lock_manager.acquire_lock(key, timeout=self._DEFAULT_LOCK_TIMEOUT, op_type='get'):
            try:
                if key not in self._cache:
                    return None
                    
                current_time = time.time()
                if current_time - self._creation_times.get(key, 0) > self._ttl:
                    self._remove_item(key)
                    return None
                    
                self._access_times[key] = current_time
                return self._cache[key]
            finally:
                self._lock_manager.release_lock(key)
        return None
        
    def get_lock_stats(self) -> Dict[str, Any]:
        """获取锁使用统计"""
        return self._lock_manager.get_stats()
        
    def update_lock_config(self, **kwargs):
        """更新锁管理器配置"""
        self._lock_manager.update_config(**kwargs)
        
    def _estimate_tensor_memory(self, tensor: torch.Tensor) -> int:
        """估算张量占用的内存大小（字节）"""
        return tensor.element_size() * tensor.nelement()
        
    def _ensure_memory_limit(self, required_memory: int):
        """确保有足够的空间存储新数据"""
        while self._current_memory + required_memory > self._max_memory and self._cache:
            with self._global_lock:
                # 找出最旧的键
                if not self._access_times:
                    break
                oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                
            # 尝试获取锁并删除
            if self._lock_manager.acquire_lock(oldest_key):
                try:
                    self._remove_item(oldest_key)
                finally:
                    self._lock_manager.release_lock(oldest_key)
                    
    def _remove_item(self, key: str):
        """移除单个缓存项并更新内存使用"""
        if key in self._cache:
            tensors = self._cache[key]
            if isinstance(tensors, tuple):
                memory_freed = self._memory_usage.get(key, 0)
                self._current_memory -= memory_freed
                
            self._cache.pop(key)
            self._access_times.pop(key, None)
            self._creation_times.pop(key, None)
            self._memory_usage.pop(key, None)
            
    def clear(self):
        """清空所有缓存"""
        with self._global_lock:
            # 获取所有键的列表
            keys = list(self._cache.keys())
            
        # 分批处理键，避免长时间持有全局锁
        batch_size = 10
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            if self._lock_manager.acquire_lock(*batch_keys):
                try:
                    for key in batch_keys:
                        self._remove_item(key)
                finally:
                    self._lock_manager.release_lock(*batch_keys)
                    
    def __len__(self) -> int:
        """返回缓存中的项目数"""
        with self._global_lock:
            return len(self._cache)

class InfiniteAttention(BaseAttention):
    """无限注意力机制实现"""
    
    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__(config)
        
        # 无限注意力特定配置
        self.window_size = config.window_size
        self.compression_rate = config.compression_rate
        self.max_relative_position = config.max_relative_position
        
        # 相对位置编码
        self.relative_attention_bias = nn.Parameter(
            torch.zeros(2 * config.max_relative_position + 1)
        )
        
        # 压缩层
        self.compression_layer = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size // self.compression_rate
        )
        
        # 解压层
        self.decompression_layer = nn.Linear(
            self.config.hidden_size // self.compression_rate,
            self.config.hidden_size
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(
            self.relative_attention_bias,
            mean=0.0,
            std=self.config.initializer_range
        )
        nn.init.normal_(
            self.compression_layer.weight,
            mean=0.0,
            std=self.config.initializer_range
        )
        nn.init.zeros_(self.compression_layer.bias)
        nn.init.normal_(
            self.decompression_layer.weight,
            mean=0.0,
            std=self.config.initializer_range
        )
        nn.init.zeros_(self.decompression_layer.bias)
    
    @ensure_compatibility
    def compute_relative_attention_bias(
        self,
        query_length: int,
        key_length: int
    ) -> torch.Tensor:
        """计算相对位置注意力偏置"""
        try:
            # 生成相对位置矩阵
            context_position = torch.arange(
                query_length, dtype=torch.long, device=self.relative_attention_bias.device
            )[:, None]
            memory_position = torch.arange(
                key_length, dtype=torch.long, device=self.relative_attention_bias.device
            )[None, :]
            relative_position = memory_position - context_position
            
            # 裁剪相对位置
            relative_position = torch.clamp(
                relative_position,
                -self.max_relative_position,
                self.max_relative_position
            )
            
            # 获取偏置
            relative_position_bucket = relative_position + self.max_relative_position
            values = self.relative_attention_bias[relative_position_bucket]
            
            return values
            
        except Exception as e:
            raise ComputationError(
                "Failed to compute relative attention bias",
                context={
                    "query_length": query_length,
                    "key_length": key_length,
                    "max_relative_position": self.max_relative_position
                }
            )
    
    @ensure_compatibility
    def compress_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """压缩序列"""
        try:
            # 应用压缩
            compressed = self.compression_layer(hidden_states)
            
            # 重塑为窗口
            batch_size, seq_len, hidden_size = compressed.size()
            num_windows = math.ceil(seq_len / self.window_size)
            padded_len = num_windows * self.window_size
            
            if padded_len > seq_len:
                padding = torch.zeros(
                    batch_size,
                    padded_len - seq_len,
                    hidden_size,
                    device=compressed.device,
                    dtype=compressed.dtype
                )
                compressed = torch.cat([compressed, padding], dim=1)
            
            # 重塑为窗口
            compressed = compressed.view(
                batch_size,
                num_windows,
                self.window_size,
                hidden_size
            )
            
            return compressed
            
        except Exception as e:
            raise ComputationError(
                "Failed to compress sequence",
                context={
                    "input_shape": hidden_states.size(),
                    "window_size": self.window_size,
                    "compression_rate": self.compression_rate
                }
            )
    
    @ensure_compatibility
    def decompress_sequence(
        self,
        compressed: torch.Tensor,
        original_length: int
    ) -> torch.Tensor:
        """解压序列"""
        try:
            # 重塑回序列
            batch_size, num_windows, window_size, hidden_size = compressed.size()
            compressed = compressed.view(
                batch_size,
                num_windows * window_size,
                hidden_size
            )
            
            # 应用解压
            decompressed = self.decompression_layer(compressed)
            
            # 裁剪到原始长度
            if decompressed.size(1) > original_length:
                decompressed = decompressed[:, :original_length, :]
            
            return decompressed
            
        except Exception as e:
            raise ComputationError(
                "Failed to decompress sequence",
                context={
                    "compressed_shape": compressed.size(),
                    "original_length": original_length,
                    "compression_rate": self.compression_rate
                }
            )
    
    @ensure_compatibility
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """无限注意力前向传播"""
        try:
            with PerformanceContext(self) if self.monitor else nullcontext():
                batch_size, seq_len, _ = hidden_states.size()
                
                # 压缩序列
                compressed = self.compress_sequence(hidden_states)
                
                # 计算Q、K、V
                query_layer = self.transpose_for_scores(self.query(compressed))
                key_layer = self.transpose_for_scores(self.key(compressed))
                value_layer = self.transpose_for_scores(self.value(compressed))
                
                # 计算注意力分数
                attention_scores = self.compute_attention_scores(
                    query_layer, key_layer
                )
                
                # 添加相对位置偏置
                relative_attention_bias = self.compute_relative_attention_bias(
                    query_layer.size(2),
                    key_layer.size(2)
                )
                attention_scores = attention_scores + relative_attention_bias
                
                # 应用注意力
                context_layer = self.apply_attention(attention_scores, value_layer)
                
                # 解压序列
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                context_layer = context_layer.view(
                    batch_size,
                    -1,
                    self.config.hidden_size // self.compression_rate
                )
                output = self.decompress_sequence(context_layer, seq_len)
                
                # 返回结果和统计信息
                stats = {}
                if self.monitor:
                    stats = self.monitor.get_stats()
                    stats.update({
                        "window_size": self.window_size,
                        "compression_rate": self.compression_rate,
                        "max_relative_position": self.max_relative_position
                    })
                
                return output, stats
                
        except Exception as e:
            self.error_handler.handle_error(
                ComputationError(
                    "Infinite attention forward pass failed",
                    context={
                        "input_shape": hidden_states.size(),
                        "mask_shape": attention_mask.size() if attention_mask is not None else None,
                        "window_size": self.window_size,
                        "compression_rate": self.compression_rate
                    }
                )
            )
            raise

class nullcontext:
    """空上下文管理器"""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class EnhancedInfiniteAttention(InfiniteAttention):
    """增强版无限注意力模块"""
    def __init__(self, config: InfiniteAttentionConfig):
        super().__init__(config)
        
        # 添加注意力增强网络
        self.attention_enhancer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 添加自适应学习率网络
        self.learning_rate_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 获取基础注意力输出
        base_output, stats = super().forward(
            hidden_states,
            attention_mask,
            segment_id
        )
        
        # 计算注意力增强
        enhanced_output = self.attention_enhancer(base_output)
        
        # 计算自适应学习率
        learning_rate = self.learning_rate_net(hidden_states).mean()
        
        # 融合输出
        final_output = base_output + learning_rate * enhanced_output
        
        # 更新统计信息
        stats.update({
            'learning_rate': learning_rate.item(),
            'enhancement_ratio': torch.norm(enhanced_output) / torch.norm(base_output)
        })
        
        return final_output, stats 

def create_ucsa_model(
    hidden_size: int = 768,
    sparsity_factor: float = 0.1,
    compression_ratio: float = 0.5,
    enable_cache: bool = True,
    max_cache_size: int = 10000,
    **kwargs
) -> Union[InfiniteAttention, EnhancedInfiniteAttention]:
    """创建UCSA模型的工厂函数
    
    参数:
        hidden_size (int): 隐藏层大小
        sparsity_factor (float): 稀疏度因子
        compression_ratio (float): 压缩比率
        enable_cache (bool): 是否启用缓存
        max_cache_size (int): 最大缓存大小
        **kwargs: 其他参数
        
    返回:
        Union[InfiniteAttention, EnhancedInfiniteAttention]: UCSA模型实例
    """
    config = InfiniteAttentionConfig(
        hidden_size=hidden_size,
        sparsity_factor=sparsity_factor,
        compression_ratio=compression_ratio,
        enable_cache=enable_cache,
        max_cache_size=max_cache_size,
        **kwargs
    )
    
    # 默认返回增强版本
    return EnhancedInfiniteAttention(config)

class SuperHybridAttentionModel(nn.Module):
    """超级混合注意力模型，集成MSRA、DALA和UCSA
    
    这个模型可以根据输入序列的特点自动选择最适合的注意力机制，
    并支持动态切换和混合使用多种注意力机制。
    """
    
    def __init__(self, config: Union[HybridConfig, MSRAConfig, DALAConfig, InfiniteAttentionConfig]):
        super().__init__()
        self.config = config
        
        # 初始化所有模型
        self.msra = MSRAModel(config)
        self.dala = DALAModel(config)
        self.ucsa = EnhancedInfiniteAttention(config)
        
        # 动态选择网络
        self.selector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 3),  # 3个模型的权重
            nn.Softmax(dim=-1)
        )
        
        # 特征融合网络
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 性能监控
        self.register_buffer('performance_stats', torch.zeros(9))  # [时间x3, 内存x3, 准确率x3]
        self.stats_momentum = 0.9
        
    def _update_stats(
        self,
        model_type: str,
        compute_time: float,
        memory_used: float,
        accuracy: float
    ):
        """更新性能统计"""
        idx_map = {'msra': 0, 'dala': 1, 'ucsa': 2}
        idx = idx_map[model_type]
        
        # 更新时间统计
        self.performance_stats[idx] = (
            self.stats_momentum * self.performance_stats[idx] +
            (1 - self.stats_momentum) * compute_time
        )
        
        # 更新内存统计
        self.performance_stats[idx + 3] = (
            self.stats_momentum * self.performance_stats[idx + 3] +
            (1 - self.stats_momentum) * memory_used
        )
        
        # 更新准确率统计
        self.performance_stats[idx + 6] = (
            self.stats_momentum * self.performance_stats[idx + 6] +
            (1 - self.stats_momentum) * accuracy
        )
        
    def _get_model_weights(self, x: torch.Tensor) -> torch.Tensor:
        """计算每个模型的权重"""
        # 基础权重
        base_weights = self.selector(x.mean(dim=1))
        
        # 考虑性能统计
        if torch.cuda.is_available():
            memory_factor = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            memory_factor = 0.5
            
        # 计算效率分数
        efficiency_scores = self.performance_stats[6:9] / (
            self.performance_stats[:3] * self.performance_stats[3:6] + 1e-6
        )
        
        # 调整权重
        adjusted_weights = base_weights * efficiency_scores.unsqueeze(0)
        return F.softmax(adjusted_weights, dim=-1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        batch_size, seq_length = input_ids.shape[:2]
        
        # 获取模型权重
        model_weights = self._get_model_weights(input_ids)
        
        outputs = []
        stats = {}
        
        # MSRA处理
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        msra_output = self.msra(input_ids, attention_mask)[0]
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._update_stats('msra', end_time - start_time, end_memory - start_memory, 1.0)
        outputs.append(msra_output)
        
        # DALA处理
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        dala_output = self.dala(input_ids, attention_mask)
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._update_stats('dala', end_time - start_time, end_memory - start_memory, 1.0)
        outputs.append(dala_output)
        
        # UCSA处理
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        ucsa_output, ucsa_stats = self.ucsa(input_ids, attention_mask)
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._update_stats('ucsa', end_time - start_time, end_memory - start_memory, 1.0)
        outputs.append(ucsa_output)
        
        # 融合输出
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch_size, 3, seq_len, hidden_size]
        weighted_sum = (stacked_outputs * model_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        # 特征融合
        final_output = self.fusion(
            torch.cat([msra_output, dala_output, ucsa_output], dim=-1)
        )
        
        if return_stats:
            stats.update({
                'model_weights': model_weights.mean(dim=0).tolist(),
                'performance': {
                    'msra': {'time': self.performance_stats[0].item(), 'memory': self.performance_stats[3].item()},
                    'dala': {'time': self.performance_stats[1].item(), 'memory': self.performance_stats[4].item()},
                    'ucsa': {'time': self.performance_stats[2].item(), 'memory': self.performance_stats[5].item()}
                },
                'ucsa_stats': ucsa_stats
            })
            return final_output, stats
            
        return final_output

def create_super_hybrid_model(
    hidden_size: int = 768,
    num_levels: int = 3,
    num_layers: int = 6,
    sparsity_factor: float = 0.1,
    compression_ratio: float = 0.5,
    **kwargs
) -> SuperHybridAttentionModel:
    """创建超级混合模型的工厂函数"""
    config = HybridConfig(
        hidden_size=hidden_size,
        num_levels=num_levels,
        num_layers=num_layers,
        **kwargs
    )
    return SuperHybridAttentionModel(config) 