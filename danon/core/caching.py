"""
统一缓存管理模块
提供了线程安全的缓存实现和管理功能
"""

from typing import Dict, Any, Optional, Tuple, List
from threading import Lock
import time
import torch
import weakref
from collections import OrderedDict
from .unified_config import UnifiedAttentionConfig
from .monitoring import PerformanceMetrics, PerformanceMonitor

class CacheKey:
    """缓存键，支持张量的哈希"""
    
    def __init__(self, *components):
        self.components = components
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            # 计算组件的哈希值
            component_hashes = []
            for comp in self.components:
                if isinstance(comp, torch.Tensor):
                    # 对于张量，使用其内容计算哈希
                    comp_hash = hash(comp.cpu().numpy().tobytes())
                else:
                    comp_hash = hash(comp)
                component_hashes.append(comp_hash)
            self._hash = hash(tuple(component_hashes))
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, CacheKey):
            return False
        if len(self.components) != len(other.components):
            return False
        return all(
            torch.equal(c1, c2) if isinstance(c1, torch.Tensor) else c1 == c2
            for c1, c2 in zip(self.components, other.components)
        )

class CacheEntry:
    """缓存条目，包含值和元数据"""
    
    def __init__(self, value: Any, size: int):
        self.value = value
        self.size = size
        self.last_access = time.time()
        self.access_count = 0
    
    def update_access(self):
        """更新访问信息"""
        self.last_access = time.time()
        self.access_count += 1

class Cache:
    """线程安全的LRU缓存实现"""
    
    def __init__(
        self, 
        config: UnifiedAttentionConfig,
        monitor: Optional[PerformanceMonitor] = None
    ):
        self.config = config
        self.monitor = monitor
        self.max_size = config.max_cache_size
        self.current_size = 0
        self._cache: OrderedDict[CacheKey, CacheEntry] = OrderedDict()
        self._lock = Lock()
    
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """计算张量的大小（MB）"""
        return tensor.element_size() * tensor.nelement() / (1024 * 1024)
    
    def _get_value_size(self, value: Any) -> int:
        """计算值的大小（MB）"""
        if isinstance(value, torch.Tensor):
            return self._get_tensor_size(value)
        elif isinstance(value, (tuple, list)):
            return sum(
                self._get_tensor_size(v) if isinstance(v, torch.Tensor) else 0
                for v in value
            )
        elif isinstance(value, dict):
            return sum(
                self._get_tensor_size(v) if isinstance(v, torch.Tensor) else 0
                for v in value.values()
            )
        return 0
    
    def _evict_entries(self, required_size: int):
        """清除缓存条目以释放空间"""
        while self.current_size + required_size > self.max_size and self._cache:
            # 按照最近最少使用原则清除
            _, entry = self._cache.popitem(last=False)
            self.current_size -= entry.size
    
    def get(self, key: CacheKey) -> Tuple[Optional[Any], bool]:
        """获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Tuple[Optional[Any], bool]: (缓存值, 是否命中)
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                # 更新访问信息
                entry.update_access()
                # 移动到最新位置
                self._cache.move_to_end(key)
                
                if self.monitor:
                    metrics = PerformanceMetrics(
                        compute_time=0.0,
                        memory_usage=0.0,
                        attention_score=0.0,
                        cache_hits=1,
                        cache_misses=0
                    )
                    self.monitor.update_stats(metrics)
                
                return entry.value, True
            
            if self.monitor:
                metrics = PerformanceMetrics(
                    compute_time=0.0,
                    memory_usage=0.0,
                    attention_score=0.0,
                    cache_hits=0,
                    cache_misses=1
                )
                self.monitor.update_stats(metrics)
            
            return None, False
    
    def put(self, key: CacheKey, value: Any):
        """存储缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            # 计算新值的大小
            size = self._get_value_size(value)
            
            # 如果单个值超过最大缓存大小，则不缓存
            if size > self.max_size:
                return
            
            # 清除现有条目（如果存在）
            if key in self._cache:
                old_entry = self._cache[key]
                self.current_size -= old_entry.size
                del self._cache[key]
            
            # 清除其他条目以腾出空间
            self._evict_entries(size)
            
            # 添加新条目
            entry = CacheEntry(value, size)
            self._cache[key] = entry
            self.current_size += size
    
    def clear(self):
        """清除所有缓存"""
        with self._lock:
            self._cache.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息
        
        Returns:
            Dict[str, float]: 统计信息字典
        """
        with self._lock:
            return {
                'size': self.current_size,
                'utilization': self.current_size / self.max_size,
                'num_entries': len(self._cache)
            }

class CacheManager:
    """缓存管理器，管理多个缓存实例"""
    
    def __init__(self, config: UnifiedAttentionConfig):
        self.config = config
        self.caches: Dict[str, Cache] = {}
        self._lock = Lock()
    
    def get_cache(self, name: str, monitor: Optional[PerformanceMonitor] = None) -> Cache:
        """获取或创建命名缓存
        
        Args:
            name: 缓存名称
            monitor: 性能监控器
            
        Returns:
            Cache: 缓存实例
        """
        with self._lock:
            if name not in self.caches:
                self.caches[name] = Cache(self.config, monitor)
            return self.caches[name]
    
    def clear_all(self):
        """清除所有缓存"""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
    
    def get_total_stats(self) -> Dict[str, float]:
        """获取所有缓存的统计信息
        
        Returns:
            Dict[str, float]: 统计信息字典
        """
        with self._lock:
            total_size = 0
            total_entries = 0
            for cache in self.caches.values():
                stats = cache.get_stats()
                total_size += stats['size']
                total_entries += stats['num_entries']
            
            return {
                'total_size': total_size,
                'total_entries': total_entries,
                'num_caches': len(self.caches)
            } 