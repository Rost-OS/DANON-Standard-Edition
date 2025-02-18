"""
快速访问缓存层
实现了基于LRU策略的高速缓存机制
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from collections import OrderedDict
import time

class FastCache(nn.Module):
    """快速访问缓存层，使用LRU策略管理最近访问的记忆"""
    
    def __init__(
        self,
        cache_size: int = 1024,
        dim: int = 512,
        ttl: float = 3600.0,  # 默认TTL为1小时
    ):
        super().__init__()
        self.cache_size = cache_size
        self.dim = dim
        self.ttl = ttl
        
        # 使用OrderedDict来实现LRU缓存
        self.cache: OrderedDict[str, Tuple[torch.Tensor, float]] = OrderedDict()
        
        # 缓存统计信息
        self.hits = 0
        self.misses = 0
        
        # 特征提取器，用于生成缓存键
        self.key_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 32)  # 生成32维的特征向量作为键
        )
        
    def _generate_key(self, x: torch.Tensor) -> str:
        """根据输入生成缓存键"""
        with torch.no_grad():
            features = self.key_encoder(x)
            # 使用特征向量的哈希值作为键
            key = hash(tuple(features.cpu().numpy().flatten().tolist()))
            return f"cache_{key}"
            
    def _evict_expired(self) -> None:
        """清除过期的缓存项"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            
    def _make_space(self) -> None:
        """如果缓存已满，删除最早的项目"""
        while len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # 删除最早添加的项
            
    def store(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储值到缓存
        
        Args:
            x: 输入张量，用于生成缓存键
            value: 要存储的值
            metadata: 可选的元数据
            
        Returns:
            key: 缓存键
        """
        self._evict_expired()
        self._make_space()
        
        key = self._generate_key(x)
        self.cache[key] = (value, time.time())
        
        # 如果键已存在，移动到最后（最近使用）
        if key in self.cache:
            self.cache.move_to_end(key)
            
        return key
        
    def retrieve(
        self,
        x: torch.Tensor,
        default: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        从缓存中检索值
        
        Args:
            x: 输入张量，用于生成缓存键
            default: 如果未找到时的默认值
            
        Returns:
            value: 检索到的值或默认值
            hit: 是否命中缓存
        """
        self._evict_expired()
        
        key = self._generate_key(x)
        if key in self.cache:
            self.hits += 1
            value, _ = self.cache[key]
            # 更新访问时间并移动到最后
            self.cache.move_to_end(key)
            self.cache[key] = (value, time.time())
            return value, True
        else:
            self.misses += 1
            return default, False
            
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
    def get_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'capacity': self.cache_size,
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
        }
        
    def get_recent_keys(self, n: int = 10) -> List[str]:
        """获取最近访问的n个键"""
        return list(self.cache.keys())[-n:]
        
    def update_ttl(self, new_ttl: float) -> None:
        """更新缓存项的生存时间"""
        self.ttl = new_ttl
        self._evict_expired()  # 立即应用新的TTL 