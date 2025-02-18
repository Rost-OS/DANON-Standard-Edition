"""
工作记忆层
实现了动态的短期工作记忆机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from collections import deque
import time
from ..core.attention import MSRAConfig, MSRAModel
import math

@dataclass
class MemoryItem:
    """记忆项"""
    key: torch.Tensor
    value: torch.Tensor
    importance: float
    timestamp: float
    metadata: Dict[str, Any]

class WorkingMemory(nn.Module):
    """工作记忆层，管理活跃的短期记忆"""
    
    def __init__(
        self,
        dim: int = 512,
        max_items: int = 100,
        attention_heads: int = 8,
        dropout: float = 0.1,
        cache_size: int = 1000,
        min_importance: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.max_items = max_items
        self.cache_size = cache_size
        self.min_importance = min_importance
        
        # 主记忆存储
        self.memory: deque[MemoryItem] = deque(maxlen=max_items)
        
        # 快速缓存
        self.cache: Dict[str, MemoryItem] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 使用MSRA进行记忆整合
        msra_config = MSRAConfig(
            hidden_size=dim,
            num_levels=2,
            chunk_size=32,
            num_layers=2,
            dropout=dropout
        )
        self.memory_processor = MSRAModel(msra_config)
        
        # 记忆更新网络
        self.memory_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # 缓存管理网络
        self.cache_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_updates': 0,
            'cache_evictions': 0
        }
        
    def _calculate_importance(self, x: torch.Tensor) -> float:
        """计算记忆项的重要性分数"""
        with torch.no_grad():
            importance = self.importance_net(x)
            return max(importance.item(), self.min_importance)
            
    def _should_cache(self, x: torch.Tensor) -> bool:
        """预测是否应该缓存该记忆项"""
        with torch.no_grad():
            return self.cache_predictor(x).item() > 0.5
            
    def _update_cache(self, key: str, item: MemoryItem):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最不重要的缓存项
            least_important = min(self.cache.items(), key=lambda x: x[1].importance)
            del self.cache[least_important[0]]
            self.stats['cache_evictions'] += 1
            
        self.cache[key] = item
        
    def _get_from_cache(self, key: str) -> Optional[MemoryItem]:
        """从缓存中获取记忆项"""
        if key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[key]
        self.stats['cache_misses'] += 1
        return None
        
    def add_memory(self, key: torch.Tensor, value: torch.Tensor, metadata: Dict[str, Any] = None):
        """添加新的记忆项"""
        importance = self._calculate_importance(value)
        timestamp = time.time()
        
        item = MemoryItem(
            key=key,
            value=value,
            importance=importance,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # 添加到主记忆
        self.memory.append(item)
        self.stats['memory_updates'] += 1
        
        # 根据预测决定是否缓存
        if self._should_cache(value):
            cache_key = str(key.mean().item())  # 简单的缓存键生成
            self._update_cache(cache_key, item)
            
    def retrieve_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """检索记忆"""
        # 首先尝试缓存
        cache_key = str(query.mean().item())
        cached_item = self._get_from_cache(cache_key)
        if cached_item is not None:
            return cached_item.value, {
                'source': 'cache',
                'importance': cached_item.importance,
                'age': time.time() - cached_item.timestamp
            }
            
        # 如果缓存未命中，搜索主记忆
        if not self.memory:
            return torch.zeros_like(query), {'source': 'empty'}
            
        # 使用MSRA处理所有记忆
        keys = torch.stack([item.key for item in self.memory])
        values = torch.stack([item.value for item in self.memory])
        
        # 计算注意力权重
        processed_memory = self.memory_processor(values)
        attention_weights = torch.matmul(query.unsqueeze(1), keys.transpose(0, 1))
        attention_weights = F.softmax(attention_weights / math.sqrt(self.dim), dim=-1)
        
        # 获取最相关的记忆
        retrieved_memory = torch.matmul(attention_weights, processed_memory).squeeze(1)
        
        # 更新记忆
        updated_memory = self.memory_update(torch.cat([query, retrieved_memory], dim=-1))
        
        return updated_memory, {
            'source': 'memory',
            'attention_weights': attention_weights,
            'memory_size': len(self.memory)
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_accesses == 0:
            hit_rate = 0
        else:
            hit_rate = self.stats['cache_hits'] / total_accesses
            
        return {
            **self.stats,
            'cache_hit_rate': hit_rate,
            'memory_size': len(self.memory),
            'cache_size': len(self.cache)
        }
        
    def clear_stats(self):
        """清除统计信息"""
        self.stats = {k: 0 for k in self.stats}
        
    def _consolidate_memory(self) -> None:
        """整合和优化工作记忆"""
        if len(self.memory) <= 1:
            return
            
        # 将记忆转换为张量
        keys = torch.stack([item.key for item in self.memory])
        values = torch.stack([item.value for item in self.memory])
        
        # 使用MSRA处理记忆
        processed_memory, _ = self.memory_processor(values)
        
        # 更新记忆值
        updated_values = self.memory_update(
            torch.cat([values, processed_memory], dim=-1)
        )
        
        # 更新记忆项
        for i, item in enumerate(self.memory):
            item.value = updated_values[i]
            
    def add_item(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加新的记忆项"""
        if len(self.memory) >= self.max_items:
            self._consolidate_memory()
            
        importance = self._calculate_importance(value)
        timestamp = time.time()
        
        item = MemoryItem(
            key=key.detach().clone(),
            value=value.detach().clone(),
            importance=importance,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.memory.append(item)
        
    def get_items(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[MemoryItem]:
        """检索相关的记忆项"""
        if not self.memory:
            return []
            
        # 计算相似度
        keys = torch.stack([item.key for item in self.memory])
        similarities = F.cosine_similarity(query.unsqueeze(0), keys)
        
        # 筛选相关项
        mask = similarities > threshold
        if not mask.any():
            return []
            
        # 获取top-k
        values, indices = similarities[mask].topk(min(top_k, mask.sum().item()))
        selected_items = [self.memory[idx.item()] for idx in indices]
        
        return selected_items
        
    def update_item(
        self,
        index: int,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """更新指定记忆项"""
        if 0 <= index < len(self.memory):
            item = self.memory[index]
            item.value = value.detach().clone()
            if metadata:
                item.metadata.update(metadata)
            item.importance = self._calculate_importance(value)
            item.timestamp = time.time()
            
    def remove_item(self, index: int) -> None:
        """移除指定记忆项"""
        if 0 <= index < len(self.memory):
            self.memory.remove(self.memory[index])
            
    def clear(self) -> None:
        """清空记忆"""
        self.memory.clear()
        
    def get_state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'memory': [
                {
                    'key': item.key,
                    'value': item.value,
                    'importance': item.importance,
                    'timestamp': item.timestamp,
                    'metadata': item.metadata
                }
                for item in self.memory
            ],
            'model_state': self.state_dict()
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        super().load_state_dict(state_dict['model_state'])
        self.memory.clear()
        for item_dict in state_dict['memory']:
            item = MemoryItem(
                key=item_dict['key'],
                value=item_dict['value'],
                importance=item_dict['importance'],
                timestamp=item_dict['timestamp'],
                metadata=item_dict['metadata']
            )
            self.memory.append(item)
            
    def prune_old_items(self, max_age: float = 3600) -> None:
        """清理过期的记忆项"""
        current_time = time.time()
        self.memory = deque(
            [item for item in self.memory 
             if current_time - item.timestamp < max_age],
            maxlen=self.max_items
        )
        
    def merge_items(self, indices: List[int]) -> Optional[MemoryItem]:
        """合并多个记忆项"""
        if not indices or not all(0 <= i < len(self.memory) for i in indices):
            return None
            
        items = [self.memory[i] for i in indices]
        keys = torch.stack([item.key for item in items])
        values = torch.stack([item.value for item in items])
        
        # 使用MSRA处理合并
        merged_value, _ = self.memory_processor(values.unsqueeze(0))
        merged_value = merged_value.squeeze(0).mean(0)
        
        # 创建新的记忆项
        importance = self._calculate_importance(merged_value)
        metadata = {
            'merged_from': indices,
            'original_importances': [item.importance for item in items]
        }
        
        return MemoryItem(
            key=keys.mean(0),
            value=merged_value,
            importance=importance,
            timestamp=time.time(),
            metadata=metadata
        )
        
    def optimize_memory(self, target_size: Optional[int] = None) -> None:
        """优化记忆存储"""
        if not self.memory:
            return
            
        target_size = target_size or (len(self.memory) * 3 // 4)
        
        # 计算所有项的重要性分数
        importances = torch.tensor([item.importance for item in self.memory])
        timestamps = torch.tensor([item.timestamp for item in self.memory])
        
        # 结合重要性和时间衰减
        current_time = time.time()
        time_weights = 1 / (1 + (current_time - timestamps) / 3600)  # 1小时衰减
        scores = importances * time_weights
        
        # 保留高分项
        _, indices = scores.topk(target_size)
        self.memory = deque(
            [self.memory[i] for i in sorted(indices.tolist())],
            maxlen=self.max_items
        )
        
    def get_memory_state(self) -> Dict[str, Any]:
        """获取工作记忆的状态信息"""
        if not self.memory:
            return {
                'size': 0,
                'capacity': self.max_items,
                'avg_importance': 0.0,
                'oldest_timestamp': None,
                'newest_timestamp': None
            }
            
        return {
            'size': len(self.memory),
            'capacity': self.max_items,
            'avg_importance': sum(item.importance for item in self.memory) / len(self.memory),
            'oldest_timestamp': min(item.timestamp for item in self.memory),
            'newest_timestamp': max(item.timestamp for item in self.memory)
        }
        
    def get_all_memories(self) -> List[MemoryItem]:
        """获取所有记忆项"""
        return list(self.memory)
        
    def merge_memories(self, other: 'WorkingMemory') -> None:
        """合并另一个工作记忆的内容"""
        for item in other.get_all_memories():
            self.add_item(item.key, item.value, item.metadata) 