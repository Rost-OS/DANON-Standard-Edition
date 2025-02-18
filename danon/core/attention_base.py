"""
注意力机制的基础模块
提供了共用的配置、工具类和基础实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
from dataclasses import dataclass
from collections import OrderedDict
import time
import logging
from threading import Lock

from .unified_config import UnifiedAttentionConfig
from .monitoring import PerformanceMonitor, PerformanceMetrics, PerformanceContext
from .caching import CacheManager, CacheKey
from .error_handling import (
    ErrorHandler, BaseAttentionError, ComputationError, ResourceError
)
from .compatibility import CompatibilityManager, ensure_compatibility

class BaseAttention(nn.Module):
    """注意力机制的基础实现"""
    def __init__(self, config: UnifiedAttentionConfig):
        super().__init__()
        self.config = CompatibilityManager().migrate_config(config)
        
        # 初始化兼容性管理器
        self._compatibility_manager = CompatibilityManager()
        
        # 初始化监控和错误处理
        self.monitor = self._compatibility_manager.get_monitor(self.config)
        self.error_handler = self._compatibility_manager.get_error_handler()
        
        # 初始化缓存
        self.cache_manager = self._compatibility_manager.get_cache_manager(self.config)
        self.attention_cache = self.cache_manager.get_cache(
            "attention_cache",
            self.monitor
        )
        
        self.all_head_size = self.config.num_attention_heads * self.config.attention_head_size
        
        # 注意力投影
        self.query = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.config.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.config.hidden_size, self.all_head_size)
        
        # 输出投影
        self.output = nn.Linear(self.all_head_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """配置日志器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    @ensure_compatibility
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重塑张量用于多头注意力"""
        try:
            batch_size, seq_len, _ = x.size()
            x = x.view(
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.attention_head_size
            )
            return x.permute(0, 2, 1, 3)
        except Exception as e:
            raise ComputationError(
                "Failed to transpose tensor for attention scores",
                context={
                    "tensor_shape": x.size(),
                    "num_heads": self.config.num_attention_heads,
                    "head_size": self.config.attention_head_size
                }
            )
    
    @ensure_compatibility
    def compute_attention_scores(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算注意力分数"""
        try:
            # 检查缓存
            cache_key = CacheKey(query_layer, key_layer, attention_mask)
            cached_result, hit = self.attention_cache.get(cache_key)
            if hit:
                return cached_result
            
            # 计算注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.config.attention_head_size
            )
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            # 缓存结果
            self.attention_cache.put(cache_key, attention_scores)
            
            return attention_scores
        except Exception as e:
            raise ComputationError(
                "Failed to compute attention scores",
                context={
                    "query_shape": query_layer.size(),
                    "key_shape": key_layer.size(),
                    "mask_shape": attention_mask.size() if attention_mask is not None else None
                }
            )
    
    @ensure_compatibility
    def apply_attention(
        self,
        attention_scores: torch.Tensor,
        value_layer: torch.Tensor
    ) -> torch.Tensor:
        """应用注意力权重"""
        try:
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, value_layer)
            return context_layer
        except Exception as e:
            raise ComputationError(
                "Failed to apply attention weights",
                context={
                    "scores_shape": attention_scores.size(),
                    "value_shape": value_layer.size()
                }
            )
    
    @ensure_compatibility
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """基础前向传播实现"""
        try:
            with PerformanceContext(self) if self.monitor else nullcontext():
                batch_size, seq_len, _ = hidden_states.size()
                
                # 计算Q、K、V
                query_layer = self.transpose_for_scores(self.query(hidden_states))
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(self.value(hidden_states))
                
                # 计算注意力分数
                attention_scores = self.compute_attention_scores(
                    query_layer, key_layer, attention_mask
                )
                
                # 应用注意力
                context_layer = self.apply_attention(attention_scores, value_layer)
                
                # 重塑输出
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
                context_layer = context_layer.view(
                    batch_size, seq_len, self.all_head_size
                )
                
                # 输出投影
                output = self.output(context_layer)
                
                # 返回结果和统计信息
                stats = {}
                if self.monitor:
                    stats = self.monitor.get_stats()
                    
                return output, stats
                
        except Exception as e:
            self.error_handler.handle_error(
                ComputationError(
                    "Forward pass failed",
                    context={
                        "input_shape": hidden_states.size(),
                        "mask_shape": attention_mask.size() if attention_mask is not None else None,
                        "device": hidden_states.device,
                        "dtype": hidden_states.dtype
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