"""
信息压缩优化系统
实现高效的信息压缩和解压缩机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
from threading import Lock
import logging
from collections import OrderedDict
import math
import zlib
import json
from transformers import AutoTokenizer, AutoModel

@dataclass
class CompressionConfig:
    """压缩配置"""
    # 基础配置
    hidden_size: int = 768
    compression_ratio: float = 0.5
    min_compression_ratio: float = 0.1
    max_compression_ratio: float = 0.9
    
    # 自适应压缩
    enable_adaptive_compression: bool = True
    importance_threshold: float = 0.5
    
    # 语义压缩
    enable_semantic_compression: bool = True
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_threshold: float = 0.7
    
    # 增量压缩
    enable_incremental_compression: bool = True
    max_increments: int = 5
    
    # 缓存配置
    cache_size: int = 1000
    enable_cache: bool = True
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
class SemanticCompressor(nn.Module):
    """语义压缩器"""
    def __init__(self, config: CompressionConfig):
        super().__init__()
        self.config = config
        
        if config.enable_semantic_compression:
            # 加载语义模型
            self.tokenizer = AutoTokenizer.from_pretrained(config.semantic_model_name)
            self.model = AutoModel.from_pretrained(config.semantic_model_name).to(config.device)
            self.model.eval()
            
        # 压缩网络
        compressed_size = int(config.hidden_size * config.compression_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, compressed_size)
        ).to(config.device)
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        ).to(config.device)
        
        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        ).to(config.device)
        
    def get_semantic_embedding(self, text: str) -> torch.Tensor:
        """获取文本的语义嵌入"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
            
    def compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """计算特征的重要性分数"""
        with torch.no_grad():
            return self.importance_scorer(x)
            
    def forward(
        self,
        x: torch.Tensor,
        return_importance: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """压缩和解压缩"""
        # 计算重要性分数
        importance = self.compute_importance(x)
        
        # 根据重要性调整压缩率
        if self.config.enable_adaptive_compression:
            compression_ratio = (
                self.config.min_compression_ratio +
                (self.config.max_compression_ratio - self.config.min_compression_ratio) *
                importance.mean().item()
            )
        else:
            compression_ratio = self.config.compression_ratio
            
        # 压缩
        compressed = self.encoder(x)
        
        # 解压缩
        reconstructed = self.decoder(compressed)
        
        if return_importance:
            return reconstructed, importance
        return reconstructed
        
class IncrementalCompressor:
    """增量压缩器"""
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.previous_state = None
        self.increment_count = 0
        
    def compress_incremental(
        self,
        current_state: torch.Tensor,
        force_full: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """增量压缩"""
        if not self.config.enable_incremental_compression or force_full:
            return current_state, {'is_incremental': False}
            
        if self.previous_state is None:
            self.previous_state = current_state
            return current_state, {'is_incremental': False}
            
        # 计算差异
        diff = current_state - self.previous_state
        
        # 如果差异太大或增量次数过多，执行完整压缩
        if (torch.norm(diff) > 0.5 * torch.norm(current_state) or
            self.increment_count >= self.config.max_increments):
            self.previous_state = current_state
            self.increment_count = 0
            return current_state, {'is_incremental': False}
            
        # 执行增量压缩
        self.increment_count += 1
        self.previous_state = current_state
        
        return diff, {
            'is_incremental': True,
            'increment_count': self.increment_count
        }
        
    def decompress_incremental(
        self,
        compressed_diff: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """增量解压缩"""
        if not metadata['is_incremental']:
            return compressed_diff
            
        if self.previous_state is None:
            return compressed_diff
            
        return self.previous_state + compressed_diff
        
class CompressionCache:
    """压缩缓存"""
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.cache = OrderedDict()
        self._lock = Lock()
        
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取缓存项"""
        with self._lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value  # 移到最新
                return value
        return None
        
    def put(self, key: str, value: Dict[str, Any]) -> None:
        """存入缓存项"""
        with self._lock:
            self.cache[key] = value
            while len(self.cache) > self.config.cache_size:
                self.cache.popitem(last=False)
                
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            
class Compressor:
    """压缩器主类"""
    def __init__(self, config: CompressionConfig):
        self.config = config
        
        # 初始化组件
        self.semantic_compressor = SemanticCompressor(config)
        self.incremental_compressor = IncrementalCompressor(config)
        self.cache = CompressionCache(config)
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("Compressor")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        return logger
        
    def compress(
        self,
        x: Union[torch.Tensor, str],
        key: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩数据"""
        try:
            # 检查缓存
            if key and self.config.enable_cache:
                cached = self.cache.get(key)
                if cached is not None:
                    return cached['compressed'], cached['metadata']
                    
            # 处理文本输入
            if isinstance(x, str):
                if not self.config.enable_semantic_compression:
                    raise ValueError("Semantic compression is disabled")
                x = self.semantic_compressor.get_semantic_embedding(x)
                
            # 语义压缩
            compressed, importance = self.semantic_compressor(x, return_importance=True)
            
            # 增量压缩
            compressed, incr_metadata = self.incremental_compressor.compress_incremental(
                compressed
            )
            
            # 准备元数据
            metadata = {
                'importance': importance.mean().item(),
                'compression_ratio': compressed.size(-1) / x.size(-1),
                'original_shape': list(x.shape),
                **incr_metadata
            }
            
            # 更新缓存
            if key and self.config.enable_cache:
                self.cache.put(key, {
                    'compressed': compressed,
                    'metadata': metadata
                })
                
            return compressed, metadata
            
        except Exception as e:
            self.logger.error(f"Error in compression: {str(e)}")
            raise
            
    def decompress(
        self,
        compressed: torch.Tensor,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩数据"""
        try:
            # 增量解压缩
            if metadata.get('is_incremental', False):
                decompressed = self.incremental_compressor.decompress_incremental(
                    compressed,
                    metadata
                )
            else:
                decompressed = compressed
                
            return decompressed
            
        except Exception as e:
            self.logger.error(f"Error in decompression: {str(e)}")
            raise
            
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        return {
            'cache_size': len(self.cache.cache),
            'incremental_compressions': self.incremental_compressor.increment_count
        }
        
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        
class EnhancedCompressor(Compressor):
    """增强版压缩器"""
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        
        # 添加压缩增强网络
        self.compression_enhancer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        ).to(config.device)
        
        # 添加自适应压缩率网络
        self.compression_rate_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        ).to(config.device)
        
    def compress(
        self,
        x: Union[torch.Tensor, str],
        key: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 获取基础压缩结果
        compressed, metadata = super().compress(x, key)
        
        # 应用压缩增强
        enhanced = self.compression_enhancer(compressed)
        
        # 计算自适应压缩率
        compression_rate = self.compression_rate_net(
            enhanced if isinstance(enhanced, torch.Tensor) else compressed
        ).mean()
        
        # 更新元数据
        metadata.update({
            'enhanced': True,
            'compression_rate': compression_rate.item()
        })
        
        return enhanced, metadata 