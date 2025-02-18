"""
长期存储层
实现了持久化的长期记忆存储机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from pathlib import Path
import json
import time

class LongTermStorage(nn.Module):
    """长期存储层，实现持久化的记忆存储"""
    
    def __init__(
        self,
        dim: int = 512,
        storage_size: int = 1000000,  # 默认存储100万条记录
        storage_path: Optional[str] = None,
        compression_ratio: float = 0.5,  # 默认压缩率
    ):
        super().__init__()
        self.dim = dim
        self.storage_size = storage_size
        self.storage_path = Path(storage_path) if storage_path else None
        self.compression_ratio = compression_ratio
        
        # 存储空间
        self.keys = torch.zeros((storage_size, 32))  # 32维特征键
        self.values = torch.zeros((storage_size, dim))  # 原始维度值
        self.compressed_values = torch.zeros((storage_size, int(dim * compression_ratio)))
        self.metadata = [{} for _ in range(storage_size)]
        self.timestamps = torch.zeros(storage_size)
        
        # 当前存储位置
        self.current_index = 0
        
        # 压缩编码器和解码器
        compressed_dim = int(dim * compression_ratio)
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, compressed_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
        
        # 相似度计算层
        self.similarity = nn.CosineSimilarity(dim=-1)
        
        # 加载已存储的数据
        if self.storage_path and self.storage_path.exists():
            self.load_storage()
            
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """压缩输入数据"""
        return self.encoder(x)
        
    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        """解压缩数据"""
        return self.decoder(x)
        
    def store(
        self,
        key_features: torch.Tensor,
        value: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        存储数据到长期存储
        
        Args:
            key_features: 键特征向量
            value: 要存储的值
            metadata: 可选的元数据
            
        Returns:
            index: 存储位置索引
        """
        # 如果存储已满，覆盖最旧的数据
        if self.current_index >= self.storage_size:
            self.current_index = 0
            
        index = self.current_index
        
        # 存储数据
        self.keys[index] = key_features
        self.values[index] = value
        self.compressed_values[index] = self.compress(value)
        self.metadata[index] = metadata or {}
        self.timestamps[index] = time.time()
        
        self.current_index += 1
        
        # 定期保存到磁盘
        if self.storage_path and self.current_index % 1000 == 0:
            self.save_storage()
            
        return index
        
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 1,
        threshold: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        检索最相似的k个记录
        
        Args:
            query: 查询向量
            k: 返回的结果数量
            threshold: 相似度阈值
            
        Returns:
            values: 检索到的值
            similarities: 相似度分数
            metadata_list: 元数据列表
        """
        # 计算查询向量与所有键的相似度
        similarities = self.similarity(
            query.unsqueeze(0),
            self.keys[:self.current_index]
        )
        
        # 找到最相似的k个记录
        top_k = min(k, len(similarities))
        top_similarities, indices = torch.topk(similarities, top_k)
        
        # 过滤低于阈值的结果
        mask = top_similarities >= threshold
        filtered_indices = indices[mask]
        filtered_similarities = top_similarities[mask]
        
        # 获取对应的值和元数据
        values = self.values[filtered_indices]
        metadata_list = [self.metadata[i.item()] for i in filtered_indices]
        
        return values, filtered_similarities, metadata_list
        
    def save_storage(self) -> None:
        """保存存储内容到磁盘"""
        if not self.storage_path:
            return
            
        # 创建存储目录
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存张量数据
        torch.save({
            'keys': self.keys[:self.current_index],
            'values': self.values[:self.current_index],
            'compressed_values': self.compressed_values[:self.current_index],
            'timestamps': self.timestamps[:self.current_index],
        }, self.storage_path / 'tensors.pt')
        
        # 保存元数据
        with open(self.storage_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata[:self.current_index], f)
            
    def load_storage(self) -> None:
        """从磁盘加载存储内容"""
        if not self.storage_path.exists():
            return
            
        # 加载张量数据
        tensor_data = torch.load(self.storage_path / 'tensors.pt')
        self.keys[:len(tensor_data['keys'])] = tensor_data['keys']
        self.values[:len(tensor_data['values'])] = tensor_data['values']
        self.compressed_values[:len(tensor_data['compressed_values'])] = tensor_data['compressed_values']
        self.timestamps[:len(tensor_data['timestamps'])] = tensor_data['timestamps']
        
        # 加载元数据
        with open(self.storage_path / 'metadata.json', 'r') as f:
            loaded_metadata = json.load(f)
            self.metadata[:len(loaded_metadata)] = loaded_metadata
            
        self.current_index = len(tensor_data['keys'])
        
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            'total_capacity': self.storage_size,
            'used_capacity': self.current_index,
            'compression_ratio': self.compression_ratio,
            'oldest_record_time': self.timestamps[0].item(),
            'newest_record_time': self.timestamps[self.current_index - 1].item(),
        }
        
    def clear(self) -> None:
        """清空存储"""
        self.keys.zero_()
        self.values.zero_()
        self.compressed_values.zero_()
        self.metadata = [{} for _ in range(self.storage_size)]
        self.timestamps.zero_()
        self.current_index = 0
        
        if self.storage_path:
            self.save_storage() 