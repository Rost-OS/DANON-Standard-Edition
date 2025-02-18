"""
多模态索引
实现了跨模态表示对齐和统一索引结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import faiss
import numpy as np
from dataclasses import dataclass
from threading import Lock
from .semantic import TowerBlock, SearchResult, DistributedIndex
import logging
from pathlib import Path
import json
import time
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ModalityConfig:
    """模态配置"""
    name: str
    input_dim: int
    hidden_dim: int
    embedding_dim: int
    num_layers: int = 3
    dropout: float = 0.1
    use_attention: bool = True

class CrossModalAttention(nn.Module):
    """跨模态注意力"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力
        attended_query = self.norm1(query)
        attended_query = self.attention(
            attended_query,
            attended_query,
            attended_query,
            attn_mask=mask
        )[0]
        query = query + self.dropout(attended_query)
        
        # 跨模态注意力
        cross_attended = self.norm2(query)
        cross_attended = self.attention(
            cross_attended,
            key,
            value,
            attn_mask=mask
        )[0]
        
        # 门控融合
        gate = self.gate(torch.cat([query, cross_attended], dim=-1))
        output = query + gate * self.dropout(cross_attended)
        
        return output

class ModalityFusionNetwork(nn.Module):
    """模态融合网络"""
    def __init__(
        self,
        modality_dims: Dict[str, int],
        shared_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.shared_dim = shared_dim
        
        # 模态特定的投影层
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for name, dim in modality_dims.items()
        })
        
        # 跨模态注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(
                shared_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim * len(modality_dims), shared_dim * 2),
            nn.LayerNorm(shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim)
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 投影每个模态
        projected = {
            name: self.projectors[name](x)
            for name, x in inputs.items()
        }
        
        # 跨模态注意力
        attended = {}
        for name, x in projected.items():
            # 收集其他模态的表示
            other_modalities = {
                k: v for k, v in projected.items()
                if k != name
            }
            
            # 应用跨模态注意力
            attended_x = x
            for layer in self.cross_attention_layers:
                for other_name, other_x in other_modalities.items():
                    mask = None
                    if masks and other_name in masks:
                        mask = masks[other_name]
                    attended_x = layer(
                        attended_x,
                        other_x,
                        other_x,
                        mask
                    )
            attended[name] = attended_x
            
        # 融合所有模态
        concat = torch.cat(list(attended.values()), dim=-1)
        fused = self.fusion(concat)
        
        return F.normalize(fused, dim=-1), attended

class MultiModalRetriever(nn.Module):
    """多模态检索器"""
    def __init__(
        self,
        modality_configs: List[ModalityConfig],
        shared_dim: int = 512,
        num_fusion_layers: int = 3,
        num_heads: int = 8,
        temperature: float = 0.1,
        num_shards: int = 4,
        index_type: str = 'IVF',
        num_clusters: int = 100,
        cache_size: int = 10000,
        num_workers: int = 4
    ):
        super().__init__()
        self.modality_configs = {cfg.name: cfg for cfg in modality_configs}
        self.shared_dim = shared_dim
        self.temperature = temperature
        
        # 为每个模态创建编码器
        self.encoders = nn.ModuleDict()
        for cfg in modality_configs:
            self.encoders[cfg.name] = TowerBlock(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.embedding_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                use_attention=cfg.use_attention
            )
            
        # 模态融合网络
        self.fusion_network = ModalityFusionNetwork(
            modality_dims={
                name: cfg.embedding_dim
                for name, cfg in self.modality_configs.items()
            },
            shared_dim=shared_dim,
            num_layers=num_fusion_layers,
            num_heads=num_heads
        )
        
        # 分布式索引
        self.index = DistributedIndex(
            dim=shared_dim,
            num_shards=num_shards,
            index_type=index_type,
            num_clusters=num_clusters
        )
        
        # 结果缓存
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        # 存储元数据
        self.metadata = {}
        self.next_id = 0
        self._lock = Lock()
        
    def encode_modality(
        self,
        x: torch.Tensor,
        modality: str,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码单个模态的输入"""
        if modality not in self.encoders:
            raise ValueError(f"未知的模态: {modality}")
            
        return self.encoders[modality](x, mask)
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        # 编码每个模态
        encoded = {}
        for modality, x in inputs.items():
            mask = masks.get(modality) if masks else None
            encoded[modality] = self.encode_modality(x, modality, mask)
            
        # 融合不同模态
        return self.fusion_network(encoded, masks)
        
    def add_items(
        self,
        inputs: Dict[str, torch.Tensor],
        metadata: Optional[List[Dict[str, Any]]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[int]:
        """添加多模态项到索引"""
        # 编码和融合
        with torch.no_grad():
            embeddings, _ = self.forward(inputs, masks)
            embeddings = embeddings.cpu().numpy()
            
        # 分配ID并添加到索引
        with self._lock:
            start_id = self.next_id
            ids = np.arange(start_id, start_id + len(embeddings))
            self.next_id = start_id + len(embeddings)
            
            # 存储元数据
            if metadata is not None:
                for i, id_ in enumerate(ids):
                    self.metadata[id_] = metadata[i]
                    
        # 添加到分布式索引
        self.index.add(embeddings, ids)
        
        return ids.tolist()
        
    async def search_async(
        self,
        query: Union[torch.Tensor, Dict[str, torch.Tensor]],
        modality: Optional[str] = None,
        k: int = 10,
        min_score: float = 0.0,
        mask: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ) -> List[SearchResult]:
        """异步搜索"""
        # 准备查询
        if isinstance(query, torch.Tensor) and modality:
            query_dict = {modality: query}
            mask_dict = {modality: mask} if mask is not None else None
        elif isinstance(query, dict):
            query_dict = query
            mask_dict = mask if isinstance(mask, dict) else None
        else:
            raise ValueError("查询必须是张量（带模态名称）或模态字典")
            
        # 生成缓存键
        cache_key = hash(str(query_dict))
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # 编码查询
        with torch.no_grad():
            query_emb, _ = self.forward(query_dict, mask_dict)
            query_emb = query_emb.cpu().numpy()
            
        # 在线程池中执行搜索
        scores, indices = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.index.search,
            query_emb,
            k,
            min_score
        )
        
        # 构建结果
        results = []
        for score_list, idx_list in zip(scores, indices):
            for score, idx in zip(score_list, idx_list):
                if idx == -1 or score < min_score:
                    continue
                    
                # 获取embedding
                embedding = torch.from_numpy(
                    self.index.shards[idx % self.index.num_shards].reconstruct(idx)
                )
                
                # 创建结果对象
                result = SearchResult(
                    id=int(idx),
                    score=float(score),
                    embedding=embedding,
                    metadata=self.metadata.get(idx)
                )
                results.append(result)
                
        # 更新缓存
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[cache_key] = results
        
        return results
        
    def search(
        self,
        query: Union[torch.Tensor, Dict[str, torch.Tensor]],
        modality: Optional[str] = None,
        k: int = 10,
        min_score: float = 0.0,
        mask: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    ) -> List[SearchResult]:
        """同步搜索"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.search_async(query, modality, k, min_score, mask)
        )
        
    def batch_search(
        self,
        queries: Union[torch.Tensor, Dict[str, List[torch.Tensor]]],
        modality: Optional[str] = None,
        k: int = 10,
        min_score: float = 0.0,
        masks: Optional[Union[torch.Tensor, Dict[str, List[torch.Tensor]]]] = None
    ) -> List[List[SearchResult]]:
        """批量搜索"""
        # 准备查询
        if isinstance(queries, torch.Tensor) and modality:
            query_dicts = [
                {modality: query}
                for query in queries
            ]
            mask_dicts = [
                {modality: mask[i]} if masks is not None else None
                for i in range(len(queries))
            ]
        elif isinstance(queries, dict):
            batch_size = len(next(iter(queries.values())))
            query_dicts = [
                {
                    name: queries[name][i]
                    for name in queries
                }
                for i in range(batch_size)
            ]
            if masks:
                mask_dicts = [
                    {
                        name: masks[name][i]
                        for name in masks
                    }
                    for i in range(batch_size)
                ]
            else:
                mask_dicts = [None] * batch_size
        else:
            raise ValueError("查询格式无效")
            
        # 执行批量搜索
        loop = asyncio.get_event_loop()
        tasks = [
            self.search_async(
                query_dict,
                k=k,
                min_score=min_score,
                mask=mask_dict
            )
            for query_dict, mask_dict in zip(query_dicts, mask_dicts)
        ]
        return loop.run_until_complete(asyncio.gather(*tasks))
        
    def save(self, path: str):
        """保存检索器"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save({
            'encoders': {
                name: encoder.state_dict()
                for name, encoder in self.encoders.items()
            },
            'fusion_network': self.fusion_network.state_dict(),
            'metadata': self.metadata,
            'next_id': self.next_id
        }, path / "model.pt")
        
        # 保存索引
        self.index.save(str(path / "index"))
        
    @classmethod
    def load(cls, path: str, **kwargs) -> 'MultiModalRetriever':
        """加载检索器"""
        path = Path(path)
        
        # 加载模型
        state = torch.load(path / "model.pt")
        
        # 创建实例
        retriever = cls(**kwargs)
        
        # 加载编码器
        for name, encoder_state in state['encoders'].items():
            retriever.encoders[name].load_state_dict(encoder_state)
            
        # 加载融合网络
        retriever.fusion_network.load_state_dict(state['fusion_network'])
        
        # 加载元数据
        retriever.metadata = state['metadata']
        retriever.next_id = state['next_id']
        
        # 加载索引
        retriever.index = DistributedIndex.load(str(path / "index"))
        
        return retriever
        
    def __del__(self):
        """清理资源"""
        self.thread_pool.shutdown() 