"""
语义检索引擎
实现了基于双塔模型的高效语义检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
import faiss
import numpy as np
from dataclasses import dataclass
from threading import Lock
import logging
from pathlib import Path
import json
import time
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SearchResult:
    """检索结果"""
    id: int
    score: float
    embedding: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None

class TowerBlock(nn.Module):
    """编码塔基础模块"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # 构建多层编码网络
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        # 注意力层
        if use_attention:
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads=8,
                dropout=dropout
            )
            
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 特征提取
        features = self.layers(x)
        
        # 应用注意力机制
        if self.use_attention and mask is not None:
            attended, _ = self.attention(
                features,
                features,
                features,
                key_padding_mask=mask
            )
            features = features + attended
            
        # 输出映射
        return self.output_layer(features)

class DistributedIndex:
    """分布式索引"""
    def __init__(
        self,
        dim: int,
        num_shards: int = 4,
        index_type: str = 'IVF',
        num_clusters: int = 100
    ):
        self.dim = dim
        self.num_shards = num_shards
        self.index_type = index_type
        self.num_clusters = num_clusters
        
        # 创建分片索引
        self.shards = [
            self._create_shard()
            for _ in range(num_shards)
        ]
        
        self.shard_sizes = [0] * num_shards
        self._lock = Lock()
        
    def _create_shard(self) -> faiss.Index:
        """创建索引分片"""
        if self.index_type == 'Flat':
            return faiss.IndexFlatIP(self.dim)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.dim,
                self.num_clusters,
                faiss.METRIC_INNER_PRODUCT
            )
            if not index.is_trained:
                # 生成随机训练数据
                train_size = max(self.num_clusters * 100, 10000)
                train_data = np.random.normal(
                    size=(train_size, self.dim)
                ).astype('float32')
                index.train(train_data)
            return index
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
            
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """添加向量到索引"""
        n = len(vectors)
        if ids is None:
            with self._lock:
                start_id = sum(self.shard_sizes)
                ids = np.arange(start_id, start_id + n)
                
        # 计算向量分配
        assignments = self._assign_vectors(vectors)
        
        # 添加到各个分片
        for shard_id in range(self.num_shards):
            mask = assignments == shard_id
            if not np.any(mask):
                continue
                
            shard_vectors = vectors[mask]
            shard_ids = ids[mask]
            
            with self._lock:
                self.shards[shard_id].add(shard_vectors, shard_ids)
                self.shard_sizes[shard_id] += len(shard_vectors)
                
        return ids
        
    def _assign_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """将向量分配到不同的分片"""
        # 使用向量的一些特征来决定分片
        # 这里使用简单的轮询策略，实际应用中可以使用更复杂的策略
        n = len(vectors)
        return np.arange(n) % self.num_shards
        
    def search(
        self,
        queries: np.ndarray,
        k: int,
        min_score: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """搜索最相似的向量"""
        n = len(queries)
        all_scores = np.zeros((n, k), dtype='float32')
        all_ids = np.zeros((n, k), dtype='int64')
        
        # 在每个分片上搜索
        shard_results = []
        for shard in self.shards:
            scores, ids = shard.search(queries, k)
            shard_results.append((scores, ids))
            
        # 合并结果
        for i in range(n):
            all_candidates = []
            for scores, ids in shard_results:
                mask = scores[i] >= min_score
                all_candidates.extend([
                    (score, id_)
                    for score, id_ in zip(scores[i][mask], ids[i][mask])
                ])
                
            # 排序并选择top-k
            all_candidates.sort(reverse=True)
            top_k = all_candidates[:k]
            
            if top_k:
                all_scores[i], all_ids[i] = zip(*top_k)
                
        return all_scores, all_ids
        
    def save(self, path: str):
        """保存索引"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存每个分片
        for i, shard in enumerate(self.shards):
            shard_path = path / f"shard_{i}.index"
            faiss.write_index(shard, str(shard_path))
            
        # 保存元数据
        meta = {
            'dim': self.dim,
            'num_shards': self.num_shards,
            'index_type': self.index_type,
            'num_clusters': self.num_clusters,
            'shard_sizes': self.shard_sizes
        }
        with open(path / "meta.json", 'w') as f:
            json.dump(meta, f)
            
    @classmethod
    def load(cls, path: str) -> 'DistributedIndex':
        """加载索引"""
        path = Path(path)
        
        # 加载元数据
        with open(path / "meta.json") as f:
            meta = json.load(f)
            
        # 创建索引实例
        index = cls(
            dim=meta['dim'],
            num_shards=meta['num_shards'],
            index_type=meta['index_type'],
            num_clusters=meta['num_clusters']
        )
        
        # 加载分片
        for i in range(meta['num_shards']):
            shard_path = path / f"shard_{i}.index"
            index.shards[i] = faiss.read_index(str(shard_path))
            
        index.shard_sizes = meta['shard_sizes']
        return index

class SemanticRetriever(nn.Module):
    """语义检索引擎"""
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        temperature: float = 0.1,
        num_shards: int = 4,
        index_type: str = 'IVF',
        num_clusters: int = 100,
        cache_size: int = 10000,
        num_workers: int = 4
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # 查询编码器
        self.query_tower = TowerBlock(
            input_dim=query_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            use_attention=True
        )
        
        # 键编码器
        self.key_tower = TowerBlock(
            input_dim=key_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            use_attention=True
        )
        
        # 分布式索引
        self.index = DistributedIndex(
            dim=embedding_dim,
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
        
    def encode_query(
        self,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码查询"""
        return F.normalize(self.query_tower(query, mask), dim=-1)
        
    def encode_key(
        self,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """编码键"""
        return F.normalize(self.key_tower(key, mask), dim=-1)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算查询和键的相似度"""
        # 编码
        query_emb = self.encode_query(query, query_mask)
        key_emb = self.encode_key(key, key_mask)
        
        # 计算相似度
        sim = torch.matmul(query_emb, key_emb.t()) / self.temperature
        
        return sim, query_emb, key_emb
        
    def add_items(
        self,
        keys: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
        key_mask: Optional[torch.Tensor] = None
    ) -> List[int]:
        """添加新项到索引"""
        # 编码键
        with torch.no_grad():
            embeddings = self.encode_key(keys, key_mask).cpu().numpy()
            
        # 分配ID并添加到索引
        with self._lock:
            start_id = self.next_id
            ids = np.arange(start_id, start_id + len(keys))
            self.next_id = start_id + len(keys)
            
            # 存储元数据
            if metadata is not None:
                for i, id_ in enumerate(ids):
                    self.metadata[id_] = metadata[i]
                    
        # 添加到分布式索引
        self.index.add(embeddings, ids)
        
        return ids.tolist()
        
    async def search_async(
        self,
        query: torch.Tensor,
        k: int = 10,
        min_score: float = 0.0,
        query_mask: Optional[torch.Tensor] = None
    ) -> List[SearchResult]:
        """异步搜索"""
        # 生成缓存键
        cache_key = hash(query.cpu().numpy().tobytes())
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # 编码查询
        with torch.no_grad():
            query_emb = self.encode_query(query, query_mask).cpu().numpy()
            
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
        query: torch.Tensor,
        k: int = 10,
        min_score: float = 0.0,
        query_mask: Optional[torch.Tensor] = None
    ) -> List[SearchResult]:
        """同步搜索"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.search_async(query, k, min_score, query_mask)
        )
        
    def batch_search(
        self,
        queries: torch.Tensor,
        k: int = 10,
        min_score: float = 0.0,
        query_mask: Optional[torch.Tensor] = None
    ) -> List[List[SearchResult]]:
        """批量搜索"""
        loop = asyncio.get_event_loop()
        tasks = [
            self.search_async(
                query,
                k,
                min_score,
                None if query_mask is None else query_mask[i]
            )
            for i, query in enumerate(queries)
        ]
        return loop.run_until_complete(asyncio.gather(*tasks))
        
    def save(self, path: str):
        """保存检索器"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save({
            'query_tower': self.query_tower.state_dict(),
            'key_tower': self.key_tower.state_dict(),
            'metadata': self.metadata,
            'next_id': self.next_id
        }, path / "model.pt")
        
        # 保存索引
        self.index.save(str(path / "index"))
        
    @classmethod
    def load(cls, path: str, **kwargs) -> 'SemanticRetriever':
        """加载检索器"""
        path = Path(path)
        
        # 加载模型
        state = torch.load(path / "model.pt")
        
        # 创建实例
        retriever = cls(**kwargs)
        retriever.query_tower.load_state_dict(state['query_tower'])
        retriever.key_tower.load_state_dict(state['key_tower'])
        retriever.metadata = state['metadata']
        retriever.next_id = state['next_id']
        
        # 加载索引
        retriever.index = DistributedIndex.load(str(path / "index"))
        
        return retriever
        
    def __del__(self):
        """清理资源"""
        self.thread_pool.shutdown()