"""
上下文感知检索器
实现了基于MSRA的动态上下文编码和查询扩展机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from .semantic import SearchResult, TowerBlock
from .multimodal import MultiModalIndex, ModalityConfig
from ..core.attention import MSRAConfig, MSRAModel
import numpy as np

@dataclass
class ContextConfig:
    """上下文配置"""
    max_context_length: int = 10
    context_dim: int = 512
    num_attention_heads: int = 8
    dropout: float = 0.1
    temperature: float = 0.1

class ContextEncoder(nn.Module):
    """基于MSRA的上下文编码器"""
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        
        # 使用MSRA进行上下文处理
        msra_config = MSRAConfig(
            hidden_size=hidden_size,
            num_levels=3,
            chunk_size=256,
            compression_factor=4
        )
        self.context_processor = MSRAModel(msra_config)
        
        # 上下文融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        context_items: torch.Tensor,
        query: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float]]:
        # 处理上下文序列
        processed_context, importance_scores = self.context_processor(context_items)
        
        # 融合查询和处理后的上下文
        expanded_query = query.unsqueeze(1).expand(-1, processed_context.size(1), -1)
        combined = torch.cat([processed_context, expanded_query], dim=-1)
        
        # 生成上下文感知的查询表示
        context_aware_query = self.fusion_net(combined)
        
        # 提取重要性分数
        importance_weights = [score.mean().item() for score in importance_scores[0]]
        
        return context_aware_query, importance_weights

class ContextualRetriever(nn.Module):
    """上下文感知检索器"""
    def __init__(
        self,
        base_retriever: Union[TowerBlock, MultiModalIndex],
        config: Optional[ContextConfig] = None
    ):
        super().__init__()
        self.base_retriever = base_retriever
        self.config = config or ContextConfig()
        
        # 上下文编码器
        self.context_encoder = ContextEncoder(self.config.context_dim)
        
        # 相关性重排序网络
        self.reranker = nn.Sequential(
            nn.Linear(self.config.context_dim * 3, self.config.context_dim),
            nn.LayerNorm(self.config.context_dim),
            nn.GELU(),
            nn.Linear(self.config.context_dim, 1),
            nn.Sigmoid()
        )
        
    def expand_query(
        self,
        query: torch.Tensor,
        context: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[float]]:
        """使用上下文扩展查询"""
        if not context:
            return query, []
            
        # 将上下文堆叠为批次
        context_batch = torch.stack(context[-self.config.max_context_length:])
        
        # 使用上下文编码器处理
        expanded_query, importance_weights = self.context_encoder(
            context_batch,
            query
        )
        
        return expanded_query.mean(dim=1), importance_weights
        
    def rerank_results(
        self,
        results: List[SearchResult],
        query: torch.Tensor,
        context: List[torch.Tensor]
    ) -> List[SearchResult]:
        """基于上下文重新排序结果"""
        if not results or not context:
            return results
            
        # 准备输入
        result_embeddings = torch.stack([r.embedding for r in results])
        context_vector = torch.stack(context[-self.config.max_context_length:]).mean(0)
        
        # 计算相关性分数
        inputs = torch.cat([
            result_embeddings,
            query.expand(len(results), -1),
            context_vector.expand(len(results), -1)
        ], dim=-1)
        
        relevance_scores = self.reranker(inputs).squeeze(-1)
        
        # 重新排序
        sorted_indices = torch.argsort(relevance_scores, descending=True)
        return [results[i] for i in sorted_indices]
        
    def search(
        self,
        query: torch.Tensor,
        context: Optional[List[torch.Tensor]] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """执行上下文感知检索"""
        if context:
            # 扩展查询
            expanded_query, importance_weights = self.expand_query(query, context)
            
            # 使用基础检索器搜索
            results = self.base_retriever.search(expanded_query, top_k=top_k * 2, **kwargs)
            
            # 重新排序
            results = self.rerank_results(results, query, context)
            
            # 截取top-k结果
            results = results[:top_k]
            
            # 添加重要性信息
            for r in results:
                r.metadata['context_importance'] = importance_weights
                r.metadata['context_enhanced'] = True
        else:
            # 无上下文时直接使用基础检索器
            results = self.base_retriever.search(query, top_k=top_k, **kwargs)
            for r in results:
                r.metadata['context_enhanced'] = False
            
        return results
        
    def batch_search(
        self,
        queries: torch.Tensor,
        contexts: Optional[List[List[torch.Tensor]]] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[List[SearchResult]]:
        """批量执行上下文感知检索"""
        batch_size = queries.size(0)
        if contexts is not None and len(contexts) != batch_size:
            raise ValueError("contexts长度必须与queries的batch_size相匹配")
            
        results = []
        for i in range(batch_size):
            query = queries[i]
            context = contexts[i] if contexts is not None else None
            results.append(self.search(query, context, top_k, **kwargs))
            
        return results
        
    def update_context_config(self, new_config: ContextConfig) -> None:
        """更新上下文配置"""
        self.config = new_config
        self.context_encoder = ContextEncoder(new_config.context_dim)
        
    def get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'base_retriever': self.base_retriever.state_dict(),
            'context_encoder': self.context_encoder.state_dict(),
            'reranker': self.reranker.state_dict(),
            'config': self.config.__dict__
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态"""
        self.base_retriever.load_state_dict(state_dict['base_retriever'])
        self.context_encoder.load_state_dict(state_dict['context_encoder'])
        self.reranker.load_state_dict(state_dict['reranker'])
        self.config = ContextConfig(**state_dict['config'])
        
    def optimize_context_window(
        self,
        query: torch.Tensor,
        context: List[torch.Tensor],
        max_window_size: int = 5
    ) -> List[torch.Tensor]:
        """优化上下文窗口大小"""
        if not context or len(context) <= max_window_size:
            return context
            
        # 计算每个上下文项与查询的相关性
        similarities = torch.stack([
            F.cosine_similarity(query.unsqueeze(0), ctx.unsqueeze(0))
            for ctx in context
        ])
        
        # 选择最相关的上下文项
        _, indices = similarities.topk(max_window_size)
        indices = sorted(indices.tolist())  # 保持时间顺序
        
        return [context[i] for i in indices]
        
    def cache_results(
        self,
        query: torch.Tensor,
        results: List[SearchResult],
        cache_size: int = 1000
    ) -> None:
        """缓存检索结果"""
        if not hasattr(self, '_results_cache'):
            self._results_cache = {}
            self._cache_order = []
            
        # 生成缓存键
        cache_key = str(query.cpu().numpy().tobytes())
        
        # 更新缓存
        self._results_cache[cache_key] = results
        self._cache_order.append(cache_key)
        
        # 维护缓存大小
        if len(self._results_cache) > cache_size:
            old_key = self._cache_order.pop(0)
            del self._results_cache[old_key]
            
    def get_cached_results(
        self,
        query: torch.Tensor,
        similarity_threshold: float = 0.95
    ) -> Optional[List[SearchResult]]:
        """获取缓存的检索结果"""
        if not hasattr(self, '_results_cache'):
            return None
            
        # 计算查询与缓存键的相似度
        for cache_key in self._cache_order:
            cached_query = torch.from_numpy(
                np.frombuffer(cache_key.encode(), dtype=np.float32)
            ).reshape_as(query)
            
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                cached_query.unsqueeze(0)
            )
            
            if similarity > similarity_threshold:
                return self._results_cache[cache_key]
                
        return None
        
    def clear_cache(self) -> None:
        """清空结果缓存"""
        if hasattr(self, '_results_cache'):
            self._results_cache.clear()
            self._cache_order.clear() 