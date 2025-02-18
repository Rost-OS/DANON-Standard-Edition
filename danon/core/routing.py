"""
动态路由机制
实现了基于输入特征的动态计算路径选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

class DynamicRouter(nn.Module):
    """动态路由器，用于选择最优的计算路径"""
    
    def __init__(
        self,
        dim: int,
        num_routes: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_routes = num_routes
        self.temperature = temperature
        
        # 路由特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # 路由决策网络
        self.route_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_routes)
        )
        
        # 路由重要性评估器
        self.importance_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_routes),
            nn.Sigmoid()
        )
        
    def extract_routing_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取用于路由决策的特征"""
        return self.feature_extractor(x)
        
    def predict_routes(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测路由概率和重要性分数"""
        # 路由概率
        route_logits = self.route_predictor(features)
        if mask is not None:
            route_logits = route_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        route_probs = F.softmax(route_logits / self.temperature, dim=-1)
        
        # 重要性分数
        importance_scores = self.importance_estimator(features)
        if mask is not None:
            importance_scores = importance_scores.masked_fill(~mask.unsqueeze(-1), 0.0)
            
        return route_probs, importance_scores
        
    def forward(
        self,
        x: torch.Tensor,
        compute_functions: List[nn.Module],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            compute_functions: 可用的计算函数列表
            mask: 可选的掩码张量
            
        Returns:
            output: 输出张量，形状为 [batch_size, seq_len, dim]
            meta: 包含中间状态的元信息字典
        """
        assert len(compute_functions) == self.num_routes, \
            f"计算函数数量 ({len(compute_functions)}) 必须等于路由数量 ({self.num_routes})"
            
        # 提取路由特征
        routing_features = self.extract_routing_features(x)
        
        # 预测路由概率和重要性分数
        route_probs, importance_scores = self.predict_routes(routing_features, mask)
        
        # 对每个路由执行计算
        outputs = []
        for i, func in enumerate(compute_functions):
            # 计算当前路由的输出
            route_output = func(x)
            
            # 应用路由概率和重要性分数
            weighted_output = route_output * route_probs[..., i:i+1] * importance_scores[..., i:i+1]
            outputs.append(weighted_output)
            
        # 合并所有路由的输出
        combined_output = sum(outputs)
        
        # 收集元信息
        meta = {
            'route_probabilities': route_probs,
            'importance_scores': importance_scores,
            'routing_features': routing_features
        }
        
        return combined_output, meta
        
    def get_route_stats(self, route_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算路由统计信息"""
        stats = {
            'entropy': -(route_probs * torch.log(route_probs + 1e-10)).sum(-1).mean(),
            'max_prob': route_probs.max(-1)[0].mean(),
            'min_prob': route_probs.min(-1)[0].mean(),
            'route_usage': route_probs.mean(0)  # 每个路由的平均使用率
        }
        return stats 

class EnhancedDynamicRouter(nn.Module):
    """增强型动态路由器"""
    
    def __init__(
        self,
        dim: int,
        num_routes: int,
        num_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_routes = num_routes
        self.num_heads = num_heads
        self.base_temperature = temperature
        
        # 多头路由特征提取
        self.route_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // num_heads),
                nn.LayerNorm(dim // num_heads),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_heads)
        ])
        
        # 路由决策网络
        self.route_decision = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_routes)
        )
        
        # 重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_routes),
            nn.Sigmoid()
        )
        
        # 温度控制网络
        self.temperature_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 历史记录
        self.route_history = deque(maxlen=1000)
        self.importance_history = deque(maxlen=1000)
        self.temperature_history = deque(maxlen=1000)
        
    def extract_routing_features(self, x: torch.Tensor) -> torch.Tensor:
        """多头路由特征提取"""
        head_features = []
        for head in self.route_heads:
            head_out = head(x)
            head_features.append(head_out)
        
        # 合并多头特征
        combined_features = torch.cat(head_features, dim=-1)
        return combined_features
        
    def predict_routes(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测路由概率和重要性分数"""
        # 计算自适应温度
        temperature = self.base_temperature * self.temperature_net(features).mean()
        self.temperature_history.append(temperature.item())
        
        # 路由决策
        route_logits = self.route_decision(features)
        if mask is not None:
            route_logits = route_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # 应用温度缩放的softmax
        route_probs = F.softmax(route_logits / temperature, dim=-1)
        
        # 计算重要性分数
        importance_scores = self.importance_net(features)
        
        # 更新历史记录
        self.route_history.append(route_probs.mean(0).detach())
        self.importance_history.append(importance_scores.mean(0).detach())
        
        return route_probs, importance_scores
        
    def forward(
        self,
        x: torch.Tensor,
        compute_functions: List[nn.Module],
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向传播"""
        assert len(compute_functions) == self.num_routes, \
            f"计算函数数量 ({len(compute_functions)}) 必须等于路由数量 ({self.num_routes})"
            
        # 特征提取
        routing_features = self.extract_routing_features(x)
        
        # 路由预测
        route_probs, importance_scores = self.predict_routes(routing_features, mask)
        
        # 计算输出
        outputs = []
        route_outputs = {}
        for i, func in enumerate(compute_functions):
            # 计算当前路由的输出
            route_output = func(x)
            route_outputs[f'route_{i}'] = route_output
            
            # 应用路由概率和重要性分数
            weighted_output = route_output * route_probs[..., i:i+1] * importance_scores[..., i:i+1]
            outputs.append(weighted_output)
            
        # 合并输出
        combined_output = sum(outputs)
        
        # 收集元信息
        meta = {
            'route_probabilities': route_probs,
            'importance_scores': importance_scores,
            'routing_features': routing_features,
            'route_history': self.route_history,
            'importance_history': self.importance_history,
            'route_outputs': route_outputs,
            'temperature': torch.norm(routing_features, dim=-1).mean()
        }
        
        return combined_output, meta
        
    def get_route_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        if not self.route_history:
            return {}
            
        route_history = torch.stack(list(self.route_history))
        importance_history = torch.stack(list(self.importance_history))
        temperature_history = torch.tensor(list(self.temperature_history))
        
        return {
            'route_usage': route_history.mean(0),
            'route_std': route_history.std(0),
            'route_entropy': -(route_history * torch.log(route_history + 1e-10)).sum(-1).mean(),
            'importance_mean': importance_history.mean(0),
            'importance_std': importance_history.std(0),
            'temperature_mean': temperature_history.mean(),
            'temperature_std': temperature_history.std()
        } 