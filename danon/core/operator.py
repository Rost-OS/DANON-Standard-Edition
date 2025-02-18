"""
动态算子基础类
实现了自适应计算单元的核心功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class DynamicOperator(nn.Module):
    """动态自适应神经算子的基础类"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # 动态特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 自适应计算单元
        self.adaptive_compute = nn.ModuleDict({
            'light': nn.Linear(hidden_dim, hidden_dim),
            'medium': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ),
            'heavy': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        })
        
        # 计算复杂度评估器
        self.complexity_estimator = nn.Linear(hidden_dim, 3)
        
        # 输出映射
        self.output_mapping = nn.Linear(hidden_dim, output_dim)
        
    def estimate_complexity(self, features: torch.Tensor) -> torch.Tensor:
        """评估输入数据的计算复杂度"""
        logits = self.complexity_estimator(features)
        return F.softmax(logits, dim=-1)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            mask: 可选的掩码张量
            
        Returns:
            output: 输出张量，形状为 [batch_size, seq_len, output_dim]
            meta: 包含中间状态的元信息字典
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 评估复杂度
        complexity_scores = self.estimate_complexity(features.mean(dim=1))
        
        # 根据复杂度选择计算路径
        outputs = []
        for compute_unit, score in zip(
            ['light', 'medium', 'heavy'],
            complexity_scores.unbind(dim=-1)
        ):
            unit_output = self.adaptive_compute[compute_unit](features)
            outputs.append(unit_output * score.unsqueeze(-1).unsqueeze(-1))
            
        # 合并不同计算路径的结果
        combined = sum(outputs)
        
        # 输出映射
        output = self.output_mapping(combined)
        
        # 收集元信息
        meta = {
            'complexity_scores': complexity_scores,
            'features': features,
        }
        
        return output, meta
        
    def extra_repr(self) -> str:
        """返回额外的字符串表示"""
        return f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, output_dim={self.output_dim}' 