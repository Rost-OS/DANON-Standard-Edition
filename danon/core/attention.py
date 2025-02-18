"""
MSRA (Multi-Scale Recursive Attention) + DALA (Dynamic Adaptive Long-range Attention)

这个模块实现了两种创新的注意力机制：

1. MSRA：多尺度递归注意力
   - 通过动态压缩和递归处理来解决传统Transformer的限制
   - 支持多尺度特征融合
   - 包含自校准和稳定性增强机制
   - 适用于中等长度序列的高效处理

2. DALA：动态自适应长程注意力
   - 专门设计用于处理超长序列（最大支持100万token）
   - 使用重要性网络进行动态路由
   - 实现递归状态更新以维护长期依赖
   - 支持无限长度的序列处理

主要组件：
- DynamicCompression: 自适应序列压缩
- RecursiveAttention: 递归注意力计算
- StabilityEnhancer: 稳定性增强
- MultiScaleFeatureFusion: 多尺度特征融合
- ImportanceNetwork: 重要性评估
- DynamicRouter: 动态路由
- RecursiveStateUpdate: 状态更新
- InfiniteAttention: 无限长度处理

使用示例：
```python
# 创建MSRA模型
msra_model = create_msra_model(
    hidden_size=768,
    num_levels=3,
    num_layers=6
)

# 创建DALA模型
dala_model = create_dala_model(
    hidden_size=768,
    num_heads=12,
    num_layers=12
)

# 使用混合模型
hybrid_model = create_enhanced_hybrid_model(
    hidden_size=768,
    num_levels=3,
    num_layers=6
)
```

注意事项：
1. 对于序列长度小于10000的情况，推荐使用MSRA
2. 对于超长序列，推荐使用DALA
3. 如果序列长度动态变化，可以使用混合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
from collections import deque, OrderedDict
import time
from dataclasses import dataclass
from .attention_base import BaseAttentionConfig, BaseAttention, AttentionMonitor, AttentionError
from .unified_config import MSRAConfig, UnifiedAttentionConfig
from .monitoring import PerformanceMonitor, PerformanceMetrics, PerformanceContext
from .caching import CacheManager, CacheKey
from .error_handling import (
    ErrorHandler, BaseAttentionError, ComputationError, ResourceError,
    ValidationError
)
from .compatibility import ensure_compatibility

__all__ = [
    'MSRAConfig',
    'DynamicCompression',
    'RecursiveAttention',
    'MSRALayer',
    'MSRAModel',
    'AdaptiveSparsification',
    'MultiScaleFeatureFusion',
    'DynamicHeadPruning',
    'SelfCalibratingAttention',
    'StabilityEnhancer',
    'TheoreticalAnalyzer',
    'DALAConfig',
    'ImportanceNetwork',
    'DynamicRouter',
    'RecursiveStateUpdate',
    'InfiniteAttention',
    'DALALayer',
    'DALAModel',
    'HybridConfig',
    'HybridAttentionModel',
    'AdaptiveAttentionFusion',
    'EnhancedHybridAttentionModel',
    'HyperState',
    'SuperHybridAttentionModel'
]

@dataclass
class MSRAConfig(BaseAttentionConfig):
    """MSRA配置类 - 多尺度递归注意力"""
    # 模型结构特有配置
    num_layers: int = 32  # 增加层数
    compression_factor: int = 4
    
    # 激活函数配置
    activation: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    # 高级特性配置
    calibration_factor: float = 0.1
    bidirectional_flow: bool = True
    feature_fusion: bool = True
    
    # 稳定性增强配置
    stability_threshold: float = 0.1
    auto_calibration: bool = True
    theoretical_bounds: bool = True
    
    # 性能优化配置
    hidden_size: int = 8192  # 增大隐藏层大小
    intermediate_size: int = 32768  # 4倍hidden_size
    num_attention_heads: int = 64  # 增加注意力头数
    head_dim: int = 128  # 增加每个头的维度
    chunk_size: int = 4096  # 增大块大小
    gradient_checkpointing: bool = True  # 启用梯度检查点
    mixed_precision: bool = True  # 启用混合精度训练
    optimizer_states_in_half_precision: bool = True  # 优化器状态使用半精度
    
    def __post_init__(self):
        """初始化后的验证"""
        # 验证head_dim和hidden_size的关系
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal num_attention_heads * head_dim "
                f"({self.num_attention_heads * self.head_dim})"
            )
        
        # 验证intermediate_size
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                "intermediate_size must be larger than hidden_size"
            )
        
        # 验证性能优化配置
        if self.mixed_precision and not torch.cuda.is_available():
            logging.warning("Mixed precision training requires CUDA, but CUDA is not available")
            self.mixed_precision = False
            
        # 计算理论内存占用
        self.theoretical_memory = self._calculate_theoretical_memory()
        logging.info(f"Theoretical peak memory usage: {self.theoretical_memory / 1e9:.2f}GB")
        
    def _calculate_theoretical_memory(self) -> int:
        """计算理论峰值内存占用(字节)"""
        bytes_per_param = 4 if not self.mixed_precision else 2
        
        # 模型参数内存
        param_memory = (
            self.hidden_size * self.hidden_size * 4 +  # Q,K,V,O矩阵
            self.hidden_size * 2  # LayerNorm参数
        ) * bytes_per_param
        
        # 激活值内存(考虑梯度检查点)
        if self.gradient_checkpointing:
            activation_memory = self.chunk_size * self.hidden_size * 3 * bytes_per_param
        else:
            activation_memory = self.chunk_size * self.hidden_size * 6 * bytes_per_param
            
        # 注意力分数内存
        attention_memory = (
            self.num_attention_heads * self.chunk_size * self.chunk_size
        ) * bytes_per_param
        
        return param_memory + activation_memory + attention_memory

class StabilityEnhancer(nn.Module):
    """稳定性增强器：确保训练过程的稳定性"""
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.config = config
        self.base_threshold = config.stability_threshold
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 增强稳定器网络
        self.stabilizer = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        # 自适应阈值网络
        self.threshold_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 梯度缩放网络
        self.scale_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.history = deque(maxlen=1000)  # 用于追踪历史稳定性
        
        # 性能优化
        self.gradient_checkpointing = config.gradient_checkpointing
        self.mixed_precision = config.mixed_precision
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用混合精度训练
        with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
            # 计算当前批次的稳定性指标
            stability_score = torch.norm(x, dim=-1, keepdim=True)
            
            # 更新历史记录
            self.history.append(stability_score.mean().item())
            
            # 计算自适应阈值
            if self.gradient_checkpointing and self.training:
                adaptive_threshold = torch.utils.checkpoint.checkpoint(
                    lambda x: self.base_threshold * self.threshold_net(x).mean(),
                    x
                )
            else:
                adaptive_threshold = self.base_threshold * self.threshold_net(x).mean()
            
            # 计算梯度缩放因子
            scale_factor = self.scale_net(x)
            
            # 应用自适应正则化
            if torch.any(stability_score > adaptive_threshold):
                x = self.norm(x)
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(self.stabilizer, x)
                else:
                    x = self.stabilizer(x)
                x = x * scale_factor  # 应用梯度缩放
                
            # 添加残差连接和随机扰动
            x = x + torch.randn_like(x) * 0.001  # 小幅随机扰动提高鲁棒性
            
            return x
        
    def get_stability_stats(self) -> Dict[str, float]:
        """获取稳定性统计信息"""
        if not self.history:
            return {}
        return {
            'mean_stability': sum(self.history) / len(self.history),
            'max_stability': max(self.history),
            'min_stability': min(self.history),
            'std_stability': torch.tensor(list(self.history)).std().item()
        }

class SelfCalibratingAttention(BaseAttention):
    """自校准注意力机制：动态调整注意力权重"""
    def __init__(self, config: MSRAConfig):
        super().__init__(config)
        self.config = config
        self.calibration_factor = config.calibration_factor
        self.auto_calibration = config.auto_calibration
        
        # 增强自校准网络
        self.calibration_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # 性能优化
        self.gradient_checkpointing = config.gradient_checkpointing
        self.mixed_precision = config.mixed_precision
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 使用混合精度训练
        with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
            batch_size, seq_len, _ = hidden_states.size()
            
            # 分块处理长序列
            chunk_size = self.config.chunk_size
            num_chunks = math.ceil(seq_len / chunk_size)
            
            outputs = []
            chunk_stats = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, seq_len)
                
                # 获取当前块
                chunk_hidden = hidden_states[:, start_idx:end_idx, :]
                chunk_mask = attention_mask[:, :, start_idx:end_idx, :] if attention_mask is not None else None
                
                # 使用梯度检查点
                if self.gradient_checkpointing and self.training:
                    chunk_output = torch.utils.checkpoint.checkpoint(
                        super().forward,
                        chunk_hidden,
                        chunk_mask
                    )
                    chunk_output, stats = chunk_output[0], chunk_output[1]
                else:
                    chunk_output, stats = super().forward(chunk_hidden, chunk_mask)
                    
                outputs.append(chunk_output)
                chunk_stats.append(stats)
            
            # 合并所有块的输出
            output = torch.cat(outputs, dim=1)
            
            # 合并统计信息
            merged_stats = {
                'attention_scores': torch.cat([s['attention_scores'] for s in chunk_stats], dim=-1),
                'compute_time': sum(s['compute_time'] for s in chunk_stats),
                'memory_usage': max(s['memory_usage'] for s in chunk_stats),
                'num_chunks': num_chunks,
                'chunk_size': chunk_size
            }
            
            return output, merged_stats

class TheoreticalAnalyzer:
    """理论分析工具：验证模型行为是否符合理论预期"""
    def __init__(self, config: MSRAConfig):
        self.config = config
        self.bounds_history = []
        
    def check_theoretical_bounds(self, attention_weights: torch.Tensor) -> bool:
        """检查注意力权重是否在理论界限内"""
        if not self.config.theoretical_bounds:
            return True
            
        # 计算理论界限
        lower_bound = 0.0
        upper_bound = 1.0 / math.sqrt(attention_weights.size(-1))
        
        # 检查是否满足界限
        within_bounds = (attention_weights >= lower_bound).all() and \
                       (attention_weights <= upper_bound).all()
                       
        self.bounds_history.append(within_bounds)
        return within_bounds
        
    def get_analysis_report(self) -> Dict[str, Any]:
        """生成理论分析报告"""
        return {
            "bounds_satisfaction_rate": sum(self.bounds_history) / len(self.bounds_history),
            "total_checks": len(self.bounds_history),
            "recent_violations": not all(self.bounds_history[-10:]) if self.bounds_history else False
        }

class DynamicCompression(nn.Module):
    """动态序列压缩模块

    该模块实现了自适应的序列压缩机制，可以根据输入序列的重要性动态调整压缩比例。
    主要功能包括：
    1. 自适应压缩率预测
    2. 基于重要性的序列压缩
    3. 可选的量化操作
    4. 缓存机制以提高效率

    工作流程：
    1. 计算输入序列中每个位置的重要性分数
    2. 基于重要性分数预测最优压缩率
    3. 根据预测的压缩率执行压缩操作
    4. （可选）对压缩后的序列进行量化
    5. 更新压缩统计信息

    参数:
        dim (int): 输入特征的维度
        min_compression_ratio (float): 最小压缩比例，默认为0.1
            表示最少保留原序列长度的比例
        max_compression_ratio (float): 最大压缩比例，默认为0.9
            表示最多保留原序列长度的比例
        cache_size (int): 缓存大小，默认为1000
            用于存储最近的压缩结果，提高效率
        enable_adaptive_compression (bool): 是否启用自适应压缩，默认为True
            如果为False，将使用固定的压缩率
        importance_threshold (float): 重要性阈值，默认为0.5
            用于判断序列中的重要位置
        use_quantization (bool): 是否使用量化，默认为True
            启用后可以进一步减少内存占用

    属性:
        compression_stats (Dict): 存储压缩统计信息
        cache (OrderedDict): 压缩结果缓存
        importance_network (nn.Module): 重要性评估网络

    示例:
        ```python
        compressor = DynamicCompression(
            dim=512,
            min_compression_ratio=0.1,
            max_compression_ratio=0.8,
            enable_adaptive_compression=True
        )
        
        # 单独使用
        compressed_seq = compressor(input_sequence)
        
        # 获取压缩统计信息
        compressed_seq, stats = compressor(
            input_sequence,
            return_stats=True
        )
        ```

    注意:
        1. 输入序列的shape应为(batch_size, seq_len, dim)
        2. 压缩比例会根据序列的重要性动态调整
        3. 启用量化可能会导致一定的信息损失
        4. 缓存机制会占用额外的内存，请根据实际情况调整cache_size
    """

    def __init__(
        self,
        dim: int,
        min_compression_ratio: float = 0.1,
        max_compression_ratio: float = 0.9,
        cache_size: int = 1000,
        enable_adaptive_compression: bool = True,
        importance_threshold: float = 0.5,
        use_quantization: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.cache_size = cache_size
        self.enable_adaptive_compression = enable_adaptive_compression
        self.importance_threshold = importance_threshold
        self.use_quantization = use_quantization
        
        # 重要性评分网络
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 压缩率预测网络
        self.compression_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 自适应量化网络
        if use_quantization:
            self.quantization_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 8),  # 8种量化级别
                nn.Softmax(dim=-1)
            )
            
        # 缓存
        self.importance_cache = OrderedDict()
        self.compression_stats = {
            'total_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratios': [],
            'importance_scores': [],
            'quantization_levels': []
        }
        
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """计算序列中每个token的重要性分数"""
        # 检查缓存
        cache_key = hash(x.cpu().numpy().tobytes())
        if cache_key in self.importance_cache:
            self.cache_hits += 1
            return self.importance_cache[cache_key]
            
        self.cache_misses += 1
        
        # 计算重要性分数
        importance = self.importance_scorer(x)
        
        # 更新缓存
        if len(self.importance_cache) >= self.cache_size:
            # 移除最旧的缓存项
            self.importance_cache.popitem(last=False)
        self.importance_cache[cache_key] = importance
        
        return importance
        
    def _predict_compression_ratio(self, x: torch.Tensor) -> float:
        """预测最优压缩率"""
        if not self.enable_adaptive_compression:
            return (self.max_compression_ratio + self.min_compression_ratio) / 2
            
        # 基于输入特征预测压缩率
        pred = self.compression_predictor(x.mean(dim=1))
        
        # 将预测值映射到有效范围
        ratio = self.min_compression_ratio + (
            self.max_compression_ratio - self.min_compression_ratio
        ) * pred.item()
        
        return ratio
        
    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """自适应量化"""
        if not self.use_quantization:
            return x, None
            
        # 预测量化级别
        quant_probs = self.quantization_net(x.mean(dim=1))
        num_bits = torch.argmax(quant_probs, dim=1) + 1  # 1-8位量化
        
        # 应用量化
        scale = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        step = scale / (2 ** num_bits.view(-1, 1, 1))
        x_quant = torch.round(x / step) * step
        
        return x_quant, num_bits
        
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 计算重要性分数
        importance = self._compute_importance(x)
        
        # 预测压缩率
        compression_ratio = self._predict_compression_ratio(x)
        
        # 根据重要性分数选择保留的token
        num_tokens = int(seq_len * compression_ratio)
        _, indices = torch.topk(importance.squeeze(-1), num_tokens, dim=1)
        
        # 收集选中的token
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_tokens)
        selected = x[batch_indices, indices]
        
        # 应用量化
        if self.use_quantization:
            selected, quant_bits = self._quantize(selected)
        else:
            quant_bits = None
            
        # 更新统计信息
        self.compression_stats['total_tokens'] += batch_size * seq_len
        self.compression_stats['compressed_tokens'] += batch_size * num_tokens
        self.compression_stats['compression_ratios'].append(compression_ratio)
        self.compression_stats['importance_scores'].append(
            importance.mean().item()
        )
        if quant_bits is not None:
            self.compression_stats['quantization_levels'].append(
                quant_bits.float().mean().item()
            )
            
        if return_stats:
            stats = {
                'importance': importance.detach(),
                'compression_ratio': compression_ratio,
                'num_tokens': num_tokens,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses),
                'quantization_bits': quant_bits.float().mean().item() if quant_bits is not None else None,
                'compression_stats': self.compression_stats
            }
            return selected, stats
            
        return selected

class RecursiveAttention(BaseAttention):
    """递归注意力计算模块

    该模块实现了一种创新的递归注意力机制，通过多步递归处理来捕获序列中的长距离依赖关系。
    主要特点：
    1. 递归注意力计算
    2. 自适应步长控制
    3. 残差连接
    4. 梯度稳定性保证

    工作原理：
    1. 将输入序列分解为多个子序列
    2. 对每个子序列进行递归注意力计算
    3. 通过残差连接融合各步的结果
    4. 应用注意力稳定性增强
    5. 合并子序列的计算结果

    参数:
        config (MSRAConfig): MSRA配置对象
            包含模型的所有超参数设置

    属性:
        max_recursion_depth (int): 最大递归深度
        attention_dropout (float): 注意力dropout率
        residual_scale (float): 残差连接的缩放因子
        layer_norm (nn.LayerNorm): 层归一化
        recursion_tracker (Dict): 追踪递归计算的统计信息

    方法:
        sparsemax: 实现sparsemax激活函数
        compute_compression: 计算序列压缩
        forward: 前向传播函数

    示例:
        ```python
        config = MSRAConfig(
            num_layers=6,
            compression_factor=4
        )
        
        recursive_attn = RecursiveAttention(config)
        
        # 前向计算
        output = recursive_attn(
            input_tensor,
            mask=attention_mask
        )
        ```

    注意事项：
        1. 输入tensor的shape应为(batch_size, seq_len, hidden_size)
        2. 注意力mask的shape应为(batch_size, seq_len)
        3. 递归深度会根据序列长度自动调整
        4. 建议在使用时启用梯度检查点以节省显存
    """

    def __init__(self, config: MSRAConfig):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.num_levels = config.num_levels
        
        #实现 L(x) = ⊕(i=1 to n) l_i where l_i = C(x, r^(i)) + R(x)
        self.compression_layers = nn.ModuleList([
            DynamicCompression(config.hidden_size) for _ in range(self.num_levels)
        ])
        
        # 实现 C(x, r) = SparseMax(Importance(x)/τ) ⊙ (Wx/√r)
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        self.W = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 残差连接 R(x)
        self.residual_transform = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * self.num_levels, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def sparsemax(self, x: torch.Tensor) -> torch.Tensor:
        """实现 SparseMax 函数"""
        sorted_x, _ = torch.sort(x, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_x, dim=-1)
        arange = torch.arange(1, x.size(-1) + 1, device=x.device)
        threshold = (cumsum - 1) / arange
        mask = sorted_x > threshold
        sum_mask = mask.sum(dim=-1, keepdim=True)
        tau = (cumsum.gather(-1, sum_mask - 1) - 1) / sum_mask
        return torch.clamp(x - tau, min=0)

    def compute_compression(self, x: torch.Tensor, r: int) -> torch.Tensor:
        """实现 C(x, r) 函数"""
        # 计算重要性分数
        importance = self.importance_net(x)
        importance = importance / self.temperature
        
        # 应用 SparseMax
        attention_weights = self.sparsemax(importance)
        
        # 应用线性变换并进行缩放
        transformed = self.W(x) / math.sqrt(r)
        
        # 应用注意力权重
        return attention_weights * transformed

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # 存储每一层的输出
        level_outputs = []
        
        for i in range(self.num_levels):
            # 计算当前层的压缩率 r^(i)
            r = 2 ** (i * math.log2(2))  # r^(i) = 2^(i*log2(R))
            
            # 计算压缩表示
            compressed = self.compute_compression(x, int(r))
            
            # 添加残差连接
            residual = self.residual_transform(x)
            level_output = compressed + residual
            
            # 应用层归一化
            level_output = self.layer_norm(level_output)
            
            level_outputs.append(level_output)
        
        # 拼接所有层的输出
        concatenated = torch.cat(level_outputs, dim=-1)
        
        # 融合特征
        output = self.feature_fusion(concatenated)
        
        if mask is not None:
            output = output * mask.unsqueeze(-1)
            
        return output

class AdaptiveSparsification(nn.Module):
    """自适应稀疏化模块"""
    
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.1))
        self.min_sparsity = 0.1
        self.max_sparsity = 0.9
        
        # 自适应调整网络
        self.adaptation_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 循环缓冲区配置
        self.buffer_size = 1000  # 固定缓冲区大小
        self.current_index = 0
        
        # 历史统计
        self.register_buffer('sparsity_moving_avg', torch.tensor(0.5))
        self.register_buffer('importance_moving_avg', torch.zeros(1))
        self.register_buffer('sparsity_history', torch.zeros(self.buffer_size))
        self.register_buffer('importance_history', torch.zeros(self.buffer_size))
        self.momentum = 0.9
        
        # 内存监控
        self.memory_threshold = 0.9  # 90% 内存使用率阈值
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 3600  # 每小时清理一次
        
    def _update_moving_averages(self, sparsity: torch.Tensor, importance: torch.Tensor):
        """更新移动平均值，使用循环缓冲区"""
        # 更新历史数据
        self.sparsity_history[self.current_index] = sparsity.mean()
        self.importance_history[self.current_index] = importance.mean()
        
        # 更新索引
        self.current_index = (self.current_index + 1) % self.buffer_size
        
        # 计算有效值的掩码
        valid_mask = ~torch.isnan(self.sparsity_history)
        
        # 计算新的移动平均值
        if valid_mask.any():
            self.sparsity_moving_avg = self.sparsity_history[valid_mask].mean()
            self.importance_moving_avg = self.importance_history[valid_mask].mean()
    
    def _check_memory_usage(self):
        """检查内存使用情况并在必要时进行清理"""
        current_time = time.time()
        
        # 检查是否需要定期清理
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_old_data()
            self.last_cleanup_time = current_time
        
        # 检查内存使用率
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > self.memory_threshold:
                self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        # 保留最近的数据
        keep_size = self.buffer_size // 2
        if self.current_index >= keep_size:
            self.sparsity_history[keep_size:] = 0
            self.importance_history[keep_size:] = 0
        else:
            # 处理循环情况
            self.sparsity_history[self.current_index+keep_size:] = 0
            self.importance_history[self.current_index+keep_size:] = 0
            self.sparsity_history[:self.current_index] = 0
            self.importance_history[:self.current_index] = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 检查内存使用情况
        self._check_memory_usage()
        
        # 计算重要性分数
        importance_scores = self.adaptation_net(x).squeeze(-1)
        
        # 根据历史统计计算批次稀疏度
        batch_sparsity = torch.clamp(
            self.sparsity_moving_avg + self.sparsity_threshold,
            self.min_sparsity,
            self.max_sparsity
        )
        
        # 更新统计数据
        self._update_moving_averages(batch_sparsity, importance_scores)
        
        # 应用稀疏化
        mask = importance_scores > batch_sparsity
        x_sparse = x * mask
        
        return x_sparse, {
            'sparsity': batch_sparsity,
            'importance_scores': importance_scores,
            'sparsity_moving_avg': self.sparsity_moving_avg,
            'importance_moving_avg': self.importance_moving_avg,
            'memory_usage': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        }

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_levels = config.num_levels
        
        # 多尺度特征提取
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // (2 ** i)),
                nn.LayerNorm(config.hidden_size // (2 ** i)),
                nn.ReLU()
            )
            for i in range(self.num_levels)
        ])
        
        # 特征融合
        total_features = sum(config.hidden_size // (2 ** i) 
                           for i in range(self.num_levels))
        self.fusion = nn.Sequential(
            nn.Linear(total_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
        )
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, self.num_levels),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        features = []
        for extractor in self.extractors:
            features.append(extractor(x))
            
        # 计算注意力权重
        weights = self.attention(x)
        
        # 加权融合
        weighted_features = []
        for feat, weight in zip(features, weights.unbind(-1)):
            weighted_features.append(feat * weight.unsqueeze(-1))
            
        # 拼接并融合
        concat_features = torch.cat(weighted_features, dim=-1)
        fused = self.fusion(concat_features)
        
        return fused

class DynamicHeadPruning(nn.Module):
    """动态头剪枝模块"""
    
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 头重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.LayerNorm(self.head_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 动态阈值网络
        self.threshold_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 历史统计
        self.register_buffer('head_importance_history',
                           torch.zeros(config.num_attention_heads))
        self.register_buffer('pruning_rate_history', torch.tensor(0.))
        self.momentum = 0.9
        
    def forward(
        self,
        x: torch.Tensor,
        head_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = x.size(0)
        
        # 评估头的重要性
        head_importance = []
        for i in range(self.num_heads):
            head_output = head_outputs[:, :, i]
            importance = self.importance_net(head_output)
            head_importance.append(importance)
        head_importance = torch.cat(head_importance, dim=-1)
        
        # 计算动态阈值
        threshold = self.threshold_net(x.mean(dim=1))
        
        # 生成掩码
        mask = head_importance > threshold
        
        # 更新历史统计
        with torch.no_grad():
            self.head_importance_history = (
                self.momentum * self.head_importance_history +
                (1 - self.momentum) * head_importance.mean(0)
            )
            current_pruning_rate = (~mask).float().mean()
            self.pruning_rate_history = (
                self.momentum * self.pruning_rate_history +
                (1 - self.momentum) * current_pruning_rate
            )
        
        # 应用掩码
        pruned_outputs = head_outputs * mask.view(batch_size, 1, -1, 1)
        
        return pruned_outputs, {
            'head_importance': head_importance,
            'threshold': threshold,
            'mask': mask,
            'pruning_rate': current_pruning_rate,
            'importance_history': self.head_importance_history,
            'pruning_rate_history': self.pruning_rate_history
        }

class EnhancedSelfCalibratingAttention(nn.Module):
    """增强型自校准注意力机制"""
    def __init__(self, config: MSRAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_levels = config.num_levels
        
        # 压缩模块
        self.compressions = nn.ModuleList([
            DynamicCompression(config.hidden_size)
            for _ in range(config.num_levels)
        ])
        
        # 增强型自校准注意力
        self.attentions = nn.ModuleList([
            EnhancedSelfCalibratingAttention(config)
            for _ in range(config.num_levels)
        ])
        
        # 跨层特征增强
        self.feature_enhancers = nn.ModuleList([
            CrossLayerFeatureEnhancer(config)
            for _ in range(config.num_levels)
        ])
        
        # 层次特征融合
        self.level_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * config.num_levels, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        level_outputs = []
        level_stats = []
        current_states = hidden_states
        
        # 自底向上处理
        for level in range(self.num_levels):
            # 压缩
            compressed_states, importance = self.compressions[level](current_states)
            
            # 自校准注意力
            attended_states, attention_stats = self.attentions[level](
                compressed_states,
                attention_mask
            )
            
            # 特征增强
            prev_level = level_outputs[level-1] if level > 0 else None
            next_level = None  # 将在自顶向下阶段填充
            enhanced_states = self.feature_enhancers[level](
                attended_states,
                prev_level,
                next_level
            )
            
            # 保存输出
            level_outputs.append(enhanced_states)
            level_stats.append({
                'importance': importance,
                'attention': attention_stats
            })
            
            current_states = enhanced_states
            
        # 自顶向下处理

class DALAConfig:
    """DALA配置类"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        max_sequence_length: int = 1000000,  # 支持百万级序列长度
        compression_rates: List[int] = [2, 4, 8, 16, 32],
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        router_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        use_adaptive_router: bool = True,
        use_sparse_attention: bool = True,
        use_recursive_state: bool = True,
        state_decay_rate: float = 0.9
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.compression_rates = compression_rates
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.router_dropout = router_dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_adaptive_router = use_adaptive_router
        self.use_sparse_attention = use_sparse_attention
        self.use_recursive_state = use_recursive_state
        self.state_decay_rate = state_decay_rate

class DALALayer(nn.Module):
    """DALA层"""
    def __init__(self, config: DALAConfig):
        super().__init__()
        self.attention = InfiniteAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 注意力层
        attention_output, new_state = self.attention(x, state, mask)
        
        # 前馈层
        ff_output = self.feed_forward(attention_output)
        output = self.layer_norm(ff_output + attention_output)
        
        return output, new_state

class DALAModel(nn.Module):
    """完整的DALA模型"""
    def __init__(self, config: DALAConfig):
        super().__init__()
        self.config = config
        
        # 位置编码
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.hidden_size)
        )
        
        # DALA层
        self.layers = nn.ModuleList([
            DALALayer(config)
            for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 获取序列长度
        seq_length = input_ids.size(1)
        
        # 添加位置编码
        hidden_states = input_ids + self.pos_encoder[:, :seq_length]
        
        # 初始化状态
        state = None
        
        # 通过所有层
        for layer in self.layers:
            hidden_states, state = layer(hidden_states, state, attention_mask)
            
        # 最终层归一化
        output = self.final_layer_norm(hidden_states)
        
        return output

def create_dala_model(
    hidden_size: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    max_sequence_length: int = 1000000,
    **kwargs
) -> DALAModel:
    """创建DALA模型的工厂函数"""
    config = DALAConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_sequence_length=max_sequence_length,
        **kwargs
    )
    return DALAModel(config)

class HybridConfig:
    """MSRA和DALA的混合配置"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_levels: int = 3,
        chunk_size: int = 256,
        compression_factor: int = 4,
        dropout: float = 0.1,
        num_layers: int = 6,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        # DALA specific
        max_sequence_length: int = 1000000,
        use_adaptive_router: bool = True,
        use_sparse_attention: bool = True,
        use_recursive_state: bool = True,
        state_decay_rate: float = 0.9,
        # Hybrid specific
        auto_switch_threshold: int = 10000,  # 序列长度超过此值时切换到DALA
        enable_hybrid_mode: bool = True
    ):
        # MSRA配置
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.chunk_size = chunk_size
        self.compression_factor = compression_factor
        self.dropout = dropout
        self.num_layers = num_layers
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        
        # DALA配置
        self.max_sequence_length = max_sequence_length
        self.use_adaptive_router = use_adaptive_router
        self.use_sparse_attention = use_sparse_attention
        self.use_recursive_state = use_recursive_state
        self.state_decay_rate = state_decay_rate
        
        # 混合模式配置
        self.auto_switch_threshold = auto_switch_threshold
        self.enable_hybrid_mode = enable_hybrid_mode

class nullcontext:
    """空上下文管理器"""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MSRAAttention(BaseAttention):
    """MSRA注意力机制实现"""
    
    def __init__(self, config: MSRAConfig):
        super().__init__(config)
        
        # MSRA特定配置
        self.num_scales = config.num_scales
        self.scale_factors = nn.Parameter(
            torch.ones(config.num_scales)
        )
        
        # 递归投影
        self.recursive_proj = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
            for _ in range(config.num_scales)
        ])
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for i in range(self.num_scales):
            nn.init.normal_(
                self.recursive_proj[i].weight,
                mean=0.0,
                std=self.config.initializer_range
            )
            nn.init.zeros_(self.recursive_proj[i].bias)
    
    @ensure_compatibility
    def compute_recursive_attention(
        self,
        hidden_states: torch.Tensor,
        scale_idx: int
    ) -> torch.Tensor:
        """计算递归注意力"""
        try:
            # 应用递归投影
            proj_states = self.recursive_proj[scale_idx](hidden_states)
            
            # 计算Q、K、V
            query_layer = self.transpose_for_scores(self.query(proj_states))
            key_layer = self.transpose_for_scores(self.key(proj_states))
            value_layer = self.transpose_for_scores(self.value(proj_states))
            
            # 计算注意力分数
            attention_scores = self.compute_attention_scores(query_layer, key_layer)
            
            # 应用缩放因子
            attention_scores = attention_scores * self.scale_factors[scale_idx]
            
            # 应用注意力
            context_layer = self.apply_attention(attention_scores, value_layer)
            
            return context_layer
            
        except Exception as e:
            raise ComputationError(
                f"Failed to compute recursive attention at scale {scale_idx}",
                context={
                    "scale_idx": scale_idx,
                    "input_shape": hidden_states.size()
                }
            )
    
    @ensure_compatibility
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """MSRA前向传播"""
        try:
            with PerformanceContext(self) if self.monitor else nullcontext():
                batch_size, seq_len, _ = hidden_states.size()
                
                # 多尺度递归注意力
                context_layers = []
                for i in range(self.num_scales):
                    context_layer = self.compute_recursive_attention(
                        hidden_states, i
                    )
                    context_layers.append(context_layer)
                
                # 合并多尺度结果
                context_layer = torch.stack(context_layers).mean(dim=0)
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
                    stats.update({
                        "num_scales": self.num_scales,
                        "scale_factors": self.scale_factors.tolist()
                    })
                
                return output, stats
                
        except Exception as e:
            self.error_handler.handle_error(
                ComputationError(
                    "MSRA forward pass failed",
                    context={
                        "input_shape": hidden_states.size(),
                        "mask_shape": attention_mask.size() if attention_mask is not None else None,
                        "num_scales": self.num_scales
                    }
                )
            )
            raise