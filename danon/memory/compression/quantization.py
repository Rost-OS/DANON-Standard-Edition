"""
自适应量化压缩器
实现了基于数据特征的动态量化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import math

class AdaptiveQuantizer(nn.Module):
    """自适应量化压缩器"""
    
    def __init__(
        self,
        dim: int = 512,
        min_bits: int = 4,
        max_bits: int = 16,
        temperature: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.temperature = temperature
        
        # 位宽预测网络
        self.bit_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, max_bits - min_bits + 1),
            nn.Softmax(dim=-1)
        )
        
        # 缩放因子预测网络
        self.scale_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softplus()
        )
        
        # 重构网络
        self.reconstructor = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def _predict_bits(self, x: torch.Tensor) -> torch.Tensor:
        """预测每个维度应该使用的位数"""
        logits = self.bit_predictor(x)
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        bits = torch.arange(
            self.min_bits,
            self.max_bits + 1,
            device=x.device
        ) * probs
        return bits.sum(dim=-1)
        
    def _predict_scale(self, x: torch.Tensor) -> torch.Tensor:
        """预测量化缩放因子"""
        return self.scale_predictor(x)
        
    def _quantize(
        self,
        x: torch.Tensor,
        bits: torch.Tensor,
        scale: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """量化张量"""
        # 计算量化步长
        step_size = scale / (2 ** bits - 1)
        
        # 量化
        x_scaled = x / step_size
        x_quantized = torch.round(x_scaled)
        x_dequantized = x_quantized * step_size
        
        # 收集量化信息
        info = {
            'bits': bits,
            'scale': scale,
            'step_size': step_size,
        }
        
        return x_dequantized, info
        
    def compress(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        压缩输入数据
        
        Args:
            x: 输入张量
            
        Returns:
            compressed: 压缩后的张量
            info: 压缩相关信息
        """
        # 预测量化参数
        bits = self._predict_bits(x)
        scale = self._predict_scale(x)
        
        # 执行量化
        x_quantized, quant_info = self._quantize(x, bits, scale)
        
        # 计算压缩率
        original_bits = x.element_size() * 8
        compression_ratio = original_bits / bits.mean().item()
        
        # 收集压缩信息
        info = {
            'quantization': quant_info,
            'compression_ratio': compression_ratio,
            'mean_bits': bits.mean().item(),
        }
        
        return x_quantized, info
        
    def decompress(
        self,
        x: torch.Tensor,
        info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        解压缩数据
        
        Args:
            x: 压缩的张量
            info: 压缩信息
            
        Returns:
            decompressed: 解压缩后的张量
        """
        # 使用重构网络恢复细节
        return self.reconstructor(x)
        
    def get_compression_stats(self) -> Dict[str, float]:
        """获取压缩统计信息"""
        return {
            'min_bits': self.min_bits,
            'max_bits': self.max_bits,
            'temperature': self.temperature,
        }
        
class AdaptiveGroupQuantizer(AdaptiveQuantizer):
    """分组自适应量化压缩器"""
    
    def __init__(
        self,
        dim: int = 512,
        group_size: int = 8,
        min_bits: int = 4,
        max_bits: int = 16,
        temperature: float = 1.0
    ):
        super().__init__(dim, min_bits, max_bits, temperature)
        self.group_size = group_size
        
        # 确保维度能被组大小整除
        assert dim % group_size == 0, f"维度 {dim} 必须能被组大小 {group_size} 整除"
        
        # 组特征提取器
        self.group_feature_extractor = nn.Sequential(
            nn.Linear(group_size, group_size * 2),
            nn.ReLU(),
            nn.Linear(group_size * 2, group_size)
        )
        
    def _extract_group_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取每组的特征"""
        # 重塑张量以进行分组处理
        groups = x.view(-1, self.dim // self.group_size, self.group_size)
        return self.group_feature_extractor(groups)
        
    def compress(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """分组压缩"""
        # 提取组特征
        group_features = self._extract_group_features(x)
        
        # 为每组预测量化参数
        bits = self._predict_bits(group_features)
        scale = self._predict_scale(group_features)
        
        # 执行分组量化
        groups = x.view(-1, self.dim // self.group_size, self.group_size)
        quantized_groups = []
        quant_infos = []
        
        for i in range(groups.size(1)):
            quantized, info = self._quantize(
                groups[:, i],
                bits[:, i:i+1],
                scale[:, i:i+1]
            )
            quantized_groups.append(quantized)
            quant_infos.append(info)
            
        # 合并结果
        x_quantized = torch.cat(quantized_groups, dim=1)
        x_quantized = x_quantized.view(-1, self.dim)
        
        # 计算压缩统计信息
        compression_ratio = (x.element_size() * 8) / bits.mean().item()
        
        info = {
            'group_quantization': quant_infos,
            'compression_ratio': compression_ratio,
            'mean_bits_per_group': bits.mean(dim=1),
            'group_size': self.group_size,
        }
        
        return x_quantized, info 