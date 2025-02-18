"""
层次化压缩器
实现了多级压缩策略，平衡压缩率和信息保留
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from .quantization import AdaptiveQuantizer
from .sparse import SparseEncoder

@dataclass
class CompressionLevel:
    """压缩级别配置"""
    name: str
    target_ratio: float
    quality_threshold: float
    method: str  # 'quantize' or 'sparse'

class HierarchicalCompressor(nn.Module):
    """层次化压缩器"""
    
    def __init__(
        self,
        dim: int = 512,
        num_levels: int = 3,
        base_compression_ratio: float = 2.0,
        quality_threshold: float = 0.9
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.base_compression_ratio = base_compression_ratio
        self.quality_threshold = quality_threshold
        
        # 创建压缩级别配置
        self.levels = self._create_compression_levels()
        
        # 质量评估网络
        self.quality_estimator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # 创建各级别的压缩器
        self.compressors = nn.ModuleDict()
        for level in self.levels:
            if level.method == 'quantize':
                compressor = AdaptiveQuantizer(
                    dim=dim,
                    min_bits=4,
                    max_bits=16
                )
            else:  # sparse
                compressor = SparseEncoder(
                    dim=dim,
                    codebook_size=1024,
                    num_codebooks=8
                )
            self.compressors[level.name] = compressor
            
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * num_levels, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def _create_compression_levels(self) -> List[CompressionLevel]:
        """创建压缩级别配置"""
        levels = []
        for i in range(self.num_levels):
            # 压缩率随级别指数增长
            target_ratio = self.base_compression_ratio ** (i + 1)
            # 质量阈值随级别降低
            quality_threshold = self.quality_threshold * (0.9 ** i)
            # 交替使用量化和稀疏编码
            method = 'quantize' if i % 2 == 0 else 'sparse'
            
            level = CompressionLevel(
                name=f'level_{i}',
                target_ratio=target_ratio,
                quality_threshold=quality_threshold,
                method=method
            )
            levels.append(level)
            
        return levels
        
    def _estimate_quality(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor
    ) -> torch.Tensor:
        """评估压缩质量"""
        combined = torch.cat([original, compressed], dim=-1)
        return self.quality_estimator(combined)
        
    def _compress_level(
        self,
        x: torch.Tensor,
        level: CompressionLevel
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """在指定级别进行压缩"""
        compressor = self.compressors[level.name]
        compressed, info = compressor.compress(x)
        
        # 评估质量
        quality = self._estimate_quality(x, compressed)
        info['quality'] = quality
        
        return compressed, info
        
    def compress(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        层次化压缩
        
        Args:
            x: 输入张量
            
        Returns:
            compressed: 压缩后的张量
            info: 压缩相关信息
        """
        results = []
        level_info = {}
        
        # 在每个级别进行压缩
        for level in self.levels:
            compressed, info = self._compress_level(x, level)
            results.append(compressed)
            level_info[level.name] = info
            
            # 如果达到质量要求，可以提前停止
            if info['quality'].mean() >= level.quality_threshold:
                break
                
        # 融合不同级别的结果
        if len(results) > 1:
            # 填充到相同数量的级别
            while len(results) < self.num_levels:
                results.append(torch.zeros_like(x))
            
            # 特征融合
            stacked = torch.cat(results, dim=-1)
            final = self.feature_fusion(stacked)
        else:
            final = results[0]
            
        # 计算总体压缩率
        total_ratio = 1.0
        for level_name, info in level_info.items():
            if 'compression_ratio' in info:
                total_ratio *= info['compression_ratio']
                
        # 收集压缩信息
        info = {
            'level_info': level_info,
            'num_levels_used': len(results),
            'total_compression_ratio': total_ratio,
            'final_quality': self._estimate_quality(x, final)
        }
        
        return final, info
        
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
        # 对每个使用的级别进行解压缩
        results = []
        for level_name, level_info in info['level_info'].items():
            compressor = self.compressors[level_name]
            decompressed = compressor.decompress(x, level_info)
            results.append(decompressed)
            
        # 如果只使用了一个级别
        if len(results) == 1:
            return results[0]
            
        # 填充到相同数量的级别
        while len(results) < self.num_levels:
            results.append(torch.zeros_like(x))
            
        # 融合解压缩结果
        stacked = torch.cat(results, dim=-1)
        return self.feature_fusion(stacked)
        
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        stats = {
            'num_levels': self.num_levels,
            'base_compression_ratio': self.base_compression_ratio,
            'quality_threshold': self.quality_threshold,
            'levels': []
        }
        
        for level in self.levels:
            level_stats = {
                'name': level.name,
                'method': level.method,
                'target_ratio': level.target_ratio,
                'quality_threshold': level.quality_threshold,
                'compressor_stats': self.compressors[level.name].get_compression_stats()
            }
            stats['levels'].append(level_stats)
            
        return stats 