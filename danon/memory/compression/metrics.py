"""
压缩质量评估指标
实现了各种压缩质量评估方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math

class CompressionMetrics:
    """压缩质量评估指标集合"""
    
    @staticmethod
    def mse_loss(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
        """均方误差损失"""
        return F.mse_loss(original, compressed)
        
    @staticmethod
    def psnr(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
        """峰值信噪比"""
        mse = F.mse_loss(original, compressed)
        return 10 * torch.log10(1.0 / mse)
        
    @staticmethod
    def cosine_similarity(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
        """余弦相似度"""
        return F.cosine_similarity(original, compressed, dim=-1).mean()
        
    @staticmethod
    def information_density(compressed: torch.Tensor) -> torch.Tensor:
        """信息密度（基于熵）"""
        # 将值归一化到[0,1]区间
        normalized = (compressed - compressed.min()) / (compressed.max() - compressed.min())
        # 计算直方图
        hist = torch.histc(normalized, bins=256, min=0, max=1)
        # 计算概率分布
        probs = hist / hist.sum()
        # 计算熵
        entropy = -(probs * torch.log2(probs + 1e-10)).sum()
        return entropy
        
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        """压缩率"""
        return original_size / compressed_size
        
    @staticmethod
    def relative_error(original: torch.Tensor, compressed: torch.Tensor) -> torch.Tensor:
        """相对误差"""
        return torch.abs(original - compressed) / (torch.abs(original) + 1e-10)
        
    @staticmethod
    def structural_similarity(
        original: torch.Tensor,
        compressed: torch.Tensor,
        window_size: int = 11
    ) -> torch.Tensor:
        """结构相似性"""
        # 计算均值
        mu1 = F.avg_pool1d(
            original.unsqueeze(1),
            window_size,
            stride=1,
            padding=window_size//2
        ).squeeze(1)
        
        mu2 = F.avg_pool1d(
            compressed.unsqueeze(1),
            window_size,
            stride=1,
            padding=window_size//2
        ).squeeze(1)
        
        # 计算方差
        sigma1_sq = F.avg_pool1d(
            (original.unsqueeze(1) - mu1.unsqueeze(1))**2,
            window_size,
            stride=1,
            padding=window_size//2
        ).squeeze(1)
        
        sigma2_sq = F.avg_pool1d(
            (compressed.unsqueeze(1) - mu2.unsqueeze(1))**2,
            window_size,
            stride=1,
            padding=window_size//2
        ).squeeze(1)
        
        # 计算协方差
        sigma12 = F.avg_pool1d(
            (original.unsqueeze(1) - mu1.unsqueeze(1)) *
            (compressed.unsqueeze(1) - mu2.unsqueeze(1)),
            window_size,
            stride=1,
            padding=window_size//2
        ).squeeze(1)
        
        # SSIM公式的常数
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        # 计算SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
               
        return ssim.mean()
        
    @classmethod
    def evaluate_all(
        cls,
        original: torch.Tensor,
        compressed: torch.Tensor,
        original_size: int,
        compressed_size: int
    ) -> Dict[str, float]:
        """计算所有评估指标"""
        return {
            'mse': cls.mse_loss(original, compressed).item(),
            'psnr': cls.psnr(original, compressed).item(),
            'cosine_similarity': cls.cosine_similarity(original, compressed).item(),
            'information_density': cls.information_density(compressed).item(),
            'compression_ratio': cls.compression_ratio(original_size, compressed_size),
            'relative_error': cls.relative_error(original, compressed).mean().item(),
            'ssim': cls.structural_similarity(original, compressed).item()
        }
        
class CompressionQualityMonitor:
    """压缩质量监控器"""
    
    def __init__(self, threshold_config: Dict[str, float] = None):
        self.metrics = CompressionMetrics()
        self.threshold_config = threshold_config or {
            'mse': 0.01,
            'psnr': 30.0,
            'cosine_similarity': 0.95,
            'information_density': 0.5,
            'compression_ratio': 2.0,
            'relative_error': 0.05,
            'ssim': 0.9
        }
        self.history = []
        
    def check_quality(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor,
        original_size: int,
        compressed_size: int
    ) -> Dict[str, Any]:
        """检查压缩质量"""
        # 计算所有指标
        metrics = CompressionMetrics.evaluate_all(
            original,
            compressed,
            original_size,
            compressed_size
        )
        
        # 检查是否满足阈值要求
        quality_check = {}
        for name, value in metrics.items():
            threshold = self.threshold_config.get(name)
            if threshold is not None:
                if name in ['mse', 'relative_error']:
                    passed = value <= threshold
                else:
                    passed = value >= threshold
                quality_check[f'{name}_passed'] = passed
                
        # 记录历史
        self.history.append({
            'metrics': metrics,
            'quality_check': quality_check
        })
        
        return {
            'metrics': metrics,
            'quality_check': quality_check,
            'overall_passed': all(quality_check.values())
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取历史统计信息"""
        if not self.history:
            return {}
            
        stats = {
            'num_samples': len(self.history),
            'metrics_mean': {},
            'metrics_std': {},
            'pass_rate': {}
        }
        
        # 计算平均值和标准差
        for metric_name in self.history[0]['metrics'].keys():
            values = [h['metrics'][metric_name] for h in self.history]
            stats['metrics_mean'][metric_name] = sum(values) / len(values)
            stats['metrics_std'][metric_name] = torch.tensor(values).std().item()
            
        # 计算通过率
        for check_name in self.history[0]['quality_check'].keys():
            passed = sum(h['quality_check'][check_name] for h in self.history)
            stats['pass_rate'][check_name] = passed / len(self.history)
            
        return stats 