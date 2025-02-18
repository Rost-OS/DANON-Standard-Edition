"""
稀疏编码压缩器
实现了基于MSRA的稀疏表示学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import math
from ...core.attention import MSRAConfig, MSRAModel
from dataclasses import dataclass

@dataclass
class SparseConfig:
    """稀疏编码配置"""
    hidden_size: int = 768
    sparsity_factor: float = 0.1
    msra_levels: int = 2,
    msra_chunk_size: int = 32
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.msra_levels = msra_levels
        self.msra_chunk_size = msra_chunk_size

class SparseEncoder(nn.Module):
    """基于MSRA的稀疏编码压缩器"""
    
    def __init__(self, config: SparseConfig):
        super().__init__()
        self.config = config
        
        # 使用MSRA进行特征处理
        msra_config = MSRAConfig(
            hidden_size=config.hidden_size,
            num_levels=self.config.msra_levels,
            chunk_size=self.config.msra_chunk_size,
            compression_factor=4
        )
        self.feature_processor = MSRAModel(msra_config)
        
        # 码本
        self.codebooks = nn.Parameter(
            torch.randn(
                self.config.msra_levels,
                self.config.msra_chunk_size,
                self.config.hidden_size
            )
        )
        
        # 码本注意力
        self.codebook_attention = nn.MultiheadAttention(
            self.config.hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.LayerNorm(self.config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        )
        
        # 初始化参数
        self._init_parameters()
        
        # 训练状态跟踪
        self.register_buffer('usage_count', torch.zeros(
            self.config.msra_levels,
            self.config.msra_chunk_size
        ))
        
    def _init_parameters(self) -> None:
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _update_codebook_usage(self, encodings: torch.Tensor) -> None:
        """更新码本使用统计"""
        with torch.no_grad():
            self.usage_count += encodings.sum(dim=0)
            
    def _prune_unused_codes(self, min_usage: int = 100) -> None:
        """清理未使用的码本项"""
        with torch.no_grad():
            mask = self.usage_count < min_usage
            if mask.any():
                # 重新初始化未使用的码本项
                self.codebooks.data[mask] = torch.randn_like(
                    self.codebooks.data[mask]
                )
                self.usage_count[mask] = 0
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = x.size(0)
        
        # 编码
        encoded = self.decoder(x)
        
        # 使用MSRA处理特征
        processed_features, importance_scores = self.feature_processor(encoded)
        
        # 使用注意力机制处理码本
        codebook_output, _ = self.codebook_attention(
            processed_features.transpose(0, 1),
            self.codebooks.view(-1, self.config.hidden_size).unsqueeze(0).expand(
                batch_size, -1, -1
            ).transpose(0, 1),
            self.codebooks.view(-1, self.config.hidden_size).unsqueeze(0).expand(
                batch_size, -1, -1
            ).transpose(0, 1)
        )
        codebook_output = codebook_output.transpose(0, 1)
        
        # 计算与码本的相似度
        similarities = torch.einsum(
            'bnd,mcd->bnmc',
            codebook_output,
            self.codebooks
        )
        
        # 稀疏化选择
        logits = similarities / self.config.sparsity_factor
        probs = F.softmax(logits, dim=-1)
        
        # 计算稀疏性损失
        sparsity_loss = F.mse_loss(
            probs.mean(),
            torch.tensor(self.config.sparsity_factor, device=probs.device)
        )
        
        # 选择最相似的码本项
        indices = torch.argmax(probs, dim=-1)
        encodings = torch.zeros_like(probs).scatter_(-1, indices.unsqueeze(-1), 1)
        
        # 更新码本使用统计
        self._update_codebook_usage(encodings.sum(dim=(0, 1)))
        
        # 重构
        quantized = torch.einsum('bnmc,mcd->bnd', encodings, self.codebooks)
        reconstructed = self.decoder(quantized)
        
        # 计算损失
        commitment_loss = F.mse_loss(processed_features.detach(), quantized)
        diversity_loss = -torch.mean(torch.log(probs + 1e-10))
        
        info = {
            'commitment_loss': commitment_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'importance_scores': importance_scores,
            'encodings': encodings.detach(),
            'codebook_usage': self.usage_count.clone()
        }
        
        return reconstructed, info
        
    def compress(
        self,
        x: torch.Tensor,
        adaptive_quantization: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩输入数据"""
        reconstructed, info = self.forward(x)
        
        if adaptive_quantization:
            # 根据重要性分数自适应量化
            importance = torch.cat(
                [score.mean(dim=1) for score in info['importance_scores']]
            )
            quantization_bits = torch.clamp(
                (importance * 8).round(),
                min=4,
                max=8
            ).long()
            
            # 量化重构结果
            scale = torch.max(torch.abs(reconstructed), dim=-1, keepdim=True)[0]
            normalized = reconstructed / scale
            quantized = []
            
            for i in range(reconstructed.size(0)):
                bits = quantization_bits[i]
                levels = 2 ** bits - 1
                quantized.append(
                    torch.round(normalized[i] * levels) / levels * scale[i]
                )
            quantized = torch.stack(quantized)
            
            info['quantization_bits'] = quantization_bits
            reconstructed = quantized
            
        # 计算压缩率
        active_codes = info['encodings'].sum().item()
        compression_ratio = (
            x.numel() * 32 /
            (active_codes * math.log2(self.config.msra_chunk_size))
        )
        
        info['active_codes'] = active_codes
        info['compression_ratio'] = compression_ratio
        
        return reconstructed, info
        
    def get_state_dict(self) -> Dict[str, Any]:
        """获取模型状态"""
        return {
            'model_state': self.state_dict(),
            'config': self.config.__dict__,
            'usage_count': self.usage_count
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载模型状态"""
        self.config = SparseConfig(**state_dict['config'])
        super().load_state_dict(state_dict['model_state'])
        self.usage_count.copy_(state_dict['usage_count'])
        
    def optimize_codebooks(self) -> None:
        """优化码本结构"""
        with torch.no_grad():
            # 清理未使用的码本项
            self._prune_unused_codes()
            
            # 规范化码本向量
            self.codebooks.data = F.normalize(self.codebooks.data, dim=-1)
            
    def reset_usage_stats(self) -> None:
        """重置使用统计"""
        self.usage_count.zero_()
        
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
        return self.decoder(x)
        
    def update_codebooks(self, decay: float = 0.99) -> None:
        """更新码本（指数移动平均）"""
        with torch.no_grad():
            self.codebooks.data *= decay
            
    def get_codebook_usage(self) -> torch.Tensor:
        """获取码本使用统计信息"""
        return self.codebooks.norm(dim=-1)
        
class AdaptiveSparseEncoder(SparseEncoder):
    """自适应稀疏编码压缩器"""
    
    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
        temperature: float = 0.1,
        commitment_cost: float = 0.25,
        diversity_cost: float = 0.1,
        sparsity_target: float = 0.1,
        adaptation_rate: float = 0.01
    ):
        super().__init__(
            dim,
            codebook_size,
            num_codebooks,
            temperature,
            commitment_cost,
            diversity_cost,
            sparsity_target
        )
        self.adaptation_rate = adaptation_rate
        
        # 自适应参数预测器
        self.param_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 3)  # [temperature, sparsity_target, commitment_cost]
        )
        
    def _adapt_parameters(self, x: torch.Tensor) -> None:
        """根据输入自适应调整参数"""
        params = self.param_predictor(x.mean(dim=0))
        
        # 更新温度参数
        self.temperature = (
            (1 - self.adaptation_rate) * self.temperature +
            self.adaptation_rate * F.softplus(params[0])
        )
        
        # 更新稀疏目标
        self.sparsity_target = (
            (1 - self.adaptation_rate) * self.sparsity_target +
            self.adaptation_rate * torch.sigmoid(params[1])
        )
        
        # 更新承诺成本
        self.commitment_cost = (
            (1 - self.adaptation_rate) * self.commitment_cost +
            self.adaptation_rate * F.softplus(params[2])
        )
        
    def compress(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """自适应压缩"""
        # 调整参数
        self._adapt_parameters(x)
        
        # 执行压缩
        return super().compress(x) 