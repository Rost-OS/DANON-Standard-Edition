"""
统一的通信优化模块
提供了高效的分布式通信功能
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Tuple, Optional
import math
import time
from dataclasses import dataclass
from enum import Enum

class CompressionMethod(Enum):
    """压缩方法"""
    NONE = "none"
    TOPK = "topk"
    QUANTIZATION = "quantization"
    ADAPTIVE = "adaptive"

@dataclass
class CompressionConfig:
    """压缩配置"""
    method: CompressionMethod = CompressionMethod.ADAPTIVE
    compression_ratio: float = 0.01
    min_compression_ratio: float = 0.001
    max_compression_ratio: float = 0.1
    quantization_bits: int = 8
    adapt_compression: bool = True
    importance_threshold: float = 0.1

class CommunicationOptimizer:
    """通信优化器"""
    
    def __init__(
        self,
        compression_config: Optional[CompressionConfig] = None,
        enable_overlap: bool = True,
        bucket_size_mb: int = 25,
        gradient_accumulation: bool = False,
        accumulation_steps: int = 1
    ):
        self.compression_config = compression_config or CompressionConfig()
        self.enable_overlap = enable_overlap
        self.bucket_size_mb = bucket_size_mb
        self.gradient_accumulation = gradient_accumulation
        self.accumulation_steps = accumulation_steps
        
        # 性能统计
        self.stats = {
            'total_bytes_sent': 0,
            'total_bytes_saved': 0,
            'total_time': 0.0,
            'num_operations': 0,
            'compression_ratios': []
        }
        
        # 通信缓冲区
        self.buffers: Dict[int, torch.Tensor] = {}
        self.meta_buffers: Dict[int, Dict[str, Any]] = {}
        
        # 梯度累积
        self.current_step = 0
        self.accumulated_grads: Dict[str, torch.Tensor] = {}
        
    def optimize_communication(
        self,
        tensor: torch.Tensor,
        name: Optional[str] = None,
        importance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """优化张量通信"""
        start_time = time.time()
        original_size = tensor.nelement() * tensor.element_size()
        
        try:
            # 决定是否压缩
            if self._should_compress(tensor, importance):
                compressed, meta = self._compress_tensor(tensor, importance)
                
                # 同步压缩后的张量
                if self.enable_overlap:
                    compressed = self._async_allreduce(compressed)
                else:
                    dist.all_reduce(compressed)
                    
                # 解压缩
                result = self._decompress_tensor(compressed, meta)
                
                # 更新统计信息
                compressed_size = compressed.nelement() * compressed.element_size()
                self.stats['total_bytes_saved'] += original_size - compressed_size
                self.stats['compression_ratios'].append(compressed_size / original_size)
            else:
                # 直接同步
                if self.enable_overlap:
                    result = self._async_allreduce(tensor)
                else:
                    result = tensor.clone()
                    dist.all_reduce(result)
                    
            # 处理梯度累积
            if self.gradient_accumulation and name is not None:
                result = self._handle_gradient_accumulation(name, result)
                
        except Exception as e:
            # 发生错误时回退到直接同步
            result = tensor.clone()
            dist.all_reduce(result)
            
        # 更新统计信息
        end_time = time.time()
        self.stats['total_time'] += end_time - start_time
        self.stats['total_bytes_sent'] += original_size
        self.stats['num_operations'] += 1
        
        return result
        
    def _should_compress(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> bool:
        """决定是否应该压缩张量"""
        if self.compression_config.method == CompressionMethod.NONE:
            return False
            
        # 检查张量大小
        if tensor.numel() < 1000:  # 小张量不压缩
            return False
            
        # 如果提供了重要性分数，使用它来决定
        if importance is not None and self.compression_config.adapt_compression:
            return importance.mean().item() < self.compression_config.importance_threshold
            
        return True
        
    def _compress_tensor(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """压缩张量"""
        if self.compression_config.method == CompressionMethod.TOPK:
            return self._topk_compression(tensor, importance)
        elif self.compression_config.method == CompressionMethod.QUANTIZATION:
            return self._quantization_compression(tensor)
        elif self.compression_config.method == CompressionMethod.ADAPTIVE:
            return self._adaptive_compression(tensor, importance)
        else:
            raise ValueError(f"Unknown compression method: {self.compression_config.method}")
            
    def _topk_compression(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Top-k 压缩"""
        # 计算要保留的元素数量
        k = max(1, int(tensor.numel() * self.compression_config.compression_ratio))
        
        # 如果有重要性分数，用它来选择元素
        if importance is not None:
            _, indices = torch.topk(importance.abs().flatten(), k)
        else:
            _, indices = torch.topk(tensor.abs().flatten(), k)
            
        # 创建压缩张量
        values = tensor.flatten()[indices]
        compressed = torch.zeros(k * 2, dtype=torch.float32, device=tensor.device)
        compressed[0::2] = values
        compressed[1::2] = indices.float()
        
        return compressed, {
            'original_shape': tensor.shape,
            'original_dtype': tensor.dtype,
            'k': k
        }
        
    def _quantization_compression(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """量化压缩"""
        bits = self.compression_config.quantization_bits
        
        # 计算缩放因子
        max_val = tensor.abs().max()
        scale = max_val / (2 ** (bits - 1) - 1)
        
        # 量化
        quantized = torch.round(tensor / scale)
        quantized = torch.clamp(quantized, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        
        return quantized, {
            'scale': scale,
            'bits': bits,
            'original_dtype': tensor.dtype
        }
        
    def _adaptive_compression(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """自适应压缩"""
        if importance is not None:
            # 根据重要性动态调整压缩率
            imp_score = importance.mean().item()
            compression_ratio = (
                self.compression_config.min_compression_ratio +
                (self.compression_config.max_compression_ratio - self.compression_config.min_compression_ratio) *
                (1 - imp_score)
            )
        else:
            compression_ratio = self.compression_config.compression_ratio
            
        # 使用 Top-k 压缩
        k = max(1, int(tensor.numel() * compression_ratio))
        values, indices = torch.topk(tensor.abs().flatten(), k)
        
        # 量化值
        bits = self.compression_config.quantization_bits
        scale = values.max() / (2 ** (bits - 1) - 1)
        quantized_values = torch.round(values / scale)
        quantized_values = torch.clamp(quantized_values, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        
        # 打包压缩数据
        compressed = torch.zeros(k * 2, dtype=torch.float32, device=tensor.device)
        compressed[0::2] = quantized_values
        compressed[1::2] = indices.float()
        
        return compressed, {
            'original_shape': tensor.shape,
            'original_dtype': tensor.dtype,
            'scale': scale,
            'k': k,
            'bits': bits
        }
        
    def _decompress_tensor(
        self,
        compressed: torch.Tensor,
        meta: Dict[str, Any]
    ) -> torch.Tensor:
        """解压缩张量"""
        if 'k' in meta:  # Top-k 或自适应压缩
            decompressed = torch.zeros(
                meta['original_shape'].numel(),
                dtype=meta['original_dtype'],
                device=compressed.device
            )
            
            values = compressed[0::2]
            indices = compressed[1::2].long()
            
            if 'scale' in meta:  # 自适应压缩
                values = values * meta['scale']
                
            decompressed[indices] = values
            decompressed = decompressed.view(meta['original_shape'])
            
        elif 'scale' in meta:  # 量化压缩
            decompressed = compressed * meta['scale']
            decompressed = decompressed.to(meta['original_dtype'])
            
        return decompressed
        
    def _async_allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """异步 AllReduce"""
        # 获取或创建通信缓冲区
        buffer = self.buffers.get(tensor.size(0))
        if buffer is None:
            buffer = torch.zeros_like(tensor)
            self.buffers[tensor.size(0)] = buffer
            
        # 启动异步通信
        buffer.copy_(tensor)
        handle = dist.all_reduce(buffer, async_op=True)
        handle.wait()
        
        return buffer
        
    def _handle_gradient_accumulation(
        self,
        name: str,
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """处理梯度累积"""
        if name not in self.accumulated_grads:
            self.accumulated_grads[name] = gradient.clone()
        else:
            self.accumulated_grads[name] += gradient
            
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            # 返回累积的梯度并清除
            result = self.accumulated_grads[name] / self.accumulation_steps
            del self.accumulated_grads[name]
            return result
        else:
            # 继续累积
            return torch.zeros_like(gradient)
            
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats.copy()
        
        if self.stats['num_operations'] > 0:
            stats['average_compression_ratio'] = (
                sum(self.stats['compression_ratios']) /
                len(self.stats['compression_ratios'])
            )
            stats['average_bandwidth'] = (
                self.stats['total_bytes_sent'] /
                max(self.stats['total_time'], 1e-6)
            )
            stats['compression_efficiency'] = (
                self.stats['total_bytes_saved'] /
                max(self.stats['total_bytes_sent'], 1)
            )
            
        return stats
        
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_bytes_sent': 0,
            'total_bytes_saved': 0,
            'total_time': 0.0,
            'num_operations': 0,
            'compression_ratios': []
        } 