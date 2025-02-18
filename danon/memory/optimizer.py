"""
内存优化系统
实现智能的内存管理和优化策略
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
from threading import Lock
import gc
import psutil
import logging
import time
from collections import OrderedDict, defaultdict
import weakref

@dataclass
class MemoryOptimizerConfig:
    """内存优化器配置"""
    # 内存限制
    max_gpu_memory: int = 8 * 1024 * 1024 * 1024  # 8GB
    max_cpu_memory: int = 32 * 1024 * 1024 * 1024  # 32GB
    
    # 优化策略
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_dynamic_batching: bool = True
    enable_memory_defrag: bool = True
    
    # 缓存配置
    cache_size: int = 1000
    enable_cache: bool = True
    
    # 量化配置
    quantization_bits: int = 8
    enable_quantization: bool = True
    
    # 垃圾回收
    gc_threshold: float = 0.8  # 当内存使用率超过此值时触发GC
    gc_interval: int = 1000  # GC间隔(毫秒)
    
    # 监控配置
    monitoring_interval: int = 100  # 监控间隔(毫秒)
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
class MemoryMonitor:
    """内存监控器"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        self.history = defaultdict(list)
        self._lock = Lock()
        self.last_gc_time = time.time()
        
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        stats = {}
        
        # GPU内存
        if torch.cuda.is_available():
            stats.update({
                'gpu_allocated': torch.cuda.memory_allocated(),
                'gpu_reserved': torch.cuda.memory_reserved(),
                'gpu_max_allocated': torch.cuda.max_memory_allocated(),
                'gpu_max_reserved': torch.cuda.max_memory_reserved()
            })
            
        # CPU内存
        process = psutil.Process()
        stats['cpu_memory'] = process.memory_info().rss
        
        return stats
        
    def should_trigger_gc(self) -> bool:
        """判断是否需要触发GC"""
        # 检查时间间隔
        current_time = time.time()
        if current_time - self.last_gc_time < self.config.gc_interval / 1000:
            return False
            
        # 检查内存使用率
        stats = self.get_memory_stats()
        if torch.cuda.is_available():
            gpu_usage = stats['gpu_allocated'] / self.config.max_gpu_memory
            if gpu_usage > self.config.gc_threshold:
                return True
                
        cpu_usage = stats['cpu_memory'] / self.config.max_cpu_memory
        return cpu_usage > self.config.gc_threshold
        
    def update_history(self) -> None:
        """更新历史记录"""
        with self._lock:
            stats = self.get_memory_stats()
            for key, value in stats.items():
                self.history[key].append(value)
                
            # 限制历史记录长度
            max_history = 1000
            for key in self.history:
                if len(self.history[key]) > max_history:
                    self.history[key] = self.history[key][-max_history:]
                    
    def get_history(self) -> Dict[str, List[float]]:
        """获取历史记录"""
        with self._lock:
            return dict(self.history)
            
class TensorCache:
    """张量缓存"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        self.cache = OrderedDict()
        self.size = 0
        self._lock = Lock()
        
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """计算张量大小(字节)"""
        return tensor.element_size() * tensor.nelement()
        
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """存入张量"""
        if not self.config.enable_cache:
            return
            
        with self._lock:
            # 如果已存在，先移除
            if key in self.cache:
                self.size -= self._get_tensor_size(self.cache[key])
                del self.cache[key]
                
            # 存入新张量
            self.cache[key] = tensor
            self.size += self._get_tensor_size(tensor)
            
            # 如果超出大小限制，移除最旧的
            while len(self.cache) > self.config.cache_size:
                _, old_tensor = self.cache.popitem(last=False)
                self.size -= self._get_tensor_size(old_tensor)
                
    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取张量"""
        if not self.config.enable_cache:
            return None
            
        with self._lock:
            if key in self.cache:
                tensor = self.cache.pop(key)
                self.cache[key] = tensor  # 移到最新
                return tensor
        return None
        
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.size = 0
            
class TensorQuantizer:
    """张量量化器"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """量化张量"""
        if not self.config.enable_quantization:
            return tensor, {}
            
        # 计算量化参数
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2 ** self.config.quantization_bits - 1)
        zero_point = min_val
        
        # 量化
        quantized = ((tensor - zero_point) / scale).round().to(torch.uint8)
        
        return quantized, {
            'scale': scale,
            'zero_point': zero_point,
            'original_dtype': tensor.dtype
        }
        
    def dequantize(
        self,
        quantized: torch.Tensor,
        params: Dict[str, Any]
    ) -> torch.Tensor:
        """反量化张量"""
        if not self.config.enable_quantization:
            return quantized
            
        # 反量化
        dequantized = quantized.float() * params['scale'] + params['zero_point']
        return dequantized.to(params['original_dtype'])
        
class MemoryDefragmenter:
    """内存碎片整理器"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        
    def defragment(self) -> None:
        """执行碎片整理"""
        if not self.config.enable_memory_defrag:
            return
            
        # 强制执行垃圾回收
        gc.collect()
        
        if torch.cuda.is_available():
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            # 尝试整理GPU内存
            current_device = torch.cuda.current_device()
            torch.cuda.memory._dump_snapshot()
            torch.cuda.synchronize()
            
class DynamicBatchSizer:
    """动态批次大小调整器"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        self.history = []
        self._lock = Lock()
        
    def compute_optimal_batch_size(
        self,
        current_batch_size: int,
        memory_usage: float
    ) -> int:
        """计算最优批次大小"""
        if not self.config.enable_dynamic_batching:
            return current_batch_size
            
        # 根据内存使用率调整批次大小
        memory_threshold = 0.9  # 90%内存使用率阈值
        
        if memory_usage > memory_threshold:
            # 减小批次大小
            new_size = max(1, int(current_batch_size * 0.8))
        elif memory_usage < memory_threshold * 0.7:
            # 增加批次大小
            new_size = int(current_batch_size * 1.2)
        else:
            new_size = current_batch_size
            
        # 更新历史
        with self._lock:
            self.history.append({
                'time': time.time(),
                'batch_size': new_size,
                'memory_usage': memory_usage
            })
            
            # 限制历史记录长度
            if len(self.history) > 100:
                self.history.pop(0)
                
        return new_size
        
class MemoryOptimizer:
    """内存优化器"""
    def __init__(self, config: MemoryOptimizerConfig):
        self.config = config
        
        # 初始化组件
        self.monitor = MemoryMonitor(config)
        self.cache = TensorCache(config)
        self.quantizer = TensorQuantizer(config)
        self.defragmenter = MemoryDefragmenter(config)
        self.batch_sizer = DynamicBatchSizer(config)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 性能统计
        self.stats = defaultdict(float)
        self._stats_lock = Lock()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("MemoryOptimizer")
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        return logger
        
    def optimize_memory(self) -> None:
        """执行内存优化"""
        try:
            # 检查是否需要GC
            if self.monitor.should_trigger_gc():
                self.logger.info("Triggering garbage collection")
                gc.collect()
                self.monitor.last_gc_time = time.time()
                
            # 执行碎片整理
            if self.config.enable_memory_defrag:
                self.defragmenter.defragment()
                
            # 更新监控历史
            self.monitor.update_history()
            
        except Exception as e:
            self.logger.error(f"Error in memory optimization: {str(e)}")
            
    def process_tensor(
        self,
        tensor: torch.Tensor,
        key: Optional[str] = None
    ) -> torch.Tensor:
        """处理张量(缓存和量化)"""
        try:
            # 检查缓存
            if key and self.config.enable_cache:
                cached = self.cache.get(key)
                if cached is not None:
                    return cached
                    
            # 量化
            if self.config.enable_quantization:
                quantized, params = self.quantizer.quantize(tensor)
                # 存入缓存
                if key:
                    self.cache.put(key, quantized)
                return self.quantizer.dequantize(quantized, params)
                
            # 如果不需要量化，直接存入缓存
            if key:
                self.cache.put(key, tensor)
                
            return tensor
            
        except Exception as e:
            self.logger.error(f"Error processing tensor: {str(e)}")
            return tensor
            
    def optimize_batch_size(
        self,
        current_batch_size: int
    ) -> int:
        """优化批次大小"""
        try:
            # 获取当前内存使用率
            stats = self.monitor.get_memory_stats()
            if torch.cuda.is_available():
                memory_usage = stats['gpu_allocated'] / self.config.max_gpu_memory
            else:
                memory_usage = stats['cpu_memory'] / self.config.max_cpu_memory
                
            # 计算新的批次大小
            return self.batch_sizer.compute_optimal_batch_size(
                current_batch_size,
                memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing batch size: {str(e)}")
            return current_batch_size
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        with self._stats_lock:
            stats = dict(self.stats)
            
        # 添加监控信息
        stats.update({
            'memory': self.monitor.get_memory_stats(),
            'cache': {
                'size': self.cache.size,
                'num_items': len(self.cache.cache)
            },
            'batch_size_history': self.batch_sizer.history[-10:]  # 最近10条记录
        })
        
        return stats
        
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        
    def register_model(self, model: nn.Module) -> None:
        """注册模型以进行优化"""
        if self.config.enable_gradient_checkpointing:
            # 启用梯度检查点
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
        if self.config.enable_mixed_precision:
            # 启用混合精度训练
            if hasattr(model, 'half'):
                model.half()
                
    def __del__(self):
        """清理资源"""
        self.clear_cache() 