"""
混合精度训练
实现了自动混合精度训练支持
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, Any
from contextlib import contextmanager
import time

class MixedPrecisionTraining:
    """混合精度训练管理器"""
    
    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24
    ):
        self.enabled = enabled
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            min_scale=min_scale,
            max_scale=max_scale
        ) if enabled else None
        
    @contextmanager
    def autocast(self):
        """自动混合精度上下文"""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
            
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        return self.scaler.scale(loss) if self.enabled else loss
        
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Optional[callable] = None
    ) -> Optional[Any]:
        """执行优化器步骤"""
        if self.enabled:
            return self.scaler.step(optimizer, closure)
        else:
            return optimizer.step(closure)
            
    def update(self) -> None:
        """更新梯度缩放器"""
        if self.enabled:
            self.scaler.update()
            
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """取消梯度缩放"""
        if self.enabled:
            self.scaler.unscale_(optimizer)
            
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        return self.scaler.get_scale() if self.enabled else 1.0
        
    def is_enabled(self) -> bool:
        """检查是否启用混合精度"""
        return self.enabled
        
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'enabled': self.enabled,
            'scaler': self.scaler.state_dict() if self.enabled else None
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载状态字典"""
        self.enabled = state_dict['enabled']
        if self.enabled and state_dict['scaler'] is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
            
class DynamicLossScaler:
    """动态损失缩放器"""
    
    def __init__(
        self,
        init_scale: float = 65536.0,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
        consecutive_hysteresis: int = 2
    ):
        self.cur_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.consecutive_hysteresis = consecutive_hysteresis
        
        self.consecutive_good_steps = 0
        self.num_bad_steps = 0
        self.total_steps = 0
        
    def update_scale(self, has_inf_nan: bool) -> None:
        """更新缩放因子"""
        if has_inf_nan:
            self.cur_scale = max(
                self.cur_scale / self.scale_factor,
                self.min_scale
            )
            self.consecutive_good_steps = 0
            self.num_bad_steps += 1
        else:
            self.consecutive_good_steps += 1
            if self.consecutive_good_steps >= self.consecutive_hysteresis:
                self.cur_scale = min(
                    self.cur_scale * self.scale_factor,
                    self.max_scale
                )
                self.consecutive_good_steps = 0
                
        self.total_steps += 1
        
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        return loss * self.cur_scale
        
    def unscale(self, gradients: torch.Tensor) -> torch.Tensor:
        """取消梯度缩放"""
        return gradients / self.cur_scale
        
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        return self.cur_scale
        
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'current_scale': self.cur_scale,
            'num_bad_steps': self.num_bad_steps,
            'total_steps': self.total_steps,
            'bad_step_ratio': self.num_bad_steps / max(1, self.total_steps)
        }

class AdaptivePrecisionManager:
    """自适应精度管理器"""
    
    def __init__(
        self,
        initial_bits: int = 16,
        min_bits: int = 8,
        max_bits: int = 32,
        adjustment_interval: int = 100
    ):
        self.current_bits = initial_bits
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.adjustment_interval = adjustment_interval
        
        self.step_counter = 0
        self.loss_history = []
        self.grad_norms = []
        self.overflow_counts = []
        
        self.momentum = 0.9
        
    def get_dtype(self) -> torch.dtype:
        """获取当前数据类型"""
        if self.current_bits == 32:
            return torch.float32
        elif self.current_bits == 16:
            return torch.float16
        else:
            return torch.bfloat16
            
    def update_stats(
        self,
        loss: float,
        grad_norm: float,
        overflow: bool
    ) -> None:
        """更新统计信息"""
        self.loss_history.append(loss)
        self.grad_norms.append(grad_norm)
        self.overflow_counts.append(1 if overflow else 0)
        self.step_counter += 1
        
        # 调整精度
        if self.step_counter % self.adjustment_interval == 0:
            self._adjust_precision()
            
    def _adjust_precision(self) -> None:
        """调整精度"""
        if len(self.loss_history) < self.adjustment_interval:
            return
            
        # 计算统计指标
        recent_losses = self.loss_history[-self.adjustment_interval:]
        recent_norms = self.grad_norms[-self.adjustment_interval:]
        recent_overflows = self.overflow_counts[-self.adjustment_interval:]
        
        loss_std = torch.tensor(recent_losses).std().item()
        norm_mean = sum(recent_norms) / len(recent_norms)
        overflow_rate = sum(recent_overflows) / len(recent_overflows)
        
        # 计算调整因子
        stability_factor = 1.0
        if loss_std > 1.0 or overflow_rate > 0.1:
            # 不稳定，增加精度
            stability_factor = 1.1
        elif loss_std < 0.1 and overflow_rate < 0.01:
            # 稳定，可以降低精度
            stability_factor = 0.9
            
        efficiency_factor = 1.0
        if norm_mean > 10.0:
            # 梯度较大，增加精度
            efficiency_factor = 1.1
        elif norm_mean < 0.1:
            # 梯度较小，可以降低精度
            efficiency_factor = 0.9
            
        # 综合调整因子
        adjustment = stability_factor * efficiency_factor
        
        # 应用动量
        adjustment = self.momentum + (1 - self.momentum) * adjustment
        
        # 更新精度
        new_bits = int(self.current_bits * adjustment)
        self.current_bits = max(self.min_bits, min(self.max_bits, new_bits))
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_bits': self.current_bits,
            'avg_loss': sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0,
            'avg_grad_norm': sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else 0,
            'overflow_rate': sum(self.overflow_counts) / len(self.overflow_counts) if self.overflow_counts else 0
        }

class EnhancedMixedPrecisionTraining:
    """增强型混合精度训练"""
    
    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0**24,
        initial_bits: int = 16,
        min_bits: int = 8,
        max_bits: int = 32
    ):
        self.enabled = enabled
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            min_scale=min_scale,
            max_scale=max_scale
        ) if enabled else None
        
        # 自适应精度管理器
        self.precision_manager = AdaptivePrecisionManager(
            initial_bits=initial_bits,
            min_bits=min_bits,
            max_bits=max_bits
        )
        
        # 性能监控
        self.performance_metrics = {
            'throughput': [],
            'memory_usage': [],
            'convergence_rate': []
        }
        
    @contextmanager
    def autocast(self):
        """自动混合精度上下文"""
        if self.enabled:
            dtype = self.precision_manager.get_dtype()
            with autocast(dtype=dtype):
                yield
        else:
            yield
            
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        return self.scaler.scale(loss) if self.enabled else loss
        
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        grad_norm: float,
        closure: Optional[callable] = None
    ) -> Optional[Any]:
        """执行优化器步骤"""
        step_start = time.time()
        
        if self.enabled:
            result = self.scaler.step(optimizer, closure)
            self.scaler.update()
            
            # 检查是否发生溢出
            overflow = self.scaler.get_scale() != self.scaler._scale
            
            # 更新精度管理器
            self.precision_manager.update_stats(
                loss=loss.item(),
                grad_norm=grad_norm,
                overflow=overflow
            )
            
            # 更新性能指标
            step_time = time.time() - step_start
            self._update_performance_metrics(step_time, loss.item())
            
            return result
        else:
            return optimizer.step(closure)
            
    def _update_performance_metrics(
        self,
        step_time: float,
        loss: float
    ) -> None:
        """更新性能指标"""
        # 计算吞吐量
        throughput = 1.0 / step_time
        self.performance_metrics['throughput'].append(throughput)
        
        # 估算内存使用
        memory_usage = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        self.performance_metrics['memory_usage'].append(memory_usage)
        
        # 估算收敛率
        if len(self.performance_metrics['convergence_rate']) > 0:
            prev_loss = self.performance_metrics['convergence_rate'][-1]
            conv_rate = (prev_loss - loss) / prev_loss
        else:
            conv_rate = 0.0
        self.performance_metrics['convergence_rate'].append(conv_rate)
        
        # 保持固定窗口大小
        window_size = 100
        for metric in self.performance_metrics.values():
            if len(metric) > window_size:
                metric.pop(0)
                
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'precision': self.precision_manager.get_stats(),
            'throughput': {
                'current': self.performance_metrics['throughput'][-1] if self.performance_metrics['throughput'] else 0,
                'average': sum(self.performance_metrics['throughput']) / len(self.performance_metrics['throughput']) if self.performance_metrics['throughput'] else 0
            },
            'memory': {
                'current_gb': self.performance_metrics['memory_usage'][-1] if self.performance_metrics['memory_usage'] else 0,
                'average_gb': sum(self.performance_metrics['memory_usage']) / len(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0
            },
            'convergence': {
                'current_rate': self.performance_metrics['convergence_rate'][-1] if self.performance_metrics['convergence_rate'] else 0,
                'average_rate': sum(self.performance_metrics['convergence_rate']) / len(self.performance_metrics['convergence_rate']) if self.performance_metrics['convergence_rate'] else 0
            }
        } 