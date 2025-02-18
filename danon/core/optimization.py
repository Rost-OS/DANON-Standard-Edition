"""
动态优化器和学习率调度器
实现了自适应优化策略和动态学习率调整
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

class DynamicAdamW(Optimizer):
    """动态AdamW优化器，具有自适应权重衰减和动态步长调整"""
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
        adaptive_momentum: bool = True,
        dynamic_weight_decay: bool = True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"无效的epsilon值: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的beta1参数: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的beta2参数: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            adaptive_momentum=adaptive_momentum,
            dynamic_weight_decay=dynamic_weight_decay,
            buffer={}  # 用于存储动态调整的状态
        )
        super().__init__(params, defaults)
        
    def _init_group(self, group):
        """初始化参数组"""
        for p in group['params']:
            if p.grad is None:
                continue
                
            state = self.state[p]
            
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                if group['adaptive_momentum']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    
    def _update_momentum(self, group, p, grad):
        """更新动量"""
        state = self.state[p]
        beta1, beta2 = group['betas']
        
        # 计算梯度方差
        grad_var = torch.var(grad)
        
        # 动态调整beta1
        if group['adaptive_momentum']:
            momentum_scale = torch.clamp(grad_var / (grad_var + 1e-8), 0.8, 1.0)
            beta1 = beta1 * momentum_scale
            
        # 更新动量
        state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        return beta1
        
    def _compute_weight_decay(self, group, p):
        """计算动态权重衰减"""
        if group['dynamic_weight_decay']:
            # 基于参数范数动态调整权重衰减
            param_norm = torch.norm(p.data)
            decay_scale = torch.clamp(param_norm / (param_norm + 1e-8), 0.1, 2.0)
            return group['weight_decay'] * decay_scale
        return group['weight_decay']
        
    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            self._init_group(group)
            
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # 更新步数
                state['step'] += 1
                
                # 动态动量更新
                beta1 = self._update_momentum(group, p, grad)
                
                # 计算偏差修正
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else:
                    bias_correction1 = bias_correction2 = 1
                    
                # 计算自适应学习率
                denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # 应用权重衰减
                weight_decay = self._compute_weight_decay(group, p)
                if weight_decay != 0:
                    p.data.mul_(1 - step_size * weight_decay)
                    
                # 更新参数
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
                
        return loss

class DynamicLRScheduler(_LRScheduler):
    """动态学习率调度器，支持多种调整策略"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        min_lr: float = 1e-7,
        decay_style: str = 'cosine',
        adaptive_warmup: bool = True,
        dynamic_decay: bool = True,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.decay_style = decay_style
        self.adaptive_warmup = adaptive_warmup
        self.dynamic_decay = dynamic_decay
        
        # 性能追踪
        self.loss_history = []
        self.lr_history = []
        self.grad_norm_history = []
        
        super().__init__(optimizer, last_epoch)
        
    def _get_warmup_factor(self, step: int) -> float:
        """计算预热因子"""
        if not self.adaptive_warmup:
            return min(1.0, step / self.warmup_steps)
            
        # 自适应预热：基于梯度范数调整
        if len(self.grad_norm_history) > 0:
            recent_grads = self.grad_norm_history[-10:]
            grad_var = np.var(recent_grads) if len(recent_grads) > 1 else 0
            warmup_scale = 1.0 / (1.0 + grad_var)
            return min(1.0, step / (self.warmup_steps * warmup_scale))
        return min(1.0, step / self.warmup_steps)
        
    def _get_decay_factor(self, step: int) -> float:
        """计算衰减因子"""
        if step <= self.warmup_steps:
            return 1.0
            
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        
        if self.decay_style == 'cosine':
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        elif self.decay_style == 'linear':
            factor = 1 - progress
        else:  # exponential
            factor = math.exp(-5 * progress)
            
        if self.dynamic_decay and len(self.loss_history) > 1:
            # 基于损失变化调整衰减速率
            loss_change = (self.loss_history[-1] - self.loss_history[-2]) / self.loss_history[-2]
            if loss_change > 0:  # 损失增加
                factor *= 0.95  # 加快衰减
            elif loss_change < -0.1:  # 损失显著下降
                factor *= 1.05  # 减缓衰减
                
        return max(0.0, factor)
        
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        warmup_factor = self._get_warmup_factor(self.last_epoch)
        decay_factor = self._get_decay_factor(self.last_epoch)
        
        return [
            max(self.min_lr, base_lr * warmup_factor * decay_factor)
            for base_lr in self.base_lrs
        ]
        
    def step(self, metrics=None):
        """更新学习率并记录性能指标"""
        if metrics is not None:
            if isinstance(metrics, dict):
                self.loss_history.append(metrics.get('loss', 0.0))
                self.grad_norm_history.append(metrics.get('grad_norm', 0.0))
            else:
                self.loss_history.append(float(metrics))
                
        super().step()
        
        # 记录学习率历史
        self.lr_history.append([group['lr'] for group in self.optimizer.param_groups])
        
    def get_last_lr(self) -> List[float]:
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self) -> Dict:
        """获取调度器状态"""
        state_dict = {key: value for key, value in self.__dict__.items()
                     if key not in ('optimizer', 'loss_history', 'lr_history', 'grad_norm_history')}
        state_dict['loss_history'] = self.loss_history
        state_dict['lr_history'] = self.lr_history
        state_dict['grad_norm_history'] = self.grad_norm_history
        return state_dict
        
    def load_state_dict(self, state_dict: Dict):
        """加载调度器状态"""
        loss_history = state_dict.pop('loss_history', [])
        lr_history = state_dict.pop('lr_history', [])
        grad_norm_history = state_dict.pop('grad_norm_history', [])
        
        self.__dict__.update(state_dict)
        
        self.loss_history = loss_history
        self.lr_history = lr_history
        self.grad_norm_history = grad_norm_history 