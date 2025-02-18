"""
分布式检查点管理
实现了高效的分布式模型检查点保存和加载机制
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict, Optional, Any, Union, List
import os
import json
import time
from pathlib import Path

class DistributedCheckpoint:
    """分布式检查点管理器"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        amp_scaler: Optional[Any] = None,
        save_dir: str = "checkpoints",
        keep_last_k: int = 5,
        save_freq: int = 1000
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp_scaler = amp_scaler
        self.save_dir = Path(save_dir)
        self.keep_last_k = keep_last_k
        self.save_freq = save_freq
        
        self.step_counter = 0
        self.last_save_time = 0
        
        # 创建保存目录
        if dist.get_rank() == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_checkpoint_path(self, step: int) -> Path:
        """获取检查点文件路径"""
        return self.save_dir / f"checkpoint_{step}.pt"
        
    def _get_metadata_path(self) -> Path:
        """获取元数据文件路径"""
        return self.save_dir / "checkpoint_metadata.json"
        
    def _save_metadata(self, step: int) -> None:
        """保存检查点元数据"""
        if dist.get_rank() == 0:
            metadata_path = self._get_metadata_path()
            metadata = {
                'last_step': step,
                'last_save_time': time.time(),
                'world_size': dist.get_world_size(),
                'model_config': {
                    'name': self.model.__class__.__name__,
                    'params': sum(p.numel() for p in self.model.parameters())
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的检查点文件"""
        if dist.get_rank() == 0:
            checkpoints = sorted(
                self.save_dir.glob("checkpoint_*.pt"),
                key=lambda x: int(x.stem.split('_')[1])
            )
            
            while len(checkpoints) > self.keep_last_k:
                old_ckpt = checkpoints.pop(0)
                try:
                    old_ckpt.unlink()
                except:
                    pass
                    
    def save(
        self,
        step: Optional[int] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存检查点
        
        Args:
            step: 当前步数，如果为None则使用内部计数器
            extra_state: 额外需要保存的状态
        """
        if step is None:
            step = self.step_counter
            
        # 只在主进程保存
        if dist.get_rank() == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'world_size': dist.get_world_size()
            }
            
            if self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
            if self.amp_scaler is not None:
                checkpoint['amp_scaler_state_dict'] = self.amp_scaler.state_dict()
                
            if extra_state is not None:
                checkpoint['extra_state'] = extra_state
                
            # 保存检查点
            torch.save(checkpoint, self._get_checkpoint_path(step))
            
            # 更新元数据
            self._save_metadata(step)
            
            # 清理旧检查点
            self._cleanup_old_checkpoints()
            
        self.last_save_time = time.time()
        self.step_counter = step + 1
        
        # 同步所有进程
        dist.barrier()
        
    def load(
        self,
        step: Optional[int] = None,
        strict: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        加载检查点
        
        Args:
            step: 要加载的步数，如果为None则加载最新的检查点
            strict: 是否严格加载模型参数
            
        Returns:
            extra_state: 额外保存的状态
        """
        # 确定要加载的检查点路径
        if step is None:
            checkpoints = sorted(
                self.save_dir.glob("checkpoint_*.pt"),
                key=lambda x: int(x.stem.split('_')[1])
            )
            if not checkpoints:
                return None
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = self._get_checkpoint_path(step)
            if not checkpoint_path.exists():
                return None
                
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 验证世界大小
        if checkpoint['world_size'] != dist.get_world_size():
            raise ValueError(
                f"检查点的世界大小 ({checkpoint['world_size']}) "
                f"与当前世界大小 ({dist.get_world_size()}) 不匹配"
            )
            
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 加载优化器状态
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 加载调度器状态
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # 加载AMP缩放器状态
        if self.amp_scaler is not None and 'amp_scaler_state_dict' in checkpoint:
            self.amp_scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
            
        self.step_counter = checkpoint['step'] + 1
        
        # 同步所有进程
        dist.barrier()
        
        return checkpoint.get('extra_state')
        
    def should_save(self, step: Optional[int] = None) -> bool:
        """检查是否应该保存检查点"""
        if step is None:
            step = self.step_counter
            
        # 检查是否达到保存频率
        if step % self.save_freq != 0:
            return False
            
        # 检查距离上次保存是否有足够时间间隔
        if time.time() - self.last_save_time < 60:  # 至少间隔1分钟
            return False
            
        return True
        
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点文件路径"""
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pt"),
            key=lambda x: int(x.stem.split('_')[1])
        )
        return checkpoints[-1] if checkpoints else None
        
    def get_checkpoint_steps(self) -> List[int]:
        """获取所有检查点的步数列表"""
        return [
            int(x.stem.split('_')[1])
            for x in self.save_dir.glob("checkpoint_*.pt")
        ] 