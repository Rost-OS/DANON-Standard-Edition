"""
对比学习训练器
用于训练语义检索模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from .semantic import SemanticRetriever

class ContrastiveTrainer:
    """对比学习训练器"""
    
    def __init__(
        self,
        model: SemanticRetriever,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        device: str = 'cuda',
        log_interval: int = 10,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.num_epochs = num_epochs
        
        # 如果未指定优化器，使用默认的AdamW
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 如果未指定调度器，使用默认的余弦退火
        self.scheduler = scheduler or torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # 记录最佳验证结果
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _compute_loss(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算对比学习损失"""
        # 计算相似度矩阵
        sim, query_emb, key_emb = self.model(query, key)
        
        # 如果提供了负样本
        if negative_keys is not None:
            neg_emb = self.model.encode_key(negative_keys)
            neg_sim = torch.matmul(query_emb, neg_emb.t()) / self.model.temperature
            sim = torch.cat([sim, neg_sim], dim=1)
            
        # 创建标签（对角线为正样本）
        labels = torch.arange(sim.size(0), device=sim.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(sim, labels)
        
        # 计算准确率
        pred = sim.argmax(dim=1)
        acc = (pred == labels).float().mean()
        
        # 计算其他指标
        pos_sim = torch.diagonal(sim[:, :query.size(0)]).mean()
        neg_sim = (sim - torch.eye(sim.size(0), device=sim.device) * 1e9).max(dim=1)[0].mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': acc.item(),
            'pos_similarity': pos_sim.item(),
            'neg_similarity': neg_sim.item()
        }
        
        return loss, metrics
        
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_metrics = {}
        num_batches = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (query, key, *rest) in enumerate(pbar):
                # 移动数据到设备
                query = query.to(self.device)
                key = key.to(self.device)
                negative_keys = rest[0].to(self.device) if rest else None
                
                # 前向传播和损失计算
                loss, metrics = self._compute_loss(query, key, negative_keys)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新进度条
                if batch_idx % self.log_interval == 0:
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                    
                # 累积指标
                for name, value in metrics.items():
                    total_metrics[name] = total_metrics.get(name, 0) + value
                num_batches += 1
                
                # 记录到wandb
                if self.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()})
                    
        # 计算平均指标
        avg_metrics = {
            name: value / num_batches
            for name, value in total_metrics.items()
        }
        
        return avg_metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with tqdm(self.val_loader, desc='Validation') as pbar:
            for query, key, *rest in pbar:
                # 移动数据到设备
                query = query.to(self.device)
                key = key.to(self.device)
                negative_keys = rest[0].to(self.device) if rest else None
                
                # 计算损失和指标
                _, metrics = self._compute_loss(query, key, negative_keys)
                
                # 累积指标
                for name, value in metrics.items():
                    total_metrics[name] = total_metrics.get(name, 0) + value
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                
        # 计算平均指标
        avg_metrics = {
            name: value / num_batches
            for name, value in total_metrics.items()
        }
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in avg_metrics.items()})
            
        return avg_metrics
        
    def train(self) -> Dict[str, Any]:
        """训练模型"""
        if self.use_wandb:
            wandb.init(project="semantic_retriever")
            
        best_epoch = 0
        train_history = []
        val_history = []
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            train_history.append(train_metrics)
            
            # 验证
            val_metrics = self.validate()
            val_history.append(val_metrics)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict()
                best_epoch = epoch
                
            # 打印指标
            print("\nTraining metrics:")
            for name, value in train_metrics.items():
                print(f"{name}: {value:.4f}")
                
            if val_metrics:
                print("\nValidation metrics:")
                for name, value in val_metrics.items():
                    print(f"{name}: {value:.4f}")
                    
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        if self.use_wandb:
            wandb.finish()
            
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss
        }
        
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }, path)
        
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state'] 