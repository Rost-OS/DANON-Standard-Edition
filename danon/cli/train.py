"""
DANON训练命令行工具
提供灵活的模型训练接口
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from danon.core.attention import MSRAModel, DALAModel
from danon.core.distributed import DistributedMSRA
from danon.core.profiler import PerformanceProfiler
from danon.utils.config import load_config
from danon.utils.logging import setup_logging
from danon.utils.optimization import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DANON模型训练工具")
    
    # 基础配置
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_type", type=str, default="msra",
                       choices=["msra", "dala", "distributed"],
                       help="模型类型")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值")
    
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="隐藏层维度")
    parser.add_argument("--num_levels", type=int, default=3,
                       help="层次结构深度")
    parser.add_argument("--chunk_size", type=int, default=256,
                       help="块大小")
    
    # 分布式训练
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="本地进程序号")
    parser.add_argument("--world_size", type=int, default=1,
                       help="总进程数")
    parser.add_argument("--distributed", action="store_true",
                       help="是否使用分布式训练")
    
    # 优化选项
    parser.add_argument("--fp16", action="store_true",
                       help="是否使用混合精度训练")
    parser.add_argument("--optimizer", type=str, default="adam",
                       choices=["adam", "adamw", "sgd", "adafactor"],
                       help="优化器类型")
    parser.add_argument("--scheduler", type=str, default="linear",
                       choices=["linear", "cosine", "constant"],
                       help="学习率调度器类型")
    
    # 日志和检查点
    parser.add_argument("--log_steps", type=int, default=100,
                       help="日志记录间隔步数")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="模型保存间隔步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="评估间隔步数")
    
    # 性能分析
    parser.add_argument("--profile", action="store_true",
                       help="是否启用性能分析")
    parser.add_argument("--profile_steps", type=int, default=100,
                       help="性能分析间隔步数")
    
    return parser.parse_args()

def setup_distributed(args):
    """设置分布式训练环境"""
    if args.distributed:
        if args.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl")
            args.world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    return device

def create_model(args, device):
    """创建模型"""
    if args.model_type == "msra":
        model = MSRAModel(
            hidden_size=args.hidden_size,
            num_levels=args.num_levels,
            chunk_size=args.chunk_size
        )
    elif args.model_type == "dala":
        model = DALAModel(
            hidden_size=args.hidden_size,
            max_sequence_length=args.chunk_size * 32
        )
    elif args.model_type == "distributed":
        model = DistributedMSRA(
            world_size=args.world_size,
            backend="nccl"
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model.to(device)
    
    if args.distributed and args.local_rank != -1:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        
    return model

def train(args, model, train_loader, optimizer, scheduler, device, writer):
    """训练循环"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # 设置性能分析器
    if args.profile:
        profiler = PerformanceProfiler(
            model=model,
            log_dir=args.output_dir / "profiler"
        )
    
    for step, (input_ids, labels) in enumerate(train_loader):
        # 性能分析
        if args.profile and step % args.profile_steps == 0:
            with profiler:
                outputs = model(input_ids.to(device))
        else:
            outputs = model(input_ids.to(device))
            
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.to(device).view(-1)
        )
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        # 混合精度训练
        if args.fp16:
            with torch.cuda.amp.autocast():
                loss.backward()
        else:
            loss.backward()
            
        total_loss += loss.item()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # 记录日志
        if step % args.log_steps == 0:
            cur_loss = total_loss / args.log_steps
            elapsed = time.time() - start_time
            
            logger.info(
                f"Step: {step} | "
                f"Loss: {cur_loss:.4f} | "
                f"Speed: {args.log_steps / elapsed:.2f} steps/s | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            if writer is not None:
                writer.add_scalar("train/loss", cur_loss, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                
            total_loss = 0
            start_time = time.time()
            
        # 保存检查点
        if step % args.save_steps == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                args.output_dir / f"checkpoint-{step}"
            )
            
        # 性能分析报告
        if args.profile and step % args.profile_steps == 0:
            report = profiler.get_performance_report()
            logger.info(f"Performance Report at step {step}:")
            logger.info(report)
            
            if writer is not None:
                for metric, value in report['metrics'].items():
                    writer.add_scalar(f"profiler/{metric}", value['mean'], step)

def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path):
    """保存检查点"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    torch.save({
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, output_dir / "model.pt")
    
def main():
    """主函数"""
    args = parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    setup_logging(output_dir / "train.log")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = setup_distributed(args)
    
    # 创建模型
    model = create_model(args, device)
    
    # 创建优化器和调度器
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(output_dir / "tensorboard")
    
    try:
        # 训练循环
        train(args, model, train_loader, optimizer, scheduler, device, writer)
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练出错: {str(e)}")
        raise
    finally:
        # 保存最终模型
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.epochs * len(train_loader),
            output_dir / "final_model"
        )
        
        # 关闭写入器
        writer.close()
        
if __name__ == "__main__":
    main() 