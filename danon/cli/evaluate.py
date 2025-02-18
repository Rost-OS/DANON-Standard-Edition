"""
DANON评估命令行工具
提供全面的模型评估功能
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from danon.core.attention import MSRAModel, DALAModel
from danon.core.distributed import DistributedMSRA
from danon.core.profiler import PerformanceProfiler
from danon.utils.metrics import (
    calculate_accuracy,
    calculate_perplexity,
    calculate_rouge,
    calculate_bleu,
    calculate_memory_usage,
    calculate_throughput
)
from danon.utils.visualization import (
    plot_attention_weights,
    plot_compression_ratio,
    plot_performance_metrics
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DANON模型评估工具")
    
    # 基础配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--model_type", type=str, default="msra",
                       choices=["msra", "dala", "distributed"],
                       help="模型类型")
    
    # 评估参数
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大评估样本数")
    parser.add_argument("--beam_size", type=int, default=4,
                       help="束搜索大小")
    
    # 性能评估
    parser.add_argument("--profile", action="store_true",
                       help="是否进行性能分析")
    parser.add_argument("--benchmark", action="store_true",
                       help="是否进行基准测试")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="基准测试运行次数")
    
    # 可视化选项
    parser.add_argument("--plot_attention", action="store_true",
                       help="是否可视化注意力权重")
    parser.add_argument("--plot_compression", action="store_true",
                       help="是否可视化压缩率")
    parser.add_argument("--plot_metrics", action="store_true",
                       help="是否可视化性能指标")
    
    return parser.parse_args()

def load_model(args) -> nn.Module:
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    if args.model_type == "msra":
        model = MSRAModel(
            hidden_size=768,  # 从检查点中获取
            num_levels=3,
            chunk_size=256
        )
    elif args.model_type == "dala":
        model = DALAModel(
            hidden_size=768,
            max_sequence_length=8192
        )
    elif args.model_type == "distributed":
        model = DistributedMSRA(
            world_size=1,
            backend="nccl"
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    args
) -> Dict[str, float]:
    """评估模型性能"""
    device = next(model.parameters()).device
    metrics = {}
    
    # 初始化指标
    total_loss = 0
    total_accuracy = 0
    total_perplexity = 0
    total_samples = 0
    
    # 收集预测结果和参考文本
    predictions = []
    references = []
    
    # 性能分析器
    if args.profile:
        profiler = PerformanceProfiler(
            model=model,
            log_dir=Path(args.output_dir) / "eval_profile"
        )
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # 性能分析
            if args.profile:
                with profiler:
                    outputs = model(input_ids)
            else:
                outputs = model(input_ids)
                
            # 计算损失
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )
            total_loss += loss.item() * input_ids.size(0)
            
            # 计算准确率
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy * input_ids.size(0)
            
            # 计算困惑度
            perplexity = calculate_perplexity(loss.item())
            total_perplexity += perplexity * input_ids.size(0)
            
            # 收集预测和参考
            pred = outputs.argmax(dim=-1)
            predictions.extend(pred.cpu().numpy())
            references.extend(labels.cpu().numpy())
            
            total_samples += input_ids.size(0)
            
            if args.max_samples and total_samples >= args.max_samples:
                break
                
    # 计算平均指标
    metrics['loss'] = total_loss / total_samples
    metrics['accuracy'] = total_accuracy / total_samples
    metrics['perplexity'] = total_perplexity / total_samples
    
    # 计算ROUGE和BLEU分数
    metrics['rouge'] = calculate_rouge(predictions, references)
    metrics['bleu'] = calculate_bleu(predictions, references)
    
    # 性能指标
    if args.profile:
        performance_report = profiler.get_performance_report()
        metrics.update(performance_report)
        
    return metrics

def benchmark_model(
    model: nn.Module,
    args
) -> Dict[str, float]:
    """模型基准测试"""
    device = next(model.parameters()).device
    results = {}
    
    # 准备测试数据
    sequence_lengths = [128, 256, 512, 1024, 2048]
    batch_sizes = [1, 4, 16, 32]
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            key = f"b{batch_size}_s{seq_len}"
            times = []
            memory_usage = []
            
            # 生成随机输入
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    model(input_tensor)
                    
            # 测试运行
            for _ in range(args.num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    model(input_tensor)
                    
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
                memory_usage.append(
                    calculate_memory_usage()
                )
                
            # 计算统计信息
            times = np.array(times)
            memory_usage = np.array(memory_usage)
            
            results[key] = {
                'avg_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput': calculate_throughput(
                    batch_size, seq_len, np.mean(times)
                ),
                'avg_memory': float(np.mean(memory_usage)),
                'peak_memory': float(np.max(memory_usage))
            }
            
    return results

def visualize_results(model: nn.Module, eval_loader: DataLoader, args):
    """可视化评估结果"""
    output_dir = Path(args.output_dir)
    device = next(model.parameters()).device
    
    # 获取一个批次的数据进行可视化
    input_ids, _ = next(iter(eval_loader))
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        # 注意力权重可视化
        if args.plot_attention:
            attention_weights = model.compute_attention_weights(input_ids)
            plot_attention_weights(
                attention_weights,
                output_dir / "attention_weights.png"
            )
            
        # 压缩率可视化
        if args.plot_compression:
            compression_stats = model.get_compression_stats(input_ids)
            plot_compression_ratio(
                compression_stats,
                output_dir / "compression_ratio.png"
            )
            
        # 性能指标可视化
        if args.plot_metrics and args.profile:
            profiler = PerformanceProfiler(model=model)
            with profiler:
                model(input_ids)
            performance_metrics = profiler.get_performance_report()
            plot_performance_metrics(
                performance_metrics,
                output_dir / "performance_metrics.png"
            )

def main():
    """主函数"""
    args = parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "eval.log"),
            logging.StreamHandler()
        ]
    )
    
    # 加载模型
    logger.info("Loading model...")
    model = load_model(args)
    
    # 加载评估数据
    logger.info("Loading evaluation data...")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 评估模型
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, eval_loader, args)
    
    # 基准测试
    if args.benchmark:
        logger.info("Running benchmarks...")
        benchmark_results = benchmark_model(model, args)
        metrics['benchmark'] = benchmark_results
        
    # 保存结果
    results_file = output_dir / "eval_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Results saved to {results_file}")
    
    # 可视化
    if any([args.plot_attention, args.plot_compression, args.plot_metrics]):
        logger.info("Generating visualizations...")
        visualize_results(model, eval_loader, args)
        
if __name__ == "__main__":
    main() 