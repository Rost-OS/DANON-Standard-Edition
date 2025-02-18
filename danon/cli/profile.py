"""
DANON性能分析命令行工具
提供详细的性能分析和优化建议
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from danon.core.attention import MSRAModel, DALAModel
from danon.core.distributed import DistributedMSRA
from danon.core.profiler import PerformanceProfiler
from danon.utils.visualization import (
    plot_memory_timeline,
    plot_gpu_utilization,
    plot_throughput_scaling,
    plot_attention_patterns
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DANON性能分析工具")
    
    # 基础配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--model_type", type=str, default="msra",
                       choices=["msra", "dala", "distributed"],
                       help="模型类型")
    
    # 分析选项
    parser.add_argument("--analysis_type", type=str, required=True,
                       choices=[
                           "memory",
                           "compute",
                           "distributed",
                           "attention",
                           "compression",
                           "all"
                       ],
                       help="分析类型")
    
    # 性能测试参数
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                       default=[1, 4, 16, 32, 64],
                       help="批处理大小列表")
    parser.add_argument("--sequence_lengths", type=int, nargs="+",
                       default=[128, 256, 512, 1024, 2048],
                       help="序列长度列表")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="每个配置的运行次数")
    parser.add_argument("--warmup_runs", type=int, default=10,
                       help="预热运行次数")
    
    # 分布式测试
    parser.add_argument("--num_nodes", type=int, default=1,
                       help="节点数量")
    parser.add_argument("--gpus_per_node", type=int, default=1,
                       help="每个节点的GPU数量")
    
    # 输出选项
    parser.add_argument("--save_traces", action="store_true",
                       help="是否保存性能追踪文件")
    parser.add_argument("--plot_results", action="store_true",
                       help="是否绘制结果图表")
    parser.add_argument("--detailed_report", action="store_true",
                       help="是否生成详细报告")
    
    return parser.parse_args()

def profile_memory(
    model: nn.Module,
    args,
    profiler: PerformanceProfiler
) -> Dict[str, Any]:
    """内存使用分析"""
    results = {}
    device = next(model.parameters()).device
    
    for batch_size in args.batch_sizes:
        for seq_len in args.sequence_lengths:
            # 生成输入数据
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            # 清理缓存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 运行模型
            with profiler:
                model(input_tensor)
                
            # 收集内存统计
            memory_stats = profiler.memory_profiling(detailed=True)
            
            results[f"b{batch_size}_s{seq_len}"] = {
                'peak_memory': memory_stats['peak_memory_mb'],
                'model_parameters': memory_stats['model_parameters'],
                'layer_memory': memory_stats['layer_memory'],
                'optimizer_states': memory_stats.get('optimizer_states', 0)
            }
            
    return results

def profile_compute(
    model: nn.Module,
    args,
    profiler: PerformanceProfiler
) -> Dict[str, Any]:
    """计算性能分析"""
    results = {}
    device = next(model.parameters()).device
    
    for batch_size in args.batch_sizes:
        for seq_len in args.sequence_lengths:
            times = []
            gpu_utils = []
            
            # 生成输入数据
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            # 预热
            for _ in range(args.warmup_runs):
                with torch.no_grad():
                    model(input_tensor)
                    
            # 测试运行
            for _ in range(args.num_runs):
                torch.cuda.synchronize()
                
                with profiler:
                    start_time = time.time()
                    with torch.no_grad():
                        model(input_tensor)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    gpu_utils.append(
                        torch.cuda.utilization()
                    )
                    
            # 计算统计信息
            results[f"b{batch_size}_s{seq_len}"] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'throughput': batch_size * seq_len / np.mean(times),
                'avg_gpu_util': np.mean(gpu_utils) if gpu_utils else None
            }
            
    return results

def profile_distributed(
    model: nn.Module,
    args,
    profiler: PerformanceProfiler
) -> Dict[str, Any]:
    """分布式性能分析"""
    if not isinstance(model, DistributedMSRA):
        raise ValueError("Model must be DistributedMSRA for distributed profiling")
        
    results = {}
    world_size = args.num_nodes * args.gpus_per_node
    
    for batch_size in args.batch_sizes:
        for seq_len in args.sequence_lengths:
            # 收集每个GPU的性能数据
            gpu_stats = []
            for gpu_id in range(world_size):
                with torch.cuda.device(gpu_id):
                    input_tensor = torch.randn(
                        batch_size // world_size,
                        seq_len,
                        model.hidden_size
                    )
                    
                    # 运行模型
                    with profiler:
                        model.forward_single_gpu(input_tensor)
                        
                    # 收集GPU统计信息
                    gpu_stats.append({
                        'memory': torch.cuda.memory_allocated() / 1024**2,
                        'utilization': torch.cuda.utilization()
                    })
                    
            # 运行分布式前向传播
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size
            )
            
            with profiler:
                start_time = time.time()
                model(input_tensor)
                end_time = time.time()
                
            # 计算加速比和效率
            single_gpu_time = gpu_stats[0]['time']
            distributed_time = end_time - start_time
            speedup = single_gpu_time / distributed_time
            efficiency = speedup / world_size
            
            results[f"b{batch_size}_s{seq_len}"] = {
                'speedup': speedup,
                'efficiency': efficiency,
                'gpu_stats': gpu_stats,
                'communication_overhead': profiler.get_communication_stats()
            }
            
    return results

def profile_attention(
    model: nn.Module,
    args,
    profiler: PerformanceProfiler
) -> Dict[str, Any]:
    """注意力机制分析"""
    results = {}
    device = next(model.parameters()).device
    
    for batch_size in args.batch_sizes:
        for seq_len in args.sequence_lengths:
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            with profiler:
                # 获取注意力权重
                attention_weights = model.compute_attention_weights(input_tensor)
                
                # 分析注意力模式
                results[f"b{batch_size}_s{seq_len}"] = {
                    'attention_patterns': {
                        level: weights.cpu().numpy()
                        for level, weights in attention_weights.items()
                    },
                    'sparsity': {
                        level: (weights == 0).float().mean().item()
                        for level, weights in attention_weights.items()
                    },
                    'entropy': {
                        level: (-weights * weights.log()).sum(dim=-1).mean().item()
                        for level, weights in attention_weights.items()
                    }
                }
                
    return results

def profile_compression(
    model: nn.Module,
    args,
    profiler: PerformanceProfiler
) -> Dict[str, Any]:
    """压缩机制分析"""
    results = {}
    device = next(model.parameters()).device
    
    for batch_size in args.batch_sizes:
        for seq_len in args.sequence_lengths:
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            with profiler:
                # 获取压缩统计信息
                compression_stats = model.get_compression_stats(input_tensor)
                
                results[f"b{batch_size}_s{seq_len}"] = {
                    'compression_ratio': compression_stats['compression_ratio'],
                    'information_retention': compression_stats['information_retention'],
                    'memory_savings': compression_stats.get('memory_savings', None)
                }
                
    return results

def generate_report(
    results: Dict[str, Any],
    args,
    output_dir: Path
) -> None:
    """生成分析报告"""
    report = {
        'summary': {
            'model_type': args.model_type,
            'analysis_type': args.analysis_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': results
    }
    
    # 添加系统信息
    report['system_info'] = {
        'cpu_info': {
            'cores': psutil.cpu_count(),
            'frequency': psutil.cpu_freq()._asdict()
        },
        'memory_info': psutil.virtual_memory()._asdict(),
        'gpu_info': [
            {
                'name': torch.cuda.get_device_name(i),
                'memory': torch.cuda.get_device_properties(i).total_memory
            }
            for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else None
    }
    
    # 生成优化建议
    report['optimization_suggestions'] = generate_optimization_suggestions(results)
    
    # 保存报告
    report_file = output_dir / "profiling_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    # 生成可视化
    if args.plot_results:
        generate_visualizations(results, output_dir)
        
def generate_optimization_suggestions(
    results: Dict[str, Any]
) -> List[str]:
    """生成优化建议"""
    suggestions = []
    
    # 分析内存使用
    if 'memory' in results:
        peak_memory = max(
            r['peak_memory']
            for r in results['memory'].values()
        )
        if peak_memory > 8 * 1024:  # 8GB
            suggestions.append(
                "内存使用过高，建议：\n"
                "1. 使用梯度检查点\n"
                "2. 减小批处理大小\n"
                "3. 使用混合精度训练"
            )
            
    # 分析计算性能
    if 'compute' in results:
        gpu_utils = [
            r['avg_gpu_util']
            for r in results['compute'].values()
            if r['avg_gpu_util'] is not None
        ]
        if gpu_utils and np.mean(gpu_utils) < 70:
            suggestions.append(
                "GPU利用率低，建议：\n"
                "1. 增加批处理大小\n"
                "2. 使用数据预取\n"
                "3. 优化数据加载pipeline"
            )
            
    # 分析分布式性能
    if 'distributed' in results:
        efficiencies = [
            r['efficiency']
            for r in results['distributed'].values()
        ]
        if np.mean(efficiencies) < 0.7:
            suggestions.append(
                "分布式扩展效率低，建议：\n"
                "1. 增加计算通信比\n"
                "2. 使用梯度压缩\n"
                "3. 优化通信策略"
            )
            
    # 分析注意力机制
    if 'attention' in results:
        sparsities = [
            np.mean(list(r['sparsity'].values()))
            for r in results['attention'].values()
        ]
        if np.mean(sparsities) < 0.5:
            suggestions.append(
                "注意力权重稠密，建议：\n"
                "1. 使用稀疏注意力机制\n"
                "2. 增加注意力dropout\n"
                "3. 考虑使用局部注意力"
            )
            
    return suggestions

def generate_visualizations(
    results: Dict[str, Any],
    output_dir: Path
) -> None:
    """生成可视化图表"""
    if 'memory' in results:
        plot_memory_timeline(
            results['memory'],
            output_dir / "memory_timeline.png"
        )
        
    if 'compute' in results:
        plot_gpu_utilization(
            results['compute'],
            output_dir / "gpu_utilization.png"
        )
        
    if 'distributed' in results:
        plot_throughput_scaling(
            results['distributed'],
            output_dir / "throughput_scaling.png"
        )
        
    if 'attention' in results:
        plot_attention_patterns(
            results['attention'],
            output_dir / "attention_patterns.png"
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
            logging.FileHandler(output_dir / "profiling.log"),
            logging.StreamHandler()
        ]
    )
    
    # 加载模型
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    
    # 创建性能分析器
    profiler = PerformanceProfiler(
        model=model,
        log_dir=output_dir / "profiler",
        enabled=True
    )
    
    # 运行性能分析
    results = {}
    
    if args.analysis_type in ['memory', 'all']:
        logger.info("Analyzing memory usage...")
        results['memory'] = profile_memory(model, args, profiler)
        
    if args.analysis_type in ['compute', 'all']:
        logger.info("Analyzing compute performance...")
        results['compute'] = profile_compute(model, args, profiler)
        
    if args.analysis_type in ['distributed', 'all'] and isinstance(model, DistributedMSRA):
        logger.info("Analyzing distributed performance...")
        results['distributed'] = profile_distributed(model, args, profiler)
        
    if args.analysis_type in ['attention', 'all']:
        logger.info("Analyzing attention mechanism...")
        results['attention'] = profile_attention(model, args, profiler)
        
    if args.analysis_type in ['compression', 'all']:
        logger.info("Analyzing compression mechanism...")
        results['compression'] = profile_compression(model, args, profiler)
        
    # 生成报告
    logger.info("Generating report...")
    generate_report(results, args, output_dir)
    
    logger.info(f"Profiling results saved to {output_dir}")
    
if __name__ == "__main__":
    main() 