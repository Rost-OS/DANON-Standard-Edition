"""
DANON基准测试命令行工具
提供全面的性能基准测试功能
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
from danon.utils.visualization import (
    plot_benchmark_results,
    plot_scaling_efficiency,
    plot_memory_usage,
    plot_latency_distribution
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DANON基准测试工具")
    
    # 基础配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--model_type", type=str, default="msra",
                       choices=["msra", "dala", "distributed"],
                       help="模型类型")
    
    # 基准测试参数
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16, 32, 64],
                       help="批处理大小列表")
    parser.add_argument("--sequence_lengths", type=int, nargs="+",
                       default=[128, 256, 512, 1024, 2048, 4096],
                       help="序列长度列表")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="每个配置的运行次数")
    parser.add_argument("--warmup_runs", type=int, default=10,
                       help="预热运行次数")
    
    # 测试模式
    parser.add_argument("--test_mode", type=str, default="all",
                       choices=["latency", "throughput", "memory", "all"],
                       help="测试模式")
    
    # 硬件配置
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="GPU数量")
    parser.add_argument("--cpu_threads", type=int, default=1,
                       help="CPU线程数")
    parser.add_argument("--pin_memory", action="store_true",
                       help="是否使用锁页内存")
    
    # 优化选项
    parser.add_argument("--fp16", action="store_true",
                       help="是否使用半精度")
    parser.add_argument("--channels_last", action="store_true",
                       help="是否使用channels_last内存格式")
    parser.add_argument("--cuda_graphs", action="store_true",
                       help="是否使用CUDA图")
    
    # 输出选项
    parser.add_argument("--save_traces", action="store_true",
                       help="是否保存性能追踪")
    parser.add_argument("--plot_results", action="store_true",
                       help="是否绘制结果图表")
    
    return parser.parse_args()

def setup_model(args) -> nn.Module:
    """设置模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    
    # 应用优化
    if args.fp16:
        model = model.half()
        
    if args.channels_last and torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
        
    if args.num_gpus > 1:
        model = nn.DataParallel(model)
        
    if args.model_type == "msra":
        model = MSRAModel(config)
    elif args.model_type == "dala":
        model = DALAModel(config)
    elif args.model_type == "distributed":
        model = DistributedMSRA(config)
        
    return model

def measure_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int,
    use_cuda_graphs: bool = False
) -> Tuple[float, float, float]:
    """测量延迟"""
    times = []
    
    if use_cuda_graphs and torch.cuda.is_available():
        # 创建CUDA图
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = model(input_tensor)
            
        # 使用CUDA图运行
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            g.replay()
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    else:
        # 常规运行
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                model(input_tensor)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                
    times = np.array(times) * 1000  # 转换为毫秒
    return float(np.mean(times)), float(np.std(times)), float(np.percentile(times, 95))

def measure_throughput(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int
) -> Tuple[float, float]:
    """测量吞吐量"""
    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    total_tokens = batch_size * seq_len
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        
    total_time = end_time - start_time
    tokens_per_second = (total_tokens * num_runs) / total_time
    batches_per_second = num_runs / total_time
    
    return tokens_per_second, batches_per_second

def measure_memory(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> Dict[str, float]:
    """测量内存使用"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        model(input_tensor)
        
    stats = {
        'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
        'peak_allocated': torch.cuda.max_memory_allocated() / 1024**2,
        'reserved': torch.cuda.memory_reserved() / 1024**2,
        'peak_reserved': torch.cuda.max_memory_reserved() / 1024**2
    }
    
    return stats

def run_benchmark(model: nn.Module, args) -> Dict[str, Any]:
    """运行基准测试"""
    results = {}
    device = next(model.parameters()).device
    
    for batch_size in tqdm(args.batch_sizes, desc="Batch sizes"):
        for seq_len in tqdm(args.sequence_lengths, desc="Sequence lengths"):
            # 生成输入数据
            input_tensor = torch.randn(
                batch_size, seq_len, model.hidden_size,
                device=device
            )
            
            if args.fp16:
                input_tensor = input_tensor.half()
                
            if args.channels_last:
                input_tensor = input_tensor.to(memory_format=torch.channels_last)
                
            # 预热
            with torch.no_grad():
                for _ in range(args.warmup_runs):
                    model(input_tensor)
                    
            # 测试结果
            result = {}
            
            # 延迟测试
            if args.test_mode in ['latency', 'all']:
                mean_latency, std_latency, p95_latency = measure_latency(
                    model, input_tensor, args.num_runs, args.cuda_graphs
                )
                result['latency'] = {
                    'mean': mean_latency,
                    'std': std_latency,
                    'p95': p95_latency
                }
                
            # 吞吐量测试
            if args.test_mode in ['throughput', 'all']:
                tokens_per_second, batches_per_second = measure_throughput(
                    model, input_tensor, args.num_runs
                )
                result['throughput'] = {
                    'tokens_per_second': tokens_per_second,
                    'batches_per_second': batches_per_second
                }
                
            # 内存测试
            if args.test_mode in ['memory', 'all']:
                memory_stats = measure_memory(model, input_tensor)
                result['memory'] = memory_stats
                
            results[f"b{batch_size}_s{seq_len}"] = result
            
    return results

def save_results(results: Dict[str, Any], args):
    """保存测试结果"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始结果
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    # 生成可视化
    if args.plot_results:
        plot_benchmark_results(
            results,
            output_dir / "benchmark_summary.png"
        )
        
        if args.test_mode in ['latency', 'all']:
            plot_latency_distribution(
                results,
                output_dir / "latency_distribution.png"
            )
            
        if args.num_gpus > 1:
            plot_scaling_efficiency(
                results,
                output_dir / "scaling_efficiency.png"
            )
            
        if args.test_mode in ['memory', 'all']:
            plot_memory_usage(
                results,
                output_dir / "memory_usage.png"
            )

def generate_report(results: Dict[str, Any], args) -> str:
    """生成测试报告"""
    report = []
    report.append("# DANON基准测试报告")
    report.append(f"\n## 测试配置")
    report.append(f"- 模型类型: {args.model_type}")
    report.append(f"- 批处理大小: {args.batch_sizes}")
    report.append(f"- 序列长度: {args.sequence_lengths}")
    report.append(f"- GPU数量: {args.num_gpus}")
    report.append(f"- FP16: {args.fp16}")
    report.append(f"- Channels Last: {args.channels_last}")
    report.append(f"- CUDA Graphs: {args.cuda_graphs}")
    
    report.append("\n## 性能摘要")
    
    # 延迟统计
    if args.test_mode in ['latency', 'all']:
        latencies = [
            r['latency']['mean']
            for r in results.values()
            if 'latency' in r
        ]
        report.append(f"\n### 延迟 (ms)")
        report.append(f"- 最小: {min(latencies):.2f}")
        report.append(f"- 最大: {max(latencies):.2f}")
        report.append(f"- 平均: {np.mean(latencies):.2f}")
        report.append(f"- 中位数: {np.median(latencies):.2f}")
        
    # 吞吐量统计
    if args.test_mode in ['throughput', 'all']:
        throughputs = [
            r['throughput']['tokens_per_second']
            for r in results.values()
            if 'throughput' in r
        ]
        report.append(f"\n### 吞吐量")
        report.append(f"- 最大吞吐量: {max(throughputs):.2f} tokens/s")
        report.append(f"- 平均吞吐量: {np.mean(throughputs):.2f} tokens/s")
        
    # 内存使用统计
    if args.test_mode in ['memory', 'all']:
        peak_memories = [
            r['memory']['peak_allocated']
            for r in results.values()
            if 'memory' in r
        ]
        report.append(f"\n### 内存使用 (MB)")
        report.append(f"- 最大内存: {max(peak_memories):.2f}")
        report.append(f"- 平均内存: {np.mean(peak_memories):.2f}")
        
    # 优化建议
    report.append("\n## 优化建议")
    
    # 基于延迟的建议
    if args.test_mode in ['latency', 'all']:
        mean_latency = np.mean(latencies)
        if mean_latency > 100:  # 100ms阈值
            report.append("- 考虑使用更大的批处理大小来提高吞吐量")
            report.append("- 启用CUDA图以减少CPU开销")
            
    # 基于吞吐量的建议
    if args.test_mode in ['throughput', 'all']:
        if not args.fp16:
            report.append("- 考虑启用FP16以提高吞吐量")
        if not args.channels_last:
            report.append("- 考虑使用channels_last内存格式优化性能")
            
    # 基于内存使用的建议
    if args.test_mode in ['memory', 'all']:
        max_memory = max(peak_memories)
        if max_memory > 8 * 1024:  # 8GB阈值
            report.append("- 考虑使用梯度检查点减少内存使用")
            report.append("- 减小批处理大小或使用梯度累积")
            
    return "\n".join(report)

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置模型
    logger.info("Setting up model...")
    model = setup_model(args)
    
    # 运行基准测试
    logger.info("Running benchmarks...")
    results = run_benchmark(model, args)
    
    # 保存结果
    logger.info("Saving results...")
    save_results(results, args)
    
    # 生成报告
    logger.info("Generating report...")
    report = generate_report(results, args)
    
    # 保存报告
    report_path = Path(args.output_dir) / "benchmark_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
        
    logger.info(f"Benchmark results saved to {args.output_dir}")
    
if __name__ == "__main__":
    main() 