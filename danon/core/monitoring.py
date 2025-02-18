"""
统一性能监控模块
提供了性能统计、日志记录和分析功能
"""

import time
from threading import Lock
from typing import Dict, Any, List, Optional
import logging
from collections import deque
import torch
import numpy as np
from dataclasses import dataclass
from .unified_config import UnifiedAttentionConfig

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    compute_time: float
    memory_usage: float
    attention_score: float
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class PerformanceMonitor:
    """性能监控器，负责收集和分析性能数据"""
    
    def __init__(self, config: UnifiedAttentionConfig):
        self.config = config
        self._stats_lock = Lock()
        self._stats = {
            'compute_times': deque(maxlen=config.monitoring_window_size),
            'memory_usage': deque(maxlen=config.monitoring_window_size),
            'attention_scores': deque(maxlen=config.monitoring_window_size),
            'cache_hits': deque(maxlen=config.monitoring_window_size),
            'cache_misses': deque(maxlen=config.monitoring_window_size)
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """配置日志器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def update_stats(self, metrics: PerformanceMetrics):
        """更新性能统计
        
        Args:
            metrics: 性能指标数据
        """
        with self._stats_lock:
            self._stats['compute_times'].append(metrics.compute_time)
            self._stats['memory_usage'].append(metrics.memory_usage)
            self._stats['attention_scores'].append(metrics.attention_score)
            self._stats['cache_hits'].append(metrics.cache_hits)
            self._stats['cache_misses'].append(metrics.cache_misses)
            
            # 记录异常值
            self._check_anomalies(metrics)
    
    def _check_anomalies(self, metrics: PerformanceMetrics):
        """检查性能异常
        
        Args:
            metrics: 当前性能指标
        """
        # 计算计算时间的异常值
        if len(self._stats['compute_times']) > 10:
            mean_time = np.mean(list(self._stats['compute_times'])[:-1])
            std_time = np.std(list(self._stats['compute_times'])[:-1])
            if metrics.compute_time > mean_time + 3 * std_time:
                self.logger.warning(
                    f"Anomaly detected: Compute time ({metrics.compute_time:.2f}s) "
                    f"is significantly higher than average ({mean_time:.2f}s)"
                )
        
        # 检查内存使用异常
        if metrics.memory_usage > 1024:  # 超过1GB
            self.logger.warning(
                f"High memory usage detected: {metrics.memory_usage:.2f}MB"
            )
        
        # 检查缓存命中率异常
        if metrics.cache_hit_rate < 0.5:  # 低于50%
            self.logger.warning(
                f"Low cache hit rate: {metrics.cache_hit_rate:.2%}"
            )
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息汇总
        
        Returns:
            Dict[str, float]: 统计信息字典
        """
        with self._stats_lock:
            if not self._stats['compute_times']:
                return {
                    'avg_compute_time': 0.0,
                    'avg_memory_usage': 0.0,
                    'avg_attention_score': 0.0,
                    'cache_hit_rate': 0.0
                }
            
            window = min(100, len(self._stats['compute_times']))
            return {
                'avg_compute_time': np.mean(list(self._stats['compute_times'])[-window:]),
                'avg_memory_usage': np.mean(list(self._stats['memory_usage'])[-window:]),
                'avg_attention_score': np.mean(list(self._stats['attention_scores'])[-window:]),
                'cache_hit_rate': (
                    sum(list(self._stats['cache_hits'])[-window:]) /
                    (sum(list(self._stats['cache_hits'])[-window:]) + 
                     sum(list(self._stats['cache_misses'])[-window:]) + 1e-10)
                )
            }
    
    def get_performance_report(self) -> str:
        """生成性能报告
        
        Returns:
            str: 格式化的性能报告
        """
        stats = self.get_stats()
        return f"""Performance Report:
        Average Compute Time: {stats['avg_compute_time']:.3f}s
        Average Memory Usage: {stats['avg_memory_usage']:.2f}MB
        Average Attention Score: {stats['avg_attention_score']:.4f}
        Cache Hit Rate: {stats['cache_hit_rate']:.2%}
        """
    
    def reset_stats(self):
        """重置所有统计信息"""
        with self._stats_lock:
            for key in self._stats:
                self._stats[key].clear()

class PerformanceContext:
    """性能监控上下文管理器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self) -> 'PerformanceContext':
        self.start_time = time.time()
        self.start_memory = (
            torch.cuda.memory_allocated() / 1024**2 
            if torch.cuda.is_available() 
            else 0
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            compute_time = time.time() - self.start_time
            current_memory = (
                torch.cuda.memory_allocated() / 1024**2 
                if torch.cuda.is_available() 
                else 0
            )
            memory_usage = current_memory - self.start_memory
            
            metrics = PerformanceMetrics(
                compute_time=compute_time,
                memory_usage=memory_usage,
                attention_score=0.0  # 需要外部设置
            )
            self.monitor.update_stats(metrics) 