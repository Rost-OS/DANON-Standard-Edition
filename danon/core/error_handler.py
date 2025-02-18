"""
统一的错误处理模块
提供了错误处理、恢复和监控功能
"""
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Type, Union, List
from functools import wraps
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from threading import Lock
import numpy as np
from enum import Enum

class ErrorSeverity(Enum):
    """错误严重程度"""
    CRITICAL = "critical" #系统不可用
    ERROR = "error" #严重错误
    WARNING = "warning" #警告
    INFO = "info" #信息

class DANONError(Exception):
    """DANON 框架的基础异常类"""
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.severity = ErrorSeverity(severity)
        self.timestamp = time.time()
        self.stack_trace = traceback.format_exc()

class ModelError(DANONError):
    """模型相关错误"""
    pass

class MemoryError(DANONError):
    """内存相关错误"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        super().__init__(
            "MEMORY_ERROR",
            message,
            details,
            severity.value
        )

class ComputationError(DANONError):
    """计算相关错误"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        super().__init__(
            "COMPUTATION_ERROR",
            message,
            details,
            severity.value
        )

class ConfigurationError(DANONError):
    """配置相关错误"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        super().__init__(
            "CONFIG_ERROR",
            message,
            details,
            severity.value
        )

class DistributedError(DANONError):
    """分布式训练相关错误"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        super().__init__(
            "DISTRIBUTED_ERROR",
            message,
            details,
            severity.value
        )

class RecoveryStrategy:
    """错误恢复策略"""
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: float = 30.0
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.current_retry = 0
        self.last_attempt = 0
        self.success_count = 0
        self.failure_count = 0
        
    def should_retry(self) -> bool:
        """判断是否应该重试"""
        if self.current_retry >= self.max_retries:
            return False
            
        # 检查是否超时
        if time.time() - self.last_attempt < self.get_wait_time():
            return False
            
        return True
        
    def get_wait_time(self) -> float:
        """获取重试等待时间"""
        return min(
            self.backoff_factor ** self.current_retry,
            self.timeout
        )
        
    def record_attempt(self, success: bool):
        """记录尝试结果"""
        if success:
            self.success_count += 1
            self.current_retry = 0  # 重置重试计数
        else:
            self.failure_count += 1
            self.current_retry += 1
            
        self.last_attempt = time.time()
        
    def get_success_rate(self) -> float:
        """获取成功率"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
        
    def reset(self):
        """重置状态"""
        self.current_retry = 0
        self.last_attempt = 0
        
class ErrorTracker:
    """错误追踪器"""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.errors: Dict[str, int] = {}
        self._lock = Lock()
        
        # 设置日志
        self.logger = logging.getLogger("ErrorTracker")
        handler = logging.FileHandler(self.log_dir / "errors.log")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        self.logger.addHandler(handler)
        
    def track(self, error: DANONError) -> None:
        """追踪错误"""
        with self._lock:
            if error.error_code not in self.errors:
                self.errors[error.error_code] = 0
            self.errors[error.error_code] += 1
            
        # 记录错误
        self.logger.error(
            f"Error {error.error_code}: {str(error)}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                "stack_trace": traceback.format_exc()
            }
        )
        
    def get_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        with self._lock:
            return dict(self.errors)
            
    def save_report(self) -> None:
        """保存错误报告"""
        report = {
            "timestamp": time.time(),
            "error_counts": self.get_stats()
        }
        
        report_path = self.log_dir / f"error_report_{int(time.time())}.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

class ErrorHandler:
    """错误处理器"""
    def __init__(
        self,
        config: Any,
        log_dir: Optional[str] = None,
        enable_recovery: bool = True,
        max_error_history: int = 1000
    ):
        self.config = config
        self.log_dir = Path(log_dir) if log_dir else None
        self.enable_recovery = enable_recovery
        self.max_error_history = max_error_history
        
        # 错误历史
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        
        # 恢复策略
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {
            "MEMORY_ERROR": RecoveryStrategy(max_retries=3),
            "COMPUTATION_ERROR": RecoveryStrategy(max_retries=5),
            "CONFIG_ERROR": RecoveryStrategy(max_retries=2),
            "DISTRIBUTED_ERROR": RecoveryStrategy(max_retries=3),
            "SYNC_ERROR": RecoveryStrategy(max_retries=3),
            "IO_ERROR": RecoveryStrategy(max_retries=3),
            "NETWORK_ERROR": RecoveryStrategy(max_retries=5)
        }
        
        # 错误处理回调
        self.error_callbacks: Dict[str, List[callable]] = {}
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 配置主日志处理器
            logging.basicConfig(
                filename=self.log_dir / "error.log",
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 添加控制台处理器用于关键错误
            console = logging.StreamHandler()
            console.setLevel(logging.ERROR)
            logging.getLogger('').addHandler(console)
            
    def register_callback(self, error_code: str, callback: callable):
        """注册错误处理回调函数"""
        if error_code not in self.error_callbacks:
            self.error_callbacks[error_code] = []
        self.error_callbacks[error_code].append(callback)
        
    def handle(self, error: DANONError) -> bool:
        """处理错误"""
        # 记录错误
        self._log_error(error)
        
        # 更新错误计数
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        
        # 执行错误回调
        if error.error_code in self.error_callbacks:
            for callback in self.error_callbacks[error.error_code]:
                try:
                    callback(error)
                except Exception as e:
                    logging.error(f"Error callback failed: {str(e)}")
                    
        # 检查是否可以恢复
        if self.enable_recovery and error.error_code in self.recovery_strategies:
            strategy = self.recovery_strategies[error.error_code]
            if strategy.should_retry():
                return self._attempt_recovery(error, strategy)
                
        return False
        
    def _log_error(self, error: DANONError):
        """记录错误信息"""
        error_info = {
            'code': error.error_code,
            'message': str(error),
            'details': error.details,
            'timestamp': error.timestamp,
            'severity': error.severity.value,
            'stack_trace': error.stack_trace
        }
        
        # 添加到历史记录
        self.error_history.append(error_info)
        
        # 限制历史记录大小
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
            
        # 记录到日志文件
        if self.log_dir:
            log_level = getattr(logging, error.severity.value.upper())
            logging.log(
                log_level,
                f"Error occurred: {error.error_code}\n"
                f"Message: {str(error)}\n"
                f"Details: {json.dumps(error.details, indent=2)}\n"
                f"Stack trace:\n{error.stack_trace}"
            )
            
    def _attempt_recovery(self, error: DANONError, strategy: RecoveryStrategy) -> bool:
        """尝试恢复"""
        wait_time = strategy.get_wait_time()
        
        logging.info(
            f"Attempting recovery for {error.error_code} "
            f"(attempt {strategy.current_retry + 1}/{strategy.max_retries})"
        )
        
        # 等待一段时间
        time.sleep(wait_time)
        
        success = False
        try:
            # 根据错误类型执行恢复操作
            if error.error_code == "MEMORY_ERROR":
                success = self._recover_from_memory_error(error)
            elif error.error_code == "COMPUTATION_ERROR":
                success = self._recover_from_computation_error(error)
            elif error.error_code == "DISTRIBUTED_ERROR":
                success = self._recover_from_distributed_error(error)
            elif error.error_code == "SYNC_ERROR":
                success = self._recover_from_sync_error(error)
            elif error.error_code == "IO_ERROR":
                success = self._recover_from_io_error(error)
            elif error.error_code == "NETWORK_ERROR":
                success = self._recover_from_network_error(error)
        except Exception as e:
            logging.error(f"Recovery attempt failed: {str(e)}")
            success = False
            
        # 记录恢复尝试结果
        strategy.record_attempt(success)
        
        return success
        
    def _recover_from_memory_error(self, error: DANONError) -> bool:
        """从内存错误中恢复"""
        try:
            import gc
            import torch
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理 PyTorch 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return True
        except Exception:
            return False
            
    def _recover_from_computation_error(self, error: DANONError) -> bool:
        """从计算错误中恢复"""
        try:
            import torch
            
            # 重置 CUDA 设备
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_device = torch.cuda.current_device()
                torch.cuda.device(current_device).empty_cache()
                
            return True
        except Exception:
            return False
            
    def _recover_from_distributed_error(self, error: DANONError) -> bool:
        """从分布式错误中恢复"""
        try:
            import torch.distributed as dist
            
            # 重新初始化进程组
            if dist.is_initialized():
                dist.destroy_process_group()
                
            # 注意：实际的重新初始化应该由分布式管理器处理
            return True
        except Exception:
            return False
            
    def _recover_from_sync_error(self, error: DANONError) -> bool:
        """从同步错误中恢复"""
        try:
            import torch.distributed as dist
            
            # 尝试同步所有进程
            if dist.is_initialized():
                dist.barrier()
                
            return True
        except Exception:
            return False
            
    def _recover_from_io_error(self, error: DANONError) -> bool:
        """从IO错误中恢复"""
        try:
            # 可以实现文件系统检查、重试等逻辑
            return True
        except Exception:
            return False
            
    def _recover_from_network_error(self, error: DANONError) -> bool:
        """从网络错误中恢复"""
        try:
            import socket
            import torch.distributed as dist
            
            # 检查网络连接
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            # 如果使用分布式训练，尝试重新同步
            if dist.is_initialized():
                dist.barrier()
                
            return True
        except Exception:
            return False
            
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {}
            
        stats = {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'error_types': {},
            'recovery_stats': {}
        }
        
        # 按错误类型统计
        for error in self.error_history:
            error_type = error['code']
            if error_type not in stats['error_types']:
                stats['error_types'][error_type] = {
                    'count': 0,
                    'last_occurrence': None,
                    'severity_distribution': {
                        severity.value: 0 for severity in ErrorSeverity
                    }
                }
                
            stats['error_types'][error_type]['count'] += 1
            stats['error_types'][error_type]['last_occurrence'] = error['timestamp']
            stats['error_types'][error_type]['severity_distribution'][error['severity']] += 1
            
        # 恢复策略统计
        for error_code, strategy in self.recovery_strategies.items():
            stats['recovery_stats'][error_code] = {
                'success_rate': strategy.get_success_rate(),
                'total_attempts': strategy.success_count + strategy.failure_count,
                'current_retry': strategy.current_retry
            }
            
        return stats
        
    def clear_history(self):
        """清除错误历史"""
        self.error_history.clear()
        self.error_counts.clear()
        for strategy in self.recovery_strategies.values():
            strategy.reset()
            
    def check_gradient_norm(self, model: nn.Module) -> bool:
        """检查梯度范数"""
        try:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.config.max_grad_norm:
                error = ComputationError(
                    "梯度范数过大",
                    {
                        "gradient_norm": total_norm,
                        "max_allowed": self.config.max_grad_norm
                    }
                )
                return self.handle(error)
            return True
        except Exception as e:
            logging.error(f"Gradient check failed: {str(e)}")
            return False
            
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                
                if allocated > total * 0.95:  # 95%阈值
                    error = MemoryError(
                        "GPU内存使用率过高",
                        {
                            "allocated": allocated,
                            "reserved": reserved,
                            "total": total
                        }
                    )
                    return self.handle(error)
            return True
        except Exception as e:
            logging.error(f"Memory check failed: {str(e)}")
            return False
            
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        try:
            # 检查必需字段
            required_fields = ["hidden_size", "num_layers", "dropout"]
            for field in required_fields:
                if field not in config:
                    error = ConfigurationError(
                        f"缺少必需的配置字段: {field}",
                        {"missing_field": field}
                    )
                    return self.handle(error)
                    
            # 验证值的范围
            if config["hidden_size"] <= 0:
                error = ConfigurationError(
                    "hidden_size必须为正数",
                    {
                        "field": "hidden_size",
                        "value": config["hidden_size"]
                    }
                )
                return self.handle(error)
                
            if not 0 <= config["dropout"] <= 1:
                error = ConfigurationError(
                    "dropout必须在0和1之间",
                    {
                        "field": "dropout",
                        "value": config["dropout"]
                    }
                )
                return self.handle(error)
                
            return True
        except Exception as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            return False
            
    def __call__(self, error: Exception) -> None:
        """使错误处理器可调用"""
        danon_error = self._convert_error(error)
        self.handle(danon_error) 