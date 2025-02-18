"""
兼容性管理模块
提供新旧系统的兼容性保证和平滑迁移方案
"""

import warnings
from typing import Any, Dict, Optional, Type, Union
import torch
from .base_config import BaseConfig
from .unified_config import UnifiedAttentionConfig, MSRAConfig, InfiniteAttentionConfig
from .monitoring import PerformanceMonitor
from .error_handling import ErrorHandler
from .caching import CacheManager

class CompatibilityManager:
    """兼容性管理器，确保新旧系统的平滑过渡"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.legacy_configs = {}
            self.config_migrations = {}
            self.feature_flags = {
                'use_enhanced_monitoring': True,
                'use_enhanced_caching': True,
                'use_enhanced_error_handling': True
            }
            self.initialized = True
    
    def register_legacy_config(
        self,
        config_class: Type[BaseConfig],
        migration_fn: callable
    ):
        """注册旧配置类及其迁移函数"""
        self.legacy_configs[config_class] = migration_fn
    
    def migrate_config(
        self,
        config: Union[BaseConfig, Dict[str, Any]]
    ) -> UnifiedAttentionConfig:
        """迁移配置到新系统"""
        if isinstance(config, UnifiedAttentionConfig):
            return config
            
        if isinstance(config, dict):
            return UnifiedAttentionConfig(**config)
            
        config_class = type(config)
        if config_class in self.legacy_configs:
            return self.legacy_configs[config_class](config)
            
        # 默认迁移策略
        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_')
        }
        return UnifiedAttentionConfig(**config_dict)
    
    def get_monitor(
        self,
        config: Union[BaseConfig, UnifiedAttentionConfig]
    ) -> PerformanceMonitor:
        """获取适当的性能监控器"""
        if isinstance(config, UnifiedAttentionConfig):
            return PerformanceMonitor(config)
            
        # 为旧配置创建兼容的监控器
        unified_config = self.migrate_config(config)
        return PerformanceMonitor(unified_config)
    
    def get_error_handler(self) -> ErrorHandler:
        """获取错误处理器"""
        return ErrorHandler()
    
    def get_cache_manager(
        self,
        config: Union[BaseConfig, UnifiedAttentionConfig]
    ) -> CacheManager:
        """获取缓存管理器"""
        if isinstance(config, UnifiedAttentionConfig):
            return CacheManager(config)
            
        unified_config = self.migrate_config(config)
        return CacheManager(unified_config)
    
    def enable_feature(self, feature_name: str):
        """启用特定功能"""
        if feature_name in self.feature_flags:
            self.feature_flags[feature_name] = True
    
    def disable_feature(self, feature_name: str):
        """禁用特定功能"""
        if feature_name in self.feature_flags:
            self.feature_flags[feature_name] = False
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """检查功能是否启用"""
        return self.feature_flags.get(feature_name, False)
    
    @staticmethod
    def wrap_legacy_model(model: torch.nn.Module) -> torch.nn.Module:
        """包装旧模型以支持新特性"""
        if not hasattr(model, '_wrapped'):
            # 添加新特性支持
            model._wrapped = True
            model._compatibility_manager = CompatibilityManager()
            
            # 包装forward方法
            original_forward = model.forward
            
            def wrapped_forward(*args, **kwargs):
                # 在这里添加新特性支持
                if model._compatibility_manager.is_feature_enabled('use_enhanced_monitoring'):
                    with PerformanceContext(model):
                        return original_forward(*args, **kwargs)
                return original_forward(*args, **kwargs)
            
            model.forward = wrapped_forward
        
        return model

# 默认迁移函数
def default_config_migration(legacy_config: BaseConfig) -> UnifiedAttentionConfig:
    """默认的配置迁移函数"""
    config_dict = {}
    
    # 基础配置迁移
    for key in UnifiedAttentionConfig.__annotations__:
        if hasattr(legacy_config, key):
            config_dict[key] = getattr(legacy_config, key)
    
    # 特殊处理
    if hasattr(legacy_config, 'enable_monitoring'):
        config_dict['enable_monitoring'] = legacy_config.enable_monitoring
    else:
        config_dict['enable_monitoring'] = True
        
    if hasattr(legacy_config, 'enable_cache'):
        config_dict['enable_cache'] = legacy_config.enable_cache
    else:
        config_dict['enable_cache'] = True
        
    return UnifiedAttentionConfig(**config_dict)

# 注册默认迁移函数
CompatibilityManager().register_legacy_config(BaseConfig, default_config_migration)

class PerformanceContext:
    """性能监控上下文"""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.start_time.record()
        
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.end_time.record()
            torch.cuda.synchronize()
            
            compute_time = self.start_time.elapsed_time(self.end_time) / 1000.0
            
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_used = end_memory - self.start_memory
            else:
                memory_used = 0
            
            # 更新性能统计
            if hasattr(self.model, '_compatibility_manager'):
                monitor = self.model._compatibility_manager.get_monitor(
                    getattr(self.model, 'config', None)
                )
                if monitor:
                    monitor.update_stats({
                        'compute_time': compute_time,
                        'memory_usage': memory_used,
                        'attention_score': 0.0  # 需要实际的注意力分数
                    })

def ensure_compatibility(func):
    """确保兼容性的装饰器"""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_compatibility_manager'):
            self._compatibility_manager = CompatibilityManager()
            self = CompatibilityManager.wrap_legacy_model(self)
        return func(self, *args, **kwargs)
    return wrapper 