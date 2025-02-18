"""
DANON分布式训练模块
提供了增强的分布式训练功能
"""

from .enhanced_manager import EnhancedDistributedManager
from .communication import (
    CommunicationOptimizer,
    CompressionConfig,
    CompressionMethod
)

__all__ = [
    'EnhancedDistributedManager',
    'CommunicationOptimizer',
    'CompressionConfig',
    'CompressionMethod'
] 