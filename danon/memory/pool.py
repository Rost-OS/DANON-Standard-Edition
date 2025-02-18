"""
增强型内存池管理系统
实现了智能内存分配、碎片管理和自适应优化
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import time
import threading
import numpy as np
from collections import defaultdict
import logging

class MemoryBlockType(Enum):
    """内存块类型"""
    SMALL = "small"      # < 1MB
    MEDIUM = "medium"    # 1MB - 10MB
    LARGE = "large"      # > 10MB
    HUGE = "huge"       # > 100MB

@dataclass
class MemoryBlock:
    """内存块"""
    tensor: torch.Tensor
    size: int
    block_type: MemoryBlockType
    last_use: float
    use_count: int
    is_free: bool
    fragmentation: float = 0.0
    allocation_time: float = 0.0
    deallocation_time: Optional[float] = None
    metadata: Dict[str, Any] = None

class EnhancedMemoryPool:
    """增强型内存池"""
    
    def __init__(
        self,
        block_type: MemoryBlockType,
        initial_size: int = 1024*1024,  # 1MB
        growth_factor: float = 1.5,
        max_size: Optional[int] = None,
        defrag_threshold: float = 0.3,  # 碎片率阈值
        allocation_strategy: str = 'best_fit'  # best_fit, first_fit, worst_fit
    ):
        self.block_type = block_type
        self.initial_size = initial_size
        self.growth_factor = growth_factor
        self.max_size = max_size
        self.defrag_threshold = defrag_threshold
        self.allocation_strategy = allocation_strategy
        
        # 内存块管理
        self.blocks: List[MemoryBlock] = []
        self.free_blocks: List[MemoryBlock] = []
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        
        # 性能统计
        self.total_allocations = 0
        self.total_deallocations = 0
        self.peak_memory = 0
        self.current_memory = 0
        self.fragmentation_ratio = 0.0
        
        # 创建初始块
        self._create_initial_block()
        
    def _create_initial_block(self):
        """创建初始内存块"""
        initial_tensor = torch.empty(
            self.initial_size,
            dtype=torch.uint8,
            device='cuda'
        )
        
        block = MemoryBlock(
            tensor=initial_tensor,
            size=self.initial_size,
            block_type=self.block_type,
            last_use=time.time(),
            use_count=0,
            is_free=True,
            allocation_time=time.time()
        )
        
        self.blocks.append(block)
        self.free_blocks.append(block)
        
    def _find_best_fit(self, size: int) -> Optional[MemoryBlock]:
        """查找最佳匹配的空闲块"""
        best_block = None
        min_waste = float('inf')
        
        for block in self.free_blocks:
            waste = block.size - size
            if waste >= 0 and waste < min_waste:
                best_block = block
                min_waste = waste
                
        return best_block
        
    def _find_first_fit(self, size: int) -> Optional[MemoryBlock]:
        """查找首个满足条件的空闲块"""
        for block in self.free_blocks:
            if block.size >= size:
                return block
        return None
        
    def _find_worst_fit(self, size: int) -> Optional[MemoryBlock]:
        """查找最大的空闲块"""
        worst_block = None
        max_size = -1
        
        for block in self.free_blocks:
            if block.size >= size and block.size > max_size:
                worst_block = block
                max_size = block.size
                
        return worst_block
        
    def allocate(self, size: int) -> Optional[torch.Tensor]:
        """分配内存"""
        # 查找合适的块
        if self.allocation_strategy == 'best_fit':
            block = self._find_best_fit(size)
        elif self.allocation_strategy == 'first_fit':
            block = self._find_first_fit(size)
        else:  # worst_fit
            block = self._find_worst_fit(size)
            
        # 如果没有找到合适的块
        if block is None:
            # 检查是否需要碎片整理
            if self._should_defrag():
                self.defragment()
                return self.allocate(size)
                
            # 检查是否需要扩展池
            if self._should_grow():
                self._grow_pool()
                return self.allocate(size)
                
            return None
            
        # 分配内存
        tensor = self._allocate_from_block(block, size)
        
        # 更新统计信息
        self.total_allocations += 1
        self.current_memory += size
        self.peak_memory = max(self.peak_memory, self.current_memory)
        self._update_fragmentation_ratio()
        
        return tensor
        
    def _allocate_from_block(
        self,
        block: MemoryBlock,
        size: int
    ) -> torch.Tensor:
        """从块中分配内存"""
        remaining_size = block.size - size
        
        if remaining_size > 0:
            # 分割块
            self._split_block(block, size, remaining_size)
        else:
            # 使用整个块
            self.free_blocks.remove(block)
            
        block.is_free = False
        block.last_use = time.time()
        block.use_count += 1
        
        # 记录分配
        self.allocated_blocks[id(block.tensor)] = block
        
        return block.tensor
        
    def _split_block(
        self,
        block: MemoryBlock,
        size: int,
        remaining_size: int
    ):
        """分割内存块"""
        # 创建新的块
        new_tensor = block.tensor[size:].view(-1)
        new_block = MemoryBlock(
            tensor=new_tensor,
            size=remaining_size,
            block_type=self.block_type,
            last_use=time.time(),
            use_count=0,
            is_free=True,
            allocation_time=time.time()
        )
        
        # 更新原块
        block.tensor = block.tensor[:size].view(-1)
        block.size = size
        
        # 更新列表
        self.blocks.append(new_block)
        self.free_blocks.append(new_block)
        self.free_blocks.remove(block)
        
    def free(self, tensor: torch.Tensor):
        """释放内存"""
        block_id = id(tensor)
        if block_id not in self.allocated_blocks:
            return
            
        block = self.allocated_blocks[block_id]
        block.is_free = True
        block.deallocation_time = time.time()
        
        # 合并相邻的空闲块
        self._merge_adjacent_blocks(block)
        
        # 更新统计信息
        self.total_deallocations += 1
        self.current_memory -= block.size
        self._update_fragmentation_ratio()
        
        # 清理记录
        del self.allocated_blocks[block_id]
        
    def _merge_adjacent_blocks(self, block: MemoryBlock):
        """合并相邻的空闲块"""
        idx = self.blocks.index(block)
        
        # 检查前一个块
        if idx > 0 and self.blocks[idx-1].is_free:
            block = self._merge_blocks(self.blocks[idx-1], block)
            idx -= 1
            
        # 检查后一个块
        if idx < len(self.blocks)-1 and self.blocks[idx+1].is_free:
            block = self._merge_blocks(block, self.blocks[idx+1])
            
    def _merge_blocks(self, first: MemoryBlock, second: MemoryBlock) -> MemoryBlock:
        """合并两个内存块"""
        # 创建新的合并块
        merged_tensor = torch.cat([first.tensor, second.tensor])
        merged_block = MemoryBlock(
            tensor=merged_tensor,
            size=first.size + second.size,
            block_type=self.block_type,
            last_use=max(first.last_use, second.last_use),
            use_count=max(first.use_count, second.use_count),
            is_free=True,
            allocation_time=min(first.allocation_time, second.allocation_time)
        )
        
        # 更新列表
        self.blocks.remove(first)
        self.blocks.remove(second)
        self.blocks.append(merged_block)
        
        if first in self.free_blocks:
            self.free_blocks.remove(first)
        if second in self.free_blocks:
            self.free_blocks.remove(second)
        self.free_blocks.append(merged_block)
        
        return merged_block
        
    def defragment(self):
        """内存碎片整理"""
        if not self._should_defrag():
            return
            
        # 收集所有已分配的块
        allocated_blocks = [
            block for block in self.blocks
            if not block.is_free
        ]
        
        # 计算总的已分配大小
        total_allocated = sum(block.size for block in allocated_blocks)
        
        # 创建新的连续内存
        new_tensor = torch.empty(
            self.initial_size,
            dtype=torch.uint8,
            device='cuda'
        )
        
        # 复制数据
        offset = 0
        for block in allocated_blocks:
            size = block.size
            new_tensor[offset:offset+size].copy_(block.tensor)
            
            # 更新块信息
            block.tensor = new_tensor[offset:offset+size].view(-1)
            offset += size
            
        # 创建新的空闲块
        if offset < self.initial_size:
            free_tensor = new_tensor[offset:].view(-1)
            free_block = MemoryBlock(
                tensor=free_tensor,
                size=self.initial_size - offset,
                block_type=self.block_type,
                last_use=time.time(),
                use_count=0,
                is_free=True,
                allocation_time=time.time()
            )
            
            # 更新列表
            self.blocks = allocated_blocks + [free_block]
            self.free_blocks = [free_block]
            
        # 更新碎片率
        self._update_fragmentation_ratio()
        
    def _should_defrag(self) -> bool:
        """判断是否需要进行碎片整理"""
        return (
            self.fragmentation_ratio > self.defrag_threshold and
            len(self.blocks) > 1
        )
        
    def _should_grow(self) -> bool:
        """判断是否需要扩展池"""
        if self.max_size and self.current_memory >= self.max_size:
            return False
            
        return (
            len(self.free_blocks) == 0 or
            max(block.size for block in self.free_blocks) < self.initial_size * 0.1
        )
        
    def _grow_pool(self):
        """扩展内存池"""
        new_size = int(self.initial_size * self.growth_factor)
        
        if self.max_size:
            new_size = min(new_size, self.max_size - self.current_memory)
            
        # 创建新的块
        new_tensor = torch.empty(
            new_size,
            dtype=torch.uint8,
            device='cuda'
        )
        
        new_block = MemoryBlock(
            tensor=new_tensor,
            size=new_size,
            block_type=self.block_type,
            last_use=time.time(),
            use_count=0,
            is_free=True,
            allocation_time=time.time()
        )
        
        # 更新列表
        self.blocks.append(new_block)
        self.free_blocks.append(new_block)
        self.initial_size = new_size
        
    def _update_fragmentation_ratio(self):
        """更新碎片率"""
        if not self.blocks:
            self.fragmentation_ratio = 0.0
            return
            
        total_size = sum(block.size for block in self.blocks)
        free_size = sum(block.size for block in self.free_blocks)
        largest_free = max(
            (block.size for block in self.free_blocks),
            default=0
        )
        
        if free_size == 0:
            self.fragmentation_ratio = 0.0
        else:
            self.fragmentation_ratio = 1.0 - (largest_free / free_size)
            
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        return {
            'block_type': self.block_type.value,
            'total_size': self.initial_size,
            'current_memory': self.current_memory,
            'peak_memory': self.peak_memory,
            'fragmentation_ratio': self.fragmentation_ratio,
            'total_blocks': len(self.blocks),
            'free_blocks': len(self.free_blocks),
            'allocated_blocks': len(self.allocated_blocks),
            'total_allocations': self.total_allocations,
            'total_deallocations': self.total_deallocations,
            'allocation_strategy': self.allocation_strategy
        }

class EnhancedMemoryPoolManager:
    """增强型内存池管理器"""
    
    def __init__(
        self,
        small_pool_size: int = 10 * 1024 * 1024,  # 10MB
        medium_pool_size: int = 100 * 1024 * 1024,  # 100MB
        large_pool_size: int = 1024 * 1024 * 1024,  # 1GB
        enable_monitoring: bool = True
    ):
        # 创建不同类型的内存池
        self.pools = {
            MemoryBlockType.SMALL: EnhancedMemoryPool(
                MemoryBlockType.SMALL,
                initial_size=small_pool_size
            ),
            MemoryBlockType.MEDIUM: EnhancedMemoryPool(
                MemoryBlockType.MEDIUM,
                initial_size=medium_pool_size
            ),
            MemoryBlockType.LARGE: EnhancedMemoryPool(
                MemoryBlockType.LARGE,
                initial_size=large_pool_size
            )
        }
        
        # 监控相关
        self.enable_monitoring = enable_monitoring
        self._monitoring_thread = None
        self._should_monitor = False
        
        # 性能统计
        self.allocation_patterns = defaultdict(list)
        self.optimization_suggestions = []
        
        # 启动监控
        if enable_monitoring:
            self.start_monitoring()
            
    def _get_block_type(self, size: int) -> MemoryBlockType:
        """确定内存块类型"""
        if size < 1024 * 1024:  # 1MB
            return MemoryBlockType.SMALL
        elif size < 10 * 1024 * 1024:  # 10MB
            return MemoryBlockType.MEDIUM
        elif size < 100 * 1024 * 1024:  # 100MB
            return MemoryBlockType.LARGE
        else:
            return MemoryBlockType.HUGE
            
    def allocate(self, size: int) -> torch.Tensor:
        """分配内存"""
        block_type = self._get_block_type(size)
        
        # 对于超大块，直接分配
        if block_type == MemoryBlockType.HUGE:
            return torch.empty(size, dtype=torch.uint8)
            
        # 从对应池中分配
        tensor = self.pools[block_type].allocate(size)
        if tensor is not None:
            return tensor
            
        # 如果分配失败，尝试从更大的池分配
        for larger_type in [t for t in MemoryBlockType if t.value > block_type.value]:
            if larger_type in self.pools:
                tensor = self.pools[larger_type].allocate(size)
                if tensor is not None:
                    return tensor
                    
        # 如果所有池都分配失败，直接分配
        return torch.empty(size, dtype=torch.uint8)
        
    def free(self, tensor: torch.Tensor):
        """释放内存"""
        size = tensor.numel() * tensor.element_size()
        block_type = self._get_block_type(size)
        
        # 对于超大块，直接释放
        if block_type == MemoryBlockType.HUGE:
            del tensor
            return
            
        # 在所有池中查找并释放
        for pool in self.pools.values():
            pool.free(tensor)
            
    def start_monitoring(self):
        """启动监控"""
        if self.enable_monitoring and not self._monitoring_thread:
            self._should_monitor = True
            self._monitoring_thread = threading.Thread(target=self._monitor_loop)
            self._monitoring_thread.start()
            
    def stop_monitoring(self):
        """停止监控"""
        self._should_monitor = False
        if self._monitoring_thread:
            self._monitoring_thread.join()
            self._monitoring_thread = None
            
    def _monitor_loop(self):
        """监控循环"""
        while self._should_monitor:
            # 收集统计信息
            stats = self.get_stats()
            
            # 分析内存使用模式
            self._analyze_memory_patterns(stats)
            
            # 生成优化建议
            self._generate_optimization_suggestions(stats)
            
            # 执行必要的维护
            self._perform_maintenance()
            
            time.sleep(1)
            
    def _analyze_memory_patterns(self, stats: Dict[str, Any]):
        """分析内存使用模式"""
        for pool_type, pool_stats in stats['pools'].items():
            recent_allocs = pool_stats['allocation_history']
            if recent_allocs:
                # 分析分配大小分布
                sizes = [alloc['size'] for alloc in recent_allocs]
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                
                self.allocation_patterns[pool_type].append({
                    'mean_size': mean_size,
                    'std_size': std_size,
                    'timestamp': time.time()
                })
                
    def _generate_optimization_suggestions(self, stats: Dict[str, Any]):
        """生成优化建议"""
        for pool_type, pool_stats in stats['pools'].items():
            # 检查碎片化
            if pool_stats['usage_ratio'] > 0.8:
                self.optimization_suggestions.append({
                    'type': 'high_usage',
                    'pool_type': pool_type,
                    'suggestion': f'考虑增加{pool_type}池的大小'
                })
                
            # 检查命中率
            if pool_stats['hit_rate'] < 0.7:
                self.optimization_suggestions.append({
                    'type': 'low_hit_rate',
                    'pool_type': pool_type,
                    'suggestion': '考虑调整内存块大小分布'
                })
                
    def _perform_maintenance(self):
        """执行维护操作"""
        for pool in self.pools.values():
            # 检查是否需要碎片整理
            if pool._should_defrag():
                pool.defragment()
                
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'pools': {
                pool_type: pool.get_stats()
                for pool_type, pool in self.pools.items()
            },
            'allocation_patterns': dict(self.allocation_patterns),
            'optimization_suggestions': self.optimization_suggestions[-10:],  # 最近10条建议
            'total_memory': sum(
                pool.get_stats()['total_size']
                for pool in self.pools.values()
            )
        }
        
        # 计算总体使用率
        total_used = sum(
            pool.get_stats()['used_size']
            for pool in self.pools.values()
        )
        stats['overall_usage_ratio'] = total_used / stats['total_memory']
        
        return stats

def create_memory_pool_manager(
    small_pool_size: Optional[int] = None,
    medium_pool_size: Optional[int] = None,
    large_pool_size: Optional[int] = None,
    enable_monitoring: bool = True
) -> EnhancedMemoryPoolManager:
    """创建内存池管理器"""
    return EnhancedMemoryPoolManager(
        small_pool_size=small_pool_size or 10 * 1024 * 1024,
        medium_pool_size=medium_pool_size or 100 * 1024 * 1024,
        large_pool_size=large_pool_size or 1024 * 1024 * 1024,
        enable_monitoring=enable_monitoring
    ) 