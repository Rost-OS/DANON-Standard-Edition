"""
计算图管理器
实现了动态计算图的构建、优化和执行
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from enum import Enum
import time

class FusionType(Enum):
    """算子融合类型"""
    VERTICAL = "vertical"  # 垂直融合（如连续的线性层）
    HORIZONTAL = "horizontal"  # 水平融合（如并行的注意力头）
    HYBRID = "hybrid"  # 混合融合

@dataclass
class MemoryPlan:
    """内存计划"""
    allocation_size: int
    reuse_map: Dict[str, List[str]]  # 张量重用映射
    peak_memory: int
    release_points: Dict[str, int]  # 张量释放点

@dataclass
class ComputeNode:
    """计算节点"""
    name: str
    operation: nn.Module
    inputs: List[str]
    outputs: List[str]
    meta: Dict[str, Any] = None
    memory_size: int = 0  # 预估内存使用量
    compute_cost: float = 0.0  # 预估计算成本

class FusedOperation(nn.Module):
    """融合算子"""
    def __init__(
        self,
        operations: List[nn.Module],
        fusion_type: FusionType
    ):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.fusion_type = fusion_type
        
    def forward(self, *args, **kwargs):
        if self.fusion_type == FusionType.VERTICAL:
            x = args[0]
            for op in self.operations:
                x = op(x)
            return x
        elif self.fusion_type == FusionType.HORIZONTAL:
            results = [op(*args, **kwargs) for op in self.operations]
            return torch.cat(results, dim=-1)
        else:  # HYBRID
            # 自适应决定融合方式
            x = args[0]
            intermediate_results = []
            for op in self.operations:
                result = op(x)
                intermediate_results.append(result)
                x = result
            return x, intermediate_results

class TensorPool:
    """智能张量池，用于管理和复用张量"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.pools = {}  # 按大小分类的张量池
        self.active_tensors = {}  # 当前活跃的张量
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_memory': 0
        }
        
    def _get_tensor_key(self, shape: Tuple[int, ...], dtype: torch.dtype) -> str:
        """生成张量键"""
        return f"{shape}_{dtype}"
        
    def acquire(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        name: str
    ) -> torch.Tensor:
        """获取张量"""
        key = self._get_tensor_key(shape, dtype)
        
        if key in self.pools and self.pools[key]:
            # 从池中获取
            tensor = self.pools[key].pop()
            self.stats['hits'] += 1
        else:
            # 创建新张量
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.stats['misses'] += 1
            self.stats['total_memory'] += tensor.numel() * tensor.element_size()
            
        self.active_tensors[name] = (tensor, key)
        return tensor
        
    def release(self, name: str) -> None:
        """释放张量"""
        if name in self.active_tensors:
            tensor, key = self.active_tensors.pop(name)
            if key not in self.pools:
                self.pools[key] = []
            self.pools[key].append(tensor)
            
    def clear(self) -> None:
        """清空张量池"""
        self.pools.clear()
        self.active_tensors.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'total_memory': 0
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.stats['hits'] + self.stats['misses']
        return {
            'hit_rate': self.stats['hits'] / total_requests if total_requests > 0 else 0,
            'total_memory_mb': self.stats['total_memory'] / (1024 * 1024),
            'active_tensors': len(self.active_tensors),
            'pool_sizes': {k: len(v) for k, v in self.pools.items()}
        }

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.tensor_pool = TensorPool(device)
        self.allocation_history = {}
        self.peak_memory = 0
        self.current_memory = 0
        
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        name: str
    ) -> torch.Tensor:
        """分配内存"""
        tensor = self.tensor_pool.acquire(shape, dtype, name)
        size = tensor.numel() * tensor.element_size()
        
        self.current_memory += size
        self.peak_memory = max(self.peak_memory, self.current_memory)
        self.allocation_history[name] = {
            'size': size,
            'shape': shape,
            'dtype': dtype,
            'time': time.time()
        }
        
        return tensor
        
    def free(self, name: str) -> None:
        """释放内存"""
        if name in self.allocation_history:
            size = self.allocation_history[name]['size']
            self.current_memory -= size
            self.tensor_pool.release(name)
            del self.allocation_history[name]
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return {
            'current_memory_mb': self.current_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'active_allocations': len(self.allocation_history),
            'pool_stats': self.tensor_pool.get_stats()
        }

class MemoryOptimizer:
    """内存优化器"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.checkpoints = {}
        self.reuse_pool = {}
        self.activation_sizes = {}
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_events = []
        
        # 智能内存池配置
        self.memory_pools = {
            'small': {'size': 1024*1024, 'tensors': []},      # 1MB以下
            'medium': {'size': 10*1024*1024, 'tensors': []},  # 1MB-10MB
            'large': {'size': float('inf'), 'tensors': []}    # 10MB以上
        }
        
        # 内存碎片追踪
        self.fragmentation_stats = {
            'total_fragments': 0,
            'fragment_sizes': [],
            'largest_fragment': 0
        }
        
        # 张量生命周期追踪
        self.tensor_lifecycle = {}
        self.current_step = 0
        
    def track_memory_usage(self, event: str, size: int, tensor_id: Optional[str] = None):
        """跟踪内存使用情况"""
        self.current_memory += size
        self.peak_memory = max(self.peak_memory, self.current_memory)
        
        event_data = {
            'event': event,
            'size': size,
            'current_memory': self.current_memory,
            'peak_memory': self.peak_memory,
            'timestamp': time.time(),
            'tensor_id': tensor_id
        }
        
        self.memory_events.append(event_data)
        
        # 更新张量生命周期
        if tensor_id is not None:
            if event.startswith('allocate'):
                self.tensor_lifecycle[tensor_id] = {
                    'birth_step': self.current_step,
                    'size': size,
                    'last_access': self.current_step
                }
            elif event.startswith('free'):
                if tensor_id in self.tensor_lifecycle:
                    self.tensor_lifecycle[tensor_id]['death_step'] = self.current_step
                    
    def _get_pool_for_size(self, size: int) -> str:
        """根据大小选择合适的内存池"""
        for pool_name, pool_info in self.memory_pools.items():
            if size <= pool_info['size']:
                return pool_name
        return 'large'
        
    def _allocate_from_pool(self, size: int) -> Optional[torch.Tensor]:
        """从内存池分配张量"""
        pool_name = self._get_pool_for_size(size)
        pool = self.memory_pools[pool_name]['tensors']
        
        # 查找最适合的张量
        best_fit = None
        best_fit_idx = -1
        min_waste = float('inf')
        
        for i, tensor in enumerate(pool):
            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size >= size:
                waste = tensor_size - size
                if waste < min_waste:
                    min_waste = waste
                    best_fit = tensor
                    best_fit_idx = i
                    
        if best_fit is not None:
            # 从池中移除
            pool.pop(best_fit_idx)
            return best_fit
            
        return None
        
    def _return_to_pool(self, tensor: torch.Tensor):
        """将张量返回到内存池"""
        size = tensor.numel() * tensor.element_size()
        pool_name = self._get_pool_for_size(size)
        self.memory_pools[pool_name]['tensors'].append(tensor)
        
    def _defragment_pool(self, pool_name: str):
        """对指定内存池进行碎片整理"""
        pool = self.memory_pools[pool_name]['tensors']
        if not pool:
            return
            
        # 按大小排序
        pool.sort(key=lambda x: x.numel() * x.element_size())
        
        # 尝试合并相邻的张量
        i = 0
        while i < len(pool) - 1:
            current = pool[i]
            next_tensor = pool[i + 1]
            
            current_size = current.numel() * current.element_size()
            next_size = next_tensor.numel() * next_tensor.element_size()
            
            # 如果两个张量大小相近，尝试合并
            if abs(current_size - next_size) < 1024:  # 1KB的阈值
                new_size = max(current_size, next_size)
                new_tensor = torch.empty(new_size // current.element_size(),
                                      dtype=current.dtype,
                                      device=current.device)
                pool[i] = new_tensor
                pool.pop(i + 1)
            else:
                i += 1
                
    def optimize_memory_allocation(self):
        """优化内存分配"""
        if not self.activation_sizes:
            self.estimate_activation_sizes()
            
        # 初始化内存池
        for size_category in self.memory_pools.keys():
            self.memory_pools[size_category]['tensors'] = []
            
        def get_tensor_from_pool(size: int) -> Optional[torch.Tensor]:
            """从内存池中获取张量"""
            tensor = self._allocate_from_pool(size)
            if tensor is not None:
                self.track_memory_usage("reuse_tensor", -size)
                return tensor
                
            self.track_memory_usage("new_tensor", size)
            return None
            
        def return_tensor_to_pool(tensor: torch.Tensor):
            """将张量返回到内存池"""
            size = tensor.numel() * tensor.element_size()
            self._return_to_pool(tensor)
            self.track_memory_usage("return_tensor", size)
            
            # 定期进行碎片整理
            if len(self.memory_events) % 100 == 0:  # 每100次操作进行一次碎片整理
                for pool_name in self.memory_pools.keys():
                    self._defragment_pool(pool_name)
                    
        # 注册钩子来管理内存
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size()
                tensor_id = f"tensor_{id(output)}"
                reuse_tensor = get_tensor_from_pool(size)
                if reuse_tensor is not None:
                    output.storage().copy_(reuse_tensor.storage())
                self.track_memory_usage("allocate", size, tensor_id)
                
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                for grad in grad_output:
                    if isinstance(grad, torch.Tensor):
                        tensor_id = f"tensor_{id(grad)}"
                        return_tensor_to_pool(grad)
                        self.track_memory_usage("free", -grad.numel() * grad.element_size(), tensor_id)
                        
        # 应用钩子
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存使用统计信息"""
        # 计算内存碎片统计
        total_fragments = 0
        fragment_sizes = []
        largest_fragment = 0
        
        for pool_info in self.memory_pools.values():
            pool_tensors = pool_info['tensors']
            if pool_tensors:
                total_fragments += len(pool_tensors)
                sizes = [t.numel() * t.element_size() for t in pool_tensors]
                fragment_sizes.extend(sizes)
                largest_fragment = max(largest_fragment, max(sizes))
                
        self.fragmentation_stats.update({
            'total_fragments': total_fragments,
            'fragment_sizes': fragment_sizes,
            'largest_fragment': largest_fragment
        })
        
        return {
            'peak_memory': self.peak_memory,
            'current_memory': self.current_memory,
            'memory_events': self.memory_events,
            'activation_sizes': self.activation_sizes,
            'fragmentation_stats': self.fragmentation_stats,
            'tensor_lifecycle': self.tensor_lifecycle,
            'pool_stats': {
                name: {
                    'total_tensors': len(pool['tensors']),
                    'total_size': sum(t.numel() * t.element_size() for t in pool['tensors'])
                }
                for name, pool in self.memory_pools.items()
            }
        }

class ComputationGraph(nn.Module):
    """动态计算图管理器"""
    
    def __init__(self):
        super().__init__()
        self.nodes: Dict[str, ComputeNode] = {}
        self.node_modules = nn.ModuleDict()
        self.execution_order: List[str] = []
        self.cached_results: Dict[str, torch.Tensor] = {}
        self.memory_plan: Optional[MemoryPlan] = None
        self.fusion_groups: List[List[str]] = []
        self.memory_optimizer: Optional[MemoryOptimizer] = None
        
    def add_node(
        self,
        name: str,
        operation: nn.Module,
        inputs: List[str],
        outputs: List[str],
        meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加计算节点"""
        if name in self.nodes:
            raise ValueError(f"节点 '{name}' 已存在")
            
        # 估算节点的内存和计算成本
        memory_size = sum(
            param.numel() * param.element_size()
            for param in operation.parameters()
        )
        compute_cost = memory_size * len(inputs)  # 简单估算
        
        self.nodes[name] = ComputeNode(
            name=name,
            operation=operation,
            inputs=inputs,
            outputs=outputs,
            meta=meta or {},
            memory_size=memory_size,
            compute_cost=compute_cost
        )
        self.node_modules[name] = operation
        self._update_execution_order()
        self._invalidate_optimizations()
        
    def _invalidate_optimizations(self) -> None:
        """使优化失效"""
        self.memory_plan = None
        self.fusion_groups = []
        
    def _build_graph(self) -> nx.DiGraph:
        """构建计算图的NetworkX表示"""
        G = nx.DiGraph()
        for node in self.nodes.values():
            G.add_node(
                node.name,
                memory_size=node.memory_size,
                compute_cost=node.compute_cost
            )
            for input_name in node.inputs:
                for other_node in self.nodes.values():
                    if input_name in other_node.outputs:
                        G.add_edge(other_node.name, node.name)
        return G
        
    def _identify_fusion_opportunities(self) -> List[List[str]]:
        """识别可以融合的算子组"""
        G = self._build_graph()
        fusion_groups = []
        
        # 识别垂直融合机会（链式操作）
        for node in G.nodes():
            if G.out_degree(node) == 1 and G.in_degree(list(G.successors(node))[0]) == 1:
                successor = list(G.successors(node))[0]
                fusion_groups.append(([node, successor], FusionType.VERTICAL))
                
        # 识别水平融合机会（并行操作）
        for node in G.nodes():
            parallel_ops = []
            for successor in G.successors(node):
                if G.in_degree(successor) == 1:
                    parallel_ops.append(successor)
            if len(parallel_ops) > 1:
                fusion_groups.append((parallel_ops, FusionType.HORIZONTAL))
                
        return fusion_groups
        
    def _optimize_memory(self) -> MemoryPlan:
        """优化内存使用"""
        G = self._build_graph()
        
        # 计算每个张量的生命周期
        tensor_lifecycle = {}
        for node in self.execution_order:
            step = self.execution_order.index(node)
            for output in self.nodes[node].outputs:
                if output not in tensor_lifecycle:
                    tensor_lifecycle[output] = {'birth': step, 'death': -1}
                    
            for input_name in self.nodes[node].inputs:
                if input_name in tensor_lifecycle and tensor_lifecycle[input_name]['death'] == -1:
                    tensor_lifecycle[input_name]['death'] = step
                    
        # 构建内存重用计划
        reuse_map = {}
        current_tensors = set()
        peak_memory = 0
        current_memory = 0
        release_points = {}
        
        for step, node_name in enumerate(self.execution_order):
            node = self.nodes[node_name]
            
            # 释放不再需要的张量
            for tensor in list(current_tensors):
                if tensor_lifecycle[tensor]['death'] <= step:
                    current_tensors.remove(tensor)
                    current_memory -= self._estimate_tensor_size(tensor)
                    release_points[tensor] = step
                    
            # 分配新张量
            for output in node.outputs:
                current_tensors.add(output)
                current_memory += self._estimate_tensor_size(output)
                
            peak_memory = max(peak_memory, current_memory)
            
            # 查找可以重用的内存
            for output in node.outputs:
                candidates = []
                for tensor in tensor_lifecycle:
                    if (tensor_lifecycle[tensor]['death'] < tensor_lifecycle[output]['birth'] and
                        self._estimate_tensor_size(tensor) >= self._estimate_tensor_size(output)):
                        candidates.append(tensor)
                if candidates:
                    reuse_map[output] = candidates
                    
        return MemoryPlan(
            allocation_size=peak_memory,
            reuse_map=reuse_map,
            peak_memory=peak_memory,
            release_points=release_points
        )
        
    def _estimate_tensor_size(self, tensor_name: str) -> int:
        """估算张量大小"""
        # TODO: 实现更精确的大小估算
        return 1024 * 1024  # 默认1MB
        
    def optimize(self) -> None:
        """优化计算图"""
        # 初始化内存优化器
        if self.memory_optimizer is None:
            self.memory_optimizer = MemoryOptimizer(self)
            
        # 设置梯度检查点
        self.memory_optimizer.setup_gradient_checkpointing()
        
        # 优化内存分配
        self.memory_optimizer.optimize_memory_allocation()
        
        # 识别融合机会
        fusion_groups = self._identify_fusion_opportunities()
        
        # 应用算子融合
        for group, fusion_type in fusion_groups:
            operations = [self.node_modules[name] for name in group]
            fused_op = FusedOperation(operations, fusion_type)
            
            # 更新图结构
            first_node = self.nodes[group[0]]
            last_node = self.nodes[group[-1]]
            fused_node = ComputeNode(
                name=f"fused_{'_'.join(group)}",
                operation=fused_op,
                inputs=first_node.inputs,
                outputs=last_node.outputs,
                meta={'fusion_type': fusion_type, 'original_nodes': group}
            )
            
            # 移除原始节点
            for name in group:
                del self.nodes[name]
                del self.node_modules[name]
                
            # 添加融合节点
            self.nodes[fused_node.name] = fused_node
            self.node_modules[fused_node.name] = fused_op
            
        # 优化内存使用
        self.memory_plan = self._optimize_memory()
        
        # 更新执行顺序
        self._update_execution_order()
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        self.cached_results = inputs.copy()
        
        # 如果没有优化过，先进行优化
        if not self.memory_plan:
            self.optimize()
            
        # 按执行顺序处理每个节点
        for node_name in self.execution_order:
            node = self.nodes[node_name]
            node_inputs = self._get_node_inputs(node)
            
            # 执行计算
            outputs = node.operation(**node_inputs)
            
            # 处理输出
            if isinstance(outputs, (tuple, list)):
                assert len(outputs) == len(node.outputs), \
                    f"节点 '{node_name}' 的输出数量不匹配"
                for output_name, output_tensor in zip(node.outputs, outputs):
                    self.cached_results[output_name] = output_tensor
            else:
                assert len(node.outputs) == 1, \
                    f"节点 '{node_name}' 的输出数量不匹配"
                self.cached_results[node.outputs[0]] = outputs
                
            # 应用内存优化
            if self.memory_plan:
                self._apply_memory_optimizations(node_name)
                
        # 返回所有输出
        return {
            name: tensor
            for name, tensor in self.cached_results.items()
            if any(name in node.outputs for node in self.nodes.values())
        }
        
    def _apply_memory_optimizations(self, current_node: str) -> None:
        """应用内存优化策略"""
        if not self.memory_plan:
            return
            
        # 释放不再需要的张量
        for tensor, release_step in self.memory_plan.release_points.items():
            if self.execution_order.index(current_node) == release_step:
                if tensor in self.cached_results:
                    del self.cached_results[tensor]
                    
        # 应用内存重用
        for output in self.nodes[current_node].outputs:
            if output in self.memory_plan.reuse_map:
                for reuse_tensor in self.memory_plan.reuse_map[output]:
                    if reuse_tensor in self.cached_results:
                        # 重用张量的存储空间
                        self.cached_results[output].storage().copy_(
                            self.cached_results[reuse_tensor].storage()
                        )
                        break

    def _update_execution_order(self) -> None:
        """更新节点执行顺序"""
        # 构建依赖图
        dependencies = defaultdict(set)
        reverse_dependencies = defaultdict(set)
        
        for node in self.nodes.values():
            for input_name in node.inputs:
                for other_node in self.nodes.values():
                    if input_name in other_node.outputs:
                        dependencies[node.name].add(other_node.name)
                        reverse_dependencies[other_node.name].add(node.name)
                        
        # 拓扑排序
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name: str) -> None:
            if name in temp_visited:
                raise ValueError("检测到循环依赖")
            if name in visited:
                return
                
            temp_visited.add(name)
            
            for dep in dependencies[name]:
                visit(dep)
                
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
            
        for name in self.nodes:
            if name not in visited:
                visit(name)
                
        self.execution_order = order
        
    def _get_node_inputs(self, node: ComputeNode) -> Dict[str, torch.Tensor]:
        """获取节点的输入张量"""
        inputs = {}
        for input_name in node.inputs:
            if input_name not in self.cached_results:
                raise RuntimeError(f"找不到输入 '{input_name}' 的值")
            inputs[input_name] = self.cached_results[input_name]
        return inputs
        
    def get_node_info(self, name: str) -> Dict[str, Any]:
        """获取节点信息"""
        if name not in self.nodes:
            raise ValueError(f"节点 '{name}' 不存在")
            
        node = self.nodes[name]
        return {
            'name': node.name,
            'operation': str(node.operation),
            'inputs': node.inputs,
            'outputs': node.outputs,
            'meta': node.meta,
            'dependencies': [
                other_name
                for other_name, other_node in self.nodes.items()
                if any(output in node.inputs for output in other_node.outputs)
            ]
        }
        
    def visualize(self) -> str:
        """生成计算图的DOT格式可视化"""
        dot = ['digraph G {']
        
        # 添加节点
        for name, node in self.nodes.items():
            dot.append(f'    "{name}" [label="{name}\\n{type(node.operation).__name__}"];')
            
        # 添加边
        for name, node in self.nodes.items():
            for input_name in node.inputs:
                for other_name, other_node in self.nodes.items():
                    if input_name in other_node.outputs:
                        dot.append(f'    "{other_name}" -> "{name}" [label="{input_name}"];')
                        
        dot.append('}')
        return '\n'.join(dot) 

class EnhancedComputationGraph(ComputationGraph):
    """增强型计算图管理器"""
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_manager = MemoryManager(self.device)
        self.execution_stats = []
        
    def _allocate_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        name: str
    ) -> torch.Tensor:
        """分配张量"""
        return self.memory_manager.allocate(shape, dtype, name)
        
    def _free_tensor(self, name: str) -> None:
        """释放张量"""
        self.memory_manager.free(name)
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        execution_start = time.time()
        self.cached_results = {}
        
        # 复制输入张量
        for name, tensor in inputs.items():
            cached_tensor = self._allocate_tensor(
                tensor.shape,
                tensor.dtype,
                f"input_{name}"
            )
            cached_tensor.copy_(tensor)
            self.cached_results[name] = cached_tensor
            
        # 按执行顺序处理每个节点
        for node_name in self.execution_order:
            node = self.nodes[node_name]
            node_start = time.time()
            
            # 准备输入
            node_inputs = {}
            for input_name in node.inputs:
                if input_name in self.cached_results:
                    node_inputs[input_name] = self.cached_results[input_name]
                    
            # 执行计算
            outputs = node.operation(**node_inputs)
            
            # 处理输出
            if isinstance(outputs, (tuple, list)):
                assert len(outputs) == len(node.outputs), \
                    f"节点 '{node_name}' 的输出数量不匹配"
                for output_name, output_tensor in zip(node.outputs, outputs):
                    cached_tensor = self._allocate_tensor(
                        output_tensor.shape,
                        output_tensor.dtype,
                        f"output_{output_name}"
                    )
                    cached_tensor.copy_(output_tensor)
                    self.cached_results[output_name] = cached_tensor
            else:
                assert len(node.outputs) == 1, \
                    f"节点 '{node_name}' 的输出数量不匹配"
                cached_tensor = self._allocate_tensor(
                    outputs.shape,
                    outputs.dtype,
                    f"output_{node.outputs[0]}"
                )
                cached_tensor.copy_(outputs)
                self.cached_results[node.outputs[0]] = cached_tensor
                
            # 释放不再需要的张量
            for input_name in node.inputs:
                if not any(input_name in n.inputs for n in self.nodes.values()):
                    self._free_tensor(f"input_{input_name}")
                    
            # 记录节点执行统计
            node_time = time.time() - node_start
            self.execution_stats.append({
                'node': node_name,
                'time': node_time,
                'memory': self.memory_manager.get_memory_stats()
            })
            
        # 准备输出
        outputs = {}
        for name, tensor in self.cached_results.items():
            if any(name in node.outputs for node in self.nodes.values()):
                outputs[name] = tensor.clone()
                self._free_tensor(f"output_{name}")
                
        # 记录总执行统计
        total_time = time.time() - execution_start
        self.execution_stats.append({
            'total_time': total_time,
            'final_memory': self.memory_manager.get_memory_stats()
        })
        
        return outputs
        
    def get_execution_stats(self) -> List[Dict[str, Any]]:
        """获取执行统计信息"""
        return self.execution_stats
        
    def clear_stats(self) -> None:
        """清空统计信息"""
        self.execution_stats.clear()
        self.memory_manager.tensor_pool.clear() 