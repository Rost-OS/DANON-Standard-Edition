"""
统一配置管理系统
实现了全面的配置管理、验证和优化功能
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Callable, Type, TypeVar
import torch
import json
from pathlib import Path
import yaml
import logging
from enum import Enum
from functools import wraps
import inspect
from typing import get_type_hints

T = TypeVar('T')

class ConfigValidationError(Exception):
    """配置验证错误"""
    pass

def validate_field(field_name: str, value: Any, constraints: Dict[str, Callable[[Any], bool]]) -> None:
    """验证单个字段"""
    for constraint_name, constraint_func in constraints.items():
        if not constraint_func(value):
            raise ConfigValidationError(f"Field '{field_name}' failed {constraint_name} validation")

def config_validator(*validators: Callable[[Any], bool], field_name: Optional[str] = None):
    """配置验证装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        original_post_init = getattr(cls, '__post_init__', None)
        
        def new_post_init(self):
            # 运行原始的 post_init
            if original_post_init:
                original_post_init(self)
            
            # 获取类型提示
            type_hints = get_type_hints(cls)
            
            # 验证所有字段
            for field_name, field_type in type_hints.items():
                if hasattr(self, field_name):
                    value = getattr(self, field_name)
                    
                    # 类型检查
                    if not isinstance(value, field_type):
                        raise ConfigValidationError(
                            f"Field '{field_name}' must be of type {field_type}, got {type(value)}"
                        )
                    
                    # 应用验证器
                    for validator in validators:
                        if not validator(value):
                            raise ConfigValidationError(
                                f"Field '{field_name}' failed validation with {validator.__name__}"
                            )
        
        cls.__post_init__ = new_post_init
        return cls
    
    return decorator

class ConfigValidator:
    """配置验证工具类"""
    @staticmethod
    def is_positive(value: Union[int, float]) -> bool:
        return value > 0
    
    @staticmethod
    def is_non_negative(value: Union[int, float]) -> bool:
        return value >= 0
    
    @staticmethod
    def is_probability(value: float) -> bool:
        return 0 <= value <= 1
    
    @staticmethod
    def is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0
    
    @staticmethod
    def has_min_length(min_length: int):
        def validator(value: Union[List, str]) -> bool:
            return len(value) >= min_length
        return validator
    
    @staticmethod
    def has_max_length(max_length: int):
        def validator(value: Union[List, str]) -> bool:
            return len(value) <= max_length
        return validator
    
    @staticmethod
    def is_in_range(min_val: Union[int, float], max_val: Union[int, float]):
        def validator(value: Union[int, float]) -> bool:
            return min_val <= value <= max_val
        return validator
    
    @staticmethod
    def is_one_of(*valid_values: Any):
        def validator(value: Any) -> bool:
            return value in valid_values
        return validator

class ConfigDocGenerator:
    """配置文档生成器"""
    @staticmethod
    def generate_markdown(config_class: Type) -> str:
        """生成配置类的 Markdown 文档"""
        doc = f"# {config_class.__name__}\n\n"
        
        if config_class.__doc__:
            doc += f"{config_class.__doc__.strip()}\n\n"
        
        doc += "## 配置参数\n\n"
        doc += "| 参数名 | 类型 | 默认值 | 验证规则 | 描述 |\n"
        doc += "|--------|------|---------|-----------|------|\n"
        
        for field_name, field_info in config_class.__dataclass_fields__.items():
            field_type = field_info.type
            default_value = field_info.default
            validators = []
            description = ""
            
            # 获取验证器信息
            if hasattr(field_info, "metadata") and "validators" in field_info.metadata:
                validators = [v.__name__ for v in field_info.metadata["validators"]]
            
            # 获取字段描述
            if field_info.metadata.get("doc"):
                description = field_info.metadata["doc"]
            elif config_class.__annotations__.get(field_name, None).__doc__:
                description = config_class.__annotations__[field_name].__doc__
            
            # 格式化验证器列表
            validator_str = ", ".join(validators) if validators else "无"
            
            doc += f"| {field_name} | {field_type.__name__} | {default_value} | {validator_str} | {description} |\n"
        
        # 添加配置示例
        doc += "\n## 配置示例\n\n"
        doc += "```python\n"
        doc += f"from danon.core.config import {config_class.__name__}\n\n"
        doc += f"config = {config_class.__name__}(\n"
        
        # 生成示例配置
        for field_name, field_info in config_class.__dataclass_fields__.items():
            default_value = field_info.default
            if isinstance(default_value, str):
                doc += f"    {field_name}='{default_value}',\n"
            else:
                doc += f"    {field_name}={default_value},\n"
        
        doc += ")\n```\n\n"
        
        # 添加验证规则说明
        doc += "## 验证规则说明\n\n"
        doc += "| 规则名 | 说明 |\n"
        doc += "|--------|------|\n"
        doc += "| is_positive | 值必须为正数 |\n"
        doc += "| is_non_negative | 值必须为非负数 |\n"
        doc += "| is_probability | 值必须在 [0, 1] 范围内 |\n"
        doc += "| is_power_of_two | 值必须为 2 的幂次方 |\n"
        doc += "| has_min_length | 长度必须大于等于指定值 |\n"
        doc += "| has_max_length | 长度必须小于等于指定值 |\n"
        doc += "| is_in_range | 值必须在指定范围内 |\n"
        doc += "| is_one_of | 值必须是指定选项之一 |\n"
        
        return doc
    
    @staticmethod
    def save_markdown(config_class: Type, output_path: Union[str, Path]) -> None:
        """保存配置文档到文件"""
        doc = ConfigDocGenerator.generate_markdown(config_class)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(doc)
    
    @staticmethod
    def generate_html(config_class: Type) -> str:
        """生成配置类的 HTML 文档"""
        markdown = ConfigDocGenerator.generate_markdown(config_class)
        try:
            import markdown2
            return markdown2.markdown(
                markdown,
                extras=['tables', 'code-blocks', 'header-ids']
            )
        except ImportError:
            logging.warning("markdown2 package not found. Please install it to generate HTML documentation.")
            return f"<pre>{markdown}</pre>"
    
    @staticmethod
    def save_html(config_class: Type, output_path: Union[str, Path]) -> None:
        """保存配置文档到 HTML 文件"""
        html = ConfigDocGenerator.generate_html(config_class)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{config_class.__name__} Configuration Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            padding: 2em;
            max-width: 1200px;
            margin: 0 auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
            """)

class OptimizationLevel(Enum):
    """优化级别"""
    NONE = "none"         # 不进行优化
    BASIC = "basic"       # 基础优化
    MODERATE = "moderate" # 中等优化
    AGGRESSIVE = "aggressive" # 激进优化

@dataclass
class BaseConfig:
    """基础配置类"""
    # 基础设置
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    seed: int = 42
    
    # 日志配置
    enable_logging: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_steps: int = 100
    
    # 性能监控
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_computation: bool = True
    profiling_steps: int = 100
    
    # 错误处理
    max_retries: int = 3
    error_tolerance: float = 1e-6
    enable_error_recovery: bool = True
    
    # 优化设置
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_memory_optimization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, torch.dtype):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """从字典创建配置"""
        # 处理特殊类型
        if 'dtype' in config_dict:
            config_dict['dtype'] = getattr(torch, config_dict['dtype'].split('.')[-1])
        if 'optimization_level' in config_dict:
            config_dict['optimization_level'] = OptimizationLevel(config_dict['optimization_level'])
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """保存配置到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in {'.yaml', '.yml'}:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseConfig":
        """从文件加载配置"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        if path.suffix == '.json':
            with open(path) as f:
                config_dict = json.load(f)
        elif path.suffix in {'.yaml', '.yml'}:
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        return cls.from_dict(config_dict)
        
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            assert self.log_steps > 0, "log_steps must be positive"
            assert self.profiling_steps > 0, "profiling_steps must be positive"
            assert self.max_retries >= 0, "max_retries must be non-negative"
            assert 0 <= self.error_tolerance <= 1, "error_tolerance must be in [0, 1]"
            
            if self.enable_mixed_precision and not torch.cuda.is_available():
                logging.warning("Mixed precision is enabled but CUDA is not available")
                
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            return False

@config_validator(
    ConfigValidator.is_positive,
    ConfigValidator.is_probability
)
@dataclass
class ModelConfig(BaseConfig):
    """模型配置类"""
    # 模型架构
    hidden_size: int = field(
        default=8192,  # 增大hidden_size以支持更大参数规模
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_power_of_two]}
    )
    intermediate_size: int = field(
        default=32768,  # 4倍hidden_size
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_power_of_two]}
    )
    num_layers: int = field(
        default=32,  # 增加层数
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    num_heads: int = field(
        default=64,  # 增加注意力头数
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    head_dim: int = field(
        default=128,  # 增加每个头的维度
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    dropout: float = field(
        default=0.1,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    activation: str = field(
        default="gelu",
        metadata={"validators": [ConfigValidator.is_one_of("gelu", "relu", "swish")]}
    )
    
    # 注意力机制
    attention_type: str = field(
        default="msra",
        metadata={"validators": [ConfigValidator.is_one_of("msra", "dala", "hybrid")]}
    )
    num_attention_levels: int = field(
        default=4,  # 增加注意力层次
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_in_range(1, 6)]}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    
    # 记忆系统
    memory_size: int = field(
        default=16384,  # 增大记忆大小
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_power_of_two]}
    )
    memory_dropout: float = field(
        default=0.1,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    enable_cache: bool = True
    cache_size: int = field(
        default=10000,  # 增大缓存大小
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    
    # 压缩设置
    compression_ratio: float = field(
        default=0.5,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    min_compression_ratio: float = field(
        default=0.1,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    max_compression_ratio: float = field(
        default=0.9,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    enable_adaptive_compression: bool = True
    
    # 性能优化
    chunk_size: int = field(
        default=4096,  # 增大块大小以处理更长序列
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_power_of_two]}
    )
    gradient_checkpointing: bool = True  # 启用梯度检查点
    mixed_precision: bool = True  # 启用混合精度训练
    optimizer_states_in_half_precision: bool = True  # 优化器状态使用半精度
    
    def __post_init__(self):
        """初始化后的验证"""
        super().__post_init__()
        
        # 验证head_dim和hidden_size的关系
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ConfigValidationError(
                f"hidden_size ({self.hidden_size}) must equal num_heads * head_dim ({self.num_heads * self.head_dim})"
            )
        
        # 验证intermediate_size
        if self.intermediate_size < self.hidden_size:
            raise ConfigValidationError(
                "intermediate_size must be larger than hidden_size"
            )
        
        # 验证缓存相关配置
        if self.enable_cache and self.cache_size <= 0:
            raise ConfigValidationError(
                "cache_size must be positive when cache is enabled"
            )
            
        # 验证chunk_size
        if self.chunk_size <= 0:
            raise ConfigValidationError(
                "chunk_size must be positive"
            )
            
        # 验证性能优化配置
        if self.mixed_precision and not torch.cuda.is_available():
            logging.warning("Mixed precision training requires CUDA, but CUDA is not available")
            self.mixed_precision = False
            
        # 计算理论参数量
        self.total_params = self._calculate_total_params()
        logging.info(f"Theoretical model size: {self.total_params / 1e9:.2f}B parameters")
        
    def _calculate_total_params(self) -> int:
        """计算模型理论参数量"""
        params = 0
        
        # 每层的参数量
        for _ in range(self.num_layers):
            # 自注意力层
            params += self.hidden_size * self.hidden_size * 4  # Q,K,V,O矩阵
            params += self.hidden_size * 2  # LayerNorm参数
            
            # 前馈网络
            params += self.hidden_size * self.intermediate_size  # 第一个线性层
            params += self.intermediate_size * self.hidden_size  # 第二个线性层
            params += self.hidden_size * 2  # LayerNorm参数
            
        # 嵌入层参数
        params += self.hidden_size * self.memory_size  # 词嵌入
        params += self.hidden_size  # 位置嵌入
        
        return params

@config_validator(
    ConfigValidator.is_positive,
    ConfigValidator.is_probability
)
@dataclass
class TrainingConfig(BaseConfig):
    """训练配置类"""
    # 训练参数
    batch_size: int = field(
        default=32,
        metadata={"validators": [ConfigValidator.is_positive, ConfigValidator.is_power_of_two]}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"validators": [ConfigValidator.is_non_negative]}
    )
    max_epochs: int = field(
        default=100,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"validators": [ConfigValidator.is_non_negative]}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    
    # 优化器设置
    optimizer: str = field(
        default="adamw",
        metadata={"validators": [ConfigValidator.is_one_of("adam", "adamw", "sgd", "adafactor")]}
    )
    scheduler: str = field(
        default="linear",
        metadata={"validators": [ConfigValidator.is_one_of("linear", "cosine", "constant")]}
    )
    scheduler_warmup_ratio: float = field(
        default=0.1,
        metadata={"validators": [ConfigValidator.is_probability]}
    )
    
    # 分布式训练
    distributed_training: bool = False
    world_size: int = field(
        default=1,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    local_rank: int = field(
        default=-1,
        metadata={"validators": [ConfigValidator.is_in_range(-1, 1000)]}
    )
    
    # 混合精度训练
    fp16: bool = False
    fp16_opt_level: str = field(
        default="O1",
        metadata={"validators": [ConfigValidator.is_one_of("O0", "O1", "O2", "O3")]}
    )
    
    # 检查点设置
    save_steps: int = field(
        default=1000,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    save_total_limit: int = field(
        default=5,
        metadata={"validators": [ConfigValidator.is_positive]}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"validators": [ConfigValidator.is_one_of("steps", "epoch")]}
    )
    
    def __post_init__(self):
        """初始化后的验证"""
        super().__post_init__()
        
        # 验证分布式训练配置
        if self.distributed_training:
            if self.world_size <= 1:
                raise ConfigValidationError(
                    "world_size must be greater than 1 for distributed training"
                )
            if self.local_rank < 0:
                raise ConfigValidationError(
                    "local_rank must be non-negative for distributed training"
                )
        
        # 验证混合精度训练配置
        if self.fp16 and not torch.cuda.is_available():
            raise ConfigValidationError(
                "CUDA is required for FP16 training"
            )
        
        # 验证学习率调度器配置
        if self.scheduler != "constant" and self.warmup_steps <= 0:
            raise ConfigValidationError(
                "warmup_steps must be positive when using non-constant scheduler"
            )
        
        # 验证梯度累积步数
        if self.gradient_accumulation_steps > self.batch_size:
            raise ConfigValidationError(
                "gradient_accumulation_steps should not be larger than batch_size"
            )

@dataclass
class BasePerformanceConfig:
    """基础性能配置"""
    # 基础性能配置
    chunk_size: int = 1024
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    # 基础缓存配置
    enable_cache: bool = True
    max_cache_size: int = 10000
    
    # 基础监控配置
    enable_monitoring: bool = True
    monitoring_window_size: int = 1000
    
    def validate_base_performance(self) -> List[str]:
        """验证基础性能配置"""
        errors = []
        
        if self.chunk_size <= 0:
            errors.append(f"chunk_size must be positive, got {self.chunk_size}")
            
        if self.layer_norm_eps <= 0:
            errors.append(f"layer_norm_eps must be positive, got {self.layer_norm_eps}")
            
        if self.max_cache_size <= 0:
            errors.append(f"max_cache_size must be positive, got {self.max_cache_size}")
            
        return errors

@dataclass
class PerformanceOptimizationMixin:
    """高级性能优化配置的Mixin类"""
    # 高级性能优化
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    optimize_memory_usage: bool = True
    enable_tensor_fusion: bool = True
    
    # 高级缓存配置
    cache_size: int = 1000
    cache_dtype: torch.dtype = torch.float16
    enable_cache_compression: bool = True
    
    # 分布式训练
    world_size: int = 1
    local_rank: int = 0
    distributed_backend: str = "nccl"
    
    def validate_performance_config(self) -> List[str]:
        """验证高级性能优化配置"""
        errors = []
        
        if self.cache_size <= 0:
            errors.append(f"cache_size must be positive, got {self.cache_size}")
            
        if self.world_size < 1:
            errors.append(f"world_size must be at least 1, got {self.world_size}")
            
        if self.local_rank < -1:
            errors.append(f"local_rank must be >= -1, got {self.local_rank}")
            
        return errors
    
    def create_optimization_config(self) -> Dict[str, Any]:
        """创建优化器配置"""
        return {
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_mixed_precision": self.use_mixed_precision,
            "optimize_memory_usage": self.optimize_memory_usage,
            "enable_tensor_fusion": self.enable_tensor_fusion
        }
    
    def optimize_for_hardware(self) -> None:
        """根据硬件优化配置"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # 根据 GPU 内存大小调整配置
            if gpu_mem < 8 * 1024 * 1024 * 1024:  # < 8GB
                self.use_gradient_checkpointing = True
                self.cache_size = 500
                self.enable_cache_compression = True
            elif gpu_mem < 16 * 1024 * 1024 * 1024:  # < 16GB
                self.use_mixed_precision = True
                self.optimize_memory_usage = True
            
            # 根据 GPU 计算能力调整配置
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] >= 7:  # Volta 或更新架构
                self.enable_tensor_fusion = True
                self.use_mixed_precision = True

@dataclass
class UnifiedAttentionConfig(BaseConfig, BasePerformanceConfig):
    """统一的注意力配置基类，包含所有共享配置项"""
    # 基础模型配置
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_head_size: int = 64
    dropout: float = 0.1
    
    def validate(self) -> List[str]:
        """验证配置参数的有效性"""
        errors = super().validate()  # 从BaseConfig继承的验证
        errors.extend(self.validate_base_performance())  # 基础性能验证
        
        if self.hidden_size <= 0:
            errors.append(f"hidden_size must be positive, got {self.hidden_size}")
        
        if self.num_attention_heads <= 0:
            errors.append(f"num_attention_heads must be positive, got {self.num_attention_heads}")
            
        if self.attention_head_size <= 0:
            errors.append(f"attention_head_size must be positive, got {self.attention_head_size}")
            
        if not 0 <= self.dropout <= 1:
            errors.append(f"dropout must be between 0 and 1, got {self.dropout}")
            
        return errors

@dataclass
class UnifiedConfig:
    """统一配置类，用于整合所有配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attention: Optional[UnifiedAttentionConfig] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.attention is None:
            # 根据模型配置选择合适的注意力配置
            if self.model.attention_type == "msra":
                self.attention = MSRAConfig()
            elif self.model.attention_type == "dala":
                self.attention = DALAConfig()
            elif self.model.attention_type == "infinite":
                self.attention = InfiniteAttentionConfig()
    
    def validate(self) -> bool:
        """验证所有配置"""
        try:
            # 验证子配置
            if not self.model.validate() or not self.training.validate():
                return False
            
            # 验证注意力配置
            if self.attention is not None:
                attention_errors = self.attention.validate()
                if attention_errors:
                    logging.error("Attention configuration validation failed:\n" + 
                                "\n".join(attention_errors))
                    return False
            
            # 验证配置间的依赖关系
            self._validate_dependencies()
            return True
        except ConfigValidationError as e:
            logging.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _validate_dependencies(self) -> None:
        """验证配置间的依赖关系"""
        # 验证模型和训练配置的兼容性
        if self.model.hidden_size % self.model.num_heads != 0:
            raise ConfigValidationError(
                "Model hidden_size must be divisible by num_heads"
            )
        
        # 验证分布式训练配置
        if self.training.distributed_training:
            if self.training.world_size <= 1:
                raise ConfigValidationError(
                    "world_size must be greater than 1 for distributed training"
                )
            if self.training.local_rank < 0:
                raise ConfigValidationError(
                    "local_rank must be non-negative for distributed training"
                )
        
        # 验证混合精度训练配置
        if self.training.fp16:
            if not torch.cuda.is_available():
                raise ConfigValidationError(
                    "CUDA is required for FP16 training"
                )
            if self.model.dtype != torch.float32:
                raise ConfigValidationError(
                    "Model dtype must be float32 when using FP16 training"
                )
        
        # 验证注意力配置与模型配置的兼容性
        if self.attention is not None:
            if self.attention.hidden_size != self.model.hidden_size:
                raise ConfigValidationError(
                    "Attention hidden_size must match model hidden_size"
                )
            if self.attention.num_attention_heads != self.model.num_heads:
                raise ConfigValidationError(
                    "Attention num_heads must match model num_heads"
                )
    
    def generate_docs(self, output_path: Union[str, Path]) -> None:
        """生成配置文档"""
        ConfigDocGenerator.save_markdown(self.__class__, output_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        config_dict = {
            "model": self.model.to_dict(),
            "training": self.training.to_dict()
        }
        if self.attention is not None:
            config_dict["attention"] = self.attention.to_dict()
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "UnifiedConfig":
        """从字典创建配置"""
        model_config = ModelConfig.from_dict(config_dict["model"])
        training_config = TrainingConfig.from_dict(config_dict["training"])
        
        attention_config = None
        if "attention" in config_dict:
            attention_type = model_config.attention_type
            if attention_type == "msra":
                attention_config = MSRAConfig.from_dict(config_dict["attention"])
            elif attention_type == "dala":
                attention_config = DALAConfig.from_dict(config_dict["attention"])
            elif attention_type == "infinite":
                attention_config = InfiniteAttentionConfig.from_dict(config_dict["attention"])
        
        return cls(
            model=model_config,
            training=training_config,
            attention=attention_config
        )

@dataclass
class MSRAConfig(UnifiedAttentionConfig):
    """MSRA特定配置"""
    # MSRA特有配置
    num_layers: int = 6
    compression_factor: int = 4
    calibration_factor: float = 0.1
    bidirectional_flow: bool = True
    feature_fusion: bool = True
    stability_threshold: float = 0.1
    auto_calibration: bool = True
    theoretical_bounds: bool = True
    
    # MSRA特有的性能配置
    feature_compression_ratio: float = 0.5
    stability_check_frequency: int = 100
    auto_adjust_compression: bool = True
    
    def validate(self) -> List[str]:
        errors = super().validate()
        errors.extend(self.validate_performance_config())
        
        if self.num_layers <= 0:
            errors.append(f"num_layers must be positive, got {self.num_layers}")
            
        if self.compression_factor <= 0:
            errors.append(f"compression_factor must be positive, got {self.compression_factor}")
            
        if not 0 <= self.calibration_factor <= 1:
            errors.append(f"calibration_factor must be between 0 and 1, got {self.calibration_factor}")
            
        if not 0 <= self.stability_threshold <= 1:
            errors.append(f"stability_threshold must be between 0 and 1, got {self.stability_threshold}")
            
        if not 0 <= self.feature_compression_ratio <= 1:
            errors.append(f"feature_compression_ratio must be between 0 and 1, got {self.feature_compression_ratio}")
            
        if self.stability_check_frequency <= 0:
            errors.append(f"stability_check_frequency must be positive, got {self.stability_check_frequency}")
            
        return errors
    
    def create_attention_config(self) -> Dict[str, Any]:
        """创建注意力模块配置"""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "bidirectional_flow": self.bidirectional_flow,
            "feature_fusion": self.feature_fusion
        }
    
    def create_compression_config(self) -> Dict[str, Any]:
        """创建压缩模块配置"""
        return {
            "compression_factor": self.compression_factor,
            "feature_compression_ratio": self.feature_compression_ratio,
            "auto_adjust_compression": self.auto_adjust_compression
        }
    
    def create_stability_config(self) -> Dict[str, Any]:
        """创建稳定性配置"""
        return {
            "stability_threshold": self.stability_threshold,
            "auto_calibration": self.auto_calibration,
            "theoretical_bounds": self.theoretical_bounds,
            "stability_check_frequency": self.stability_check_frequency
        }

@dataclass
class DALAConfig(UnifiedAttentionConfig):
    """DALA (Dynamic Adaptive Long-range Attention) 特定配置"""
    # DALA 特有配置
    max_sequence_length: int = 1000000
    num_heads: int = 8
    compression_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    attention_dropout: float = 0.1
    router_dropout: float = 0.1
    use_adaptive_router: bool = True
    use_sparse_attention: bool = True
    use_recursive_state: bool = True
    state_decay_rate: float = 0.9
    
    # 混合模式配置
    auto_switch_threshold: int = 10000
    enable_hybrid_mode: bool = True
    
    def validate(self) -> List[str]:
        errors = super().validate()
        errors.extend(self.validate_performance_config())
        
        # DALA配置验证
        if self.max_sequence_length <= 0:
            errors.append(f"max_sequence_length must be positive, got {self.max_sequence_length}")
            
        if not all(r > 1 for r in self.compression_rates):
            errors.append("all compression_rates must be greater than 1")
            
        if not 0 <= self.attention_dropout <= 1:
            errors.append(f"attention_dropout must be between 0 and 1, got {self.attention_dropout}")
            
        if not 0 <= self.router_dropout <= 1:
            errors.append(f"router_dropout must be between 0 and 1, got {self.router_dropout}")
            
        if not 0 <= self.state_decay_rate <= 1:
            errors.append(f"state_decay_rate must be between 0 and 1, got {self.state_decay_rate}")
            
        # 混合模式验证
        if self.auto_switch_threshold <= 0:
            errors.append(f"auto_switch_threshold must be positive, got {self.auto_switch_threshold}")
            
        return errors
    
    def create_attention_config(self) -> Dict[str, Any]:
        """创建注意力模块配置"""
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout
        }
    
    def create_compression_config(self) -> Dict[str, Any]:
        """创建压缩模块配置"""
        return {
            "compression_rates": self.compression_rates,
            "cache_size": self.cache_size,
            "enable_compression": self.enable_cache_compression
        }
    
    def create_state_config(self) -> Dict[str, Any]:
        """创建状态管理配置"""
        return {
            "use_recursive_state": self.use_recursive_state,
            "state_decay_rate": self.state_decay_rate
        }

@dataclass
class InfiniteAttentionConfig(UnifiedAttentionConfig):
    """无限注意力特定配置"""
    # 稀疏注意力配置
    sparsity_factor: float = 0.1
    top_k_ratio: float = 0.2
    
    # 局部注意力配置
    local_window_size: int = 512
    sliding_window_size: int = 256
    
    # 压缩配置
    compression_ratio: float = 0.5
    compression_rates: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    enable_adaptive_compression: bool = True
    
    # 层次注意力
    num_levels: int = 3
    level_dropout: float = 0.1
    
    def validate(self) -> List[str]:
        errors = super().validate()
        errors.extend(self.validate_performance_config())
        
        if not 0 <= self.sparsity_factor <= 1:
            errors.append(f"sparsity_factor must be between 0 and 1, got {self.sparsity_factor}")
            
        if not 0 <= self.top_k_ratio <= 1:
            errors.append(f"top_k_ratio must be between 0 and 1, got {self.top_k_ratio}")
            
        if self.local_window_size <= 0:
            errors.append(f"local_window_size must be positive, got {self.local_window_size}")
            
        if self.sliding_window_size <= 0:
            errors.append(f"sliding_window_size must be positive, got {self.sliding_window_size}")
            
        if not 0 <= self.compression_ratio <= 1:
            errors.append(f"compression_ratio must be between 0 and 1, got {self.compression_ratio}")
            
        if not all(r > 1 for r in self.compression_rates):
            errors.append("all compression_rates must be greater than 1")
            
        if self.num_levels <= 0:
            errors.append(f"num_levels must be positive, got {self.num_levels}")
            
        if not 0 <= self.level_dropout <= 1:
            errors.append(f"level_dropout must be between 0 and 1, got {self.level_dropout}")
            
        return errors
    
    def create_attention_config(self) -> Dict[str, Any]:
        """创建注意力模块配置"""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "sparsity_factor": self.sparsity_factor,
            "top_k_ratio": self.top_k_ratio
        }
    
    def create_window_config(self) -> Dict[str, Any]:
        """创建窗口配置"""
        return {
            "local_window_size": self.local_window_size,
            "sliding_window_size": self.sliding_window_size
        }
    
    def create_compression_config(self) -> Dict[str, Any]:
        """创建压缩配置"""
        return {
            "compression_ratio": self.compression_ratio,
            "compression_rates": self.compression_rates,
            "enable_adaptive_compression": self.enable_adaptive_compression
        }

class ConfigFactory:
    """配置工厂类，用于创建和验证各种配置"""
    
    @staticmethod
    def create_config(config_type: str, **kwargs) -> UnifiedAttentionConfig:
        """创建指定类型的配置实例
        
        Args:
            config_type: 配置类型，可选值：["msra", "dala", "infinite"]
            **kwargs: 配置参数
            
        Returns:
            UnifiedAttentionConfig: 配置实例
            
        Raises:
            ValueError: 如果配置类型无效或配置验证失败
        """
        config_map = {
            "msra": MSRAConfig,
            "dala": DALAConfig,
            "infinite": InfiniteAttentionConfig
        }
        
        if config_type not in config_map:
            raise ValueError(f"Invalid config type: {config_type}")
            
        config_class = config_map[config_type]
        config = config_class(**kwargs)
        
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
            
        return config 