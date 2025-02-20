from setuptools import setup, find_packages
import os
import sys

# 检查Python版本
if sys.version_info < (3, 8):
    sys.exit('DANON 需要 Python 3.8 或更高版本')

# 读取版本信息
VERSION = '1.0.0'

# 读取README文件
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 核心依赖
core_requirements = [
    "torch>=1.8.0",  # PyTorch深度学习框架
    "numpy>=1.19.0,<2.0.0",  # 数值计算库
    "pandas>=1.2.0,<2.0.0",  # 数据处理库
    "scikit-learn>=0.24.0,<2.0.0",  # 机器学习工具库
    "accelerate>=0.20.0",  # 大规模模型训练加速
    "deepspeed>=0.9.0",  # 分布式训练优化
    "bitsandbytes>=0.39.0",  # 8-bit优化
    "torch-xla>=1.12",  # TPU支持
    "tqdm>=4.60.0,<5.0.0",  # 进度条
]

# 可视化依赖
visualization_requirements = [
    "matplotlib>=3.3.0,<4.0.0",  # 绘图库
    "seaborn>=0.11.0,<1.0.0",  # 统计数据可视化
    "tensorboard>=2.4.0",  # 训练可视化
    "wandb>=0.12.0",  # 实验跟踪与可视化
]

# 性能监控依赖
monitoring_requirements = [
    "psutil>=5.8.0,<6.0.0",  # 系统资源监控
    "gputil>=1.4.0,<2.0.0",  # GPU监控
    "pytorch-memlab>=0.2.0",  # PyTorch内存分析
    "torch-tb-profiler>=0.4.0",  # PyTorch性能分析
]

# 测试依赖
test_requirements = [
    "pytest>=6.0.0,<7.0.0",
    "pytest-cov>=2.10.0,<3.0.0",
    "pytest-benchmark>=3.4.0,<4.0.0",
    "pytest-xdist>=2.3.0,<3.0.0",  # 并行测试
    "pytest-timeout>=2.0.0,<3.0.0",  # 测试超时控制
    "pytest-randomly>=3.8.0,<4.0.0",  # 随机化测试顺序
    "pytest-mock>=3.6.0,<4.0.0",  # 模拟测试
    "pytest-sugar>=0.9.4,<1.0.0",  # 测试界面美化
    "pytest-html>=3.1.0,<4.0.0",  # HTML测试报告
    "pytest-rerunfailures>=10.1,<11.0.0",  # 失败重试
]

# 代码质量依赖
quality_requirements = [
    "black>=21.5b2",  # 代码格式化
    "isort>=5.8.0,<6.0.0",  # import排序
    "flake8>=3.9.0,<4.0.0",  # 代码检查
    "mypy>=0.910,<1.0.0",  # 类型检查
    "pre-commit>=2.15.0,<3.0.0",  # Git提交钩子
]

# 文档依赖
docs_requirements = [
    "sphinx>=4.0.0,<5.0.0",  # 文档生成
    "sphinx-rtd-theme>=0.5.0,<1.0.0",  # Read the Docs主题
    "sphinx-autodoc-typehints>=1.12.0,<2.0.0",  # 类型提示文档
    "nbsphinx>=0.8.0,<1.0.0",  # Jupyter Notebook支持
    "jupyter>=1.0.0,<2.0.0",  # Jupyter环境
]

# 分布式计算依赖
distributed_requirements = [
    "horovod>=0.22.0,<1.0.0",  # 分布式深度学习
    "mpi4py>=3.0.0,<4.0.0",  # MPI并行计算
    "ray>=1.4.0,<2.0.0",  # 分布式计算框架
    "fairscale>=0.4.12",  # 分布式训练工具
    "torch-distributed>=1.0",  # PyTorch分布式
]

# 性能分析依赖
profiling_requirements = [
    "memory_profiler>=0.58.0,<1.0.0",  # 内存分析
    "line_profiler>=3.3.0,<4.0.0",  # 逐行性能分析
    "py-spy>=0.3.0,<1.0.0",  # Python性能采样
    "scalene>=1.3.0,<2.0.0",  # CPU/GPU性能分析
    "torch-optimizer>=0.3.0",  # 优化器集合
    "apex>=0.1",  # NVIDIA APEX优化
    "pytorch-memlab>=0.2.0",  # PyTorch内存分析
]

# 大规模模型训练依赖
large_model_requirements = [
    "transformers>=4.20.0",  # Hugging Face Transformers
    "datasets>=2.0.0",  # 数据集工具
    "tokenizers>=0.12.0",  # 分词器
    "optimum>=1.5.0",  # 模型优化工具
    "peft>=0.2.0",  # 参数高效微调
]

setup(
    name="danon",
    version=VERSION,
    description="动态自适应神经算子网络 (Dynamic Adaptive Neural Operator Network)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WaZi",
    author_email="rostos@163.com",
    url="https://github.com/Rost-OS/DANON",
    license="MIT",
    keywords=[
        "deep-learning",
        "neural-networks",
        "artificial-intelligence",
        "machine-learning",
        "dynamic-networks",
        "adaptive-computation",
        "operator-networks",
        "large-scale-models",  # 新增关键词
        "distributed-training",  # 新增关键词
        "memory-efficient"  # 新增关键词
    ],
    
    # 包配置
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    package_data={
        "danon": [
            "configs/*.yaml",
            "configs/*.json",
            "examples/*.ipynb",
            "examples/*.py",
            "core/templates/*.jinja2",
            "utils/data/*.pkl",
        ]
    },
    
    # 依赖配置
    python_requires=">=3.8",
    install_requires=core_requirements + visualization_requirements + monitoring_requirements + large_model_requirements,
    extras_require={
        "dev": test_requirements + quality_requirements + docs_requirements,
        "test": test_requirements,
        "docs": docs_requirements,
        "distributed": distributed_requirements,
        "profiling": profiling_requirements,
        "all": (test_requirements + quality_requirements + docs_requirements + 
                distributed_requirements + profiling_requirements + large_model_requirements)
    },
    
    # 命令行工具
    entry_points={
        "console_scripts": [
            "danon-train=danon.cli.train:main",
            "danon-eval=danon.cli.evaluate:main",
            "danon-profile=danon.cli.profile:main",
            "danon-benchmark=danon.cli.benchmark:main",
            "danon-visualize=danon.cli.visualize:main",
            "danon-export=danon.cli.export:main",
            "danon-serve=danon.cli.serve:main",
            "danon-monitor=danon.cli.monitor:main",
            "danon-distributed=danon.cli.distributed:main"  # 新增分布式训练命令
        ]
    },
    
    # 项目元数据
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed"
    ],
    
    # 项目配置
    zip_safe=False,
    include_package_data=True,
    platforms=["any"],
    
    # 项目URL
    project_urls={
        "Bug Tracker": "https://github.com/Rost-OS/DANON/issues",
        "Documentation": "https://danon.readthedocs.io/",
        "Source Code": "https://github.com/Rost-OS/DANON",
    }
) 