import argparse
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class SystemParameters:
    """系统参数配置"""
    # 系统规模参数
    num_ambulances: int = 5

    # 任务相关参数
    task_size_range: tuple = (0.1, 2.0)  # MB
    task_cpu_cycles_range: tuple = (100, 1000)  # Million cycles
    task_priority_range: tuple = (1, 5)  # 任务优先级

    # 通信参数
    bandwidth: float = 20.0  # MHz
    transmission_power: float = 0.1  # W
    noise_power: float = 1e-13  # W
    path_loss_exponent: int = 4

    # 计算参数
    local_cpu_freq: float = 2.0  # GHz
    edge_cpu_freq: float = 3.0  # GHz
    local_cpu_power: float = 0.9  # W
    edge_cpu_power: float = 1.5  # W

    # 边缘智能预处理参数
    preprocessing_ratio: float = 0.6  # 预处理后数据量减少比例
    preprocessing_cycles: int = 50  # Million cycles

    # 权重参数
    delay_weight: float = 0.7
    energy_weight: float = 0.3

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建实例"""
        params = cls()
        for key, value in vars(args).items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params

    def save(self, path):
        """保存参数到文件"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        """从文件加载参数"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingParameters:
    """训练参数配置"""
    # PPO超参数
    clip_epsilon: float = 0.2
    critic_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 10

    # 训练设置
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    eval_episodes: int = 10
    save_interval: int = 100
    eval_interval: int = 50

    # 缓存设置
    max_buffer_size: int = 1000

    # 早停设置
    early_stop_patience: int = 50
    early_stop_min_improvement: float = 0.01

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建实例"""
        params = cls()
        for key, value in vars(args).items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params

    def save(self, path):
        """保存参数到文件"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        """从文件加载参数"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def add_system_args(parser):
    """添加系统参数到参数解析器"""
    group = parser.add_argument_group('System Parameters')
    group.add_argument('--num_ambulances', type=int, default=5)
    group.add_argument('--bandwidth', type=float, default=20.0)
    group.add_argument('--local_cpu_freq', type=float, default=2.0)
    group.add_argument('--edge_cpu_freq', type=float, default=3.0)
    group.add_argument('--delay_weight', type=float, default=0.7)
    group.add_argument('--energy_weight', type=float, default=0.3)


def add_training_args(parser):
    """添加训练参数到参数解析器"""
    group = parser.add_argument_group('Training Parameters')
    group.add_argument('--num_episodes', type=int, default=1000)
    group.add_argument('--batch_size', type=int, default=32)
    group.add_argument('--learning_rate', type=float, default=3e-4)
    group.add_argument('--eval_episodes', type=int, default=10)
    group.add_argument('--save_interval', type=int, default=100)
    group.add_argument('--early_stop_patience', type=int, default=50)


def get_default_parser():
    """获取默认的参数解析器"""
    parser = argparse.ArgumentParser(description='Edge Computing Offloading with PPO')

    # 基本参数
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'baseline'])
    parser.add_argument('--baseline_type', type=str, default='greedy',
                        choices=['greedy', 'round_robin', 'random', 'priority', 'adaptive'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str, default='default')

    # 添加系统和训练参数
    add_system_args(parser)
    add_training_args(parser)

    return parser