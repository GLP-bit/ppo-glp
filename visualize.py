import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import List, Dict, Any, Optional
from matplotlib.figure import Figure


class Visualizer:
    """可视化工具类"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 设置默认样式
        self.set_style()

        # 颜色方案
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'success': '#4CAF50',
            'warning': '#FFC107',
            'error': '#F44336',
            'light': '#90CAF9',
            'dark': '#1976D2'
        }

    def set_style(self):
        """设置matplotlib的绘图样式"""
        # 使用内置的样式
        plt.style.use('default')

        # 设置seaborn的样式
        sns.set_theme(style="whitegrid")

        # 设置图表的默认大小和字体大小
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    def plot_training_progress(self, rewards: List[float],
                               avg_rewards: Optional[List[float]] = None,
                               window_size: int = 100) -> None:
        """绘制训练进度图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制原始奖励
        ax.plot(rewards, alpha=0.3, color=self.colors['primary'], label='Episode Reward')

        # 绘制移动平均
        if avg_rewards:
            ax.plot(avg_rewards, color=self.colors['secondary'],
                    label=f'{window_size}-Episode Average')

        ax.set_title('Training Progress', pad=20)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, 'training_progress.png')

    def plot_decision_distribution(self, local_ratio: float, edge_ratio: float,
                                   preprocess_ratio: float) -> None:
        """绘制决策分布饼图"""
        fig, ax = plt.subplots(figsize=(10, 10))

        labels = ['Local Processing', 'Edge Processing', 'Preprocess + Edge']
        sizes = [local_ratio, edge_ratio, preprocess_ratio]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        # 设置文本属性
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12)

        ax.set_title('Offloading Decision Distribution', pad=20)

        self._save_figure(fig, 'decision_distribution.png')

    def plot_performance_metrics(self, delays: List[float], energies: List[float],
                                 priorities: List[int]) -> None:
        """绘制性能指标图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 延迟分布
        sns.histplot(delays, kde=True, ax=axes[0], color=self.colors['primary'])
        axes[0].set_title('Delay Distribution')
        axes[0].set_xlabel('Delay (s)')
        axes[0].set_ylabel('Frequency')

        # 能耗分布
        sns.histplot(energies, kde=True, ax=axes[1], color=self.colors['secondary'])
        axes[1].set_title('Energy Consumption Distribution')
        axes[1].set_xlabel('Energy (J)')
        axes[1].set_ylabel('Frequency')

        # 任务优先级分布
        if priorities:
            sns.histplot(priorities, kde=False, discrete=True, ax=axes[2],
                         color=self.colors['success'])
            axes[2].set_title('Task Priority Distribution')
            axes[2].set_xlabel('Priority Level')
            axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        self._save_figure(fig, 'performance_metrics.png')

    def plot_comparison(self, data_list: List[List[float]], labels: List[str],
                        metric_name: str) -> None:
        """比较不同方法的性能"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 转换数据为DataFrame格式
        df = pd.DataFrame()
        for data, label in zip(data_list, labels):
            df[label] = pd.Series(data)

        # 箱线图比较
        sns.boxplot(data=df, ax=ax1)
        ax1.set_title(f'{metric_name} Distribution')
        ax1.set_ylabel(metric_name)

        # 小提琴图比较
        sns.violinplot(data=df, ax=ax2)
        ax2.set_title(f'{metric_name} Distribution (Violin Plot)')
        ax2.set_ylabel(metric_name)

        plt.tight_layout()
        self._save_figure(fig, f'{metric_name.lower()}_comparison.png'.replace(' ', '_'))

    def _save_figure(self, fig: Figure, filename: str) -> None:
        """保存图表"""
        try:
            filepath = os.path.join(self.save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving figure {filename}: {str(e)}")

    def save_training_info(self, config_dict: Dict[str, Any],
                           training_params: Dict[str, Any],
                           performance_metrics: Dict[str, Any]) -> None:
        """保存训练配置和性能指标"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        info_file = os.path.join(self.save_dir, f'training_info_{timestamp}.txt')

        with open(info_file, 'w') as f:
            # 写入配置信息
            f.write("=== Training Configuration ===\n")
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")

            # 写入训练参数
            f.write("\n=== Training Parameters ===\n")
            for key, value in training_params.items():
                f.write(f"{key}: {value}\n")

            # 写入性能指标
            f.write("\n=== Performance Metrics ===\n")
            for key, value in performance_metrics.items():
                f.write(f"{key}: {value}\n")