import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os


class ResultAnalyzer:
    """结果分析类，用于分析和可视化实验结果"""

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_data = None
        self.performance_metrics = None

    def load_results(self):
        """加载实验结果"""
        results = []

        # 遍历所有实验目录
        for exp_dir in self.results_dir.glob('*'):
            if exp_dir.is_dir():
                result_file = exp_dir / 'experiment_results.json'
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        results.append({
                            'experiment_id': exp_dir.name,
                            **data['parameters'],
                            **data['statistics']
                        })

        self.results_data = pd.DataFrame(results)
        return self.results_data

    def calculate_performance_metrics(self):
        """计算性能指标"""
        if self.results_data is None:
            self.load_results()

        self.performance_metrics = {
            'delay': {
                'mean': self.results_data['mean_delay'].mean(),
                'std': self.results_data['mean_delay'].std(),
                'min': self.results_data['mean_delay'].min(),
                'max': self.results_data['mean_delay'].max()
            },
            'energy': {
                'mean': self.results_data['mean_energy'].mean(),
                'std': self.results_data['mean_energy'].std(),
                'min': self.results_data['mean_energy'].min(),
                'max': self.results_data['mean_energy'].max()
            },
            'reward': {
                'mean': self.results_data['mean_reward'].mean(),
                'std': self.results_data['mean_reward'].std(),
                'min': self.results_data['mean_reward'].min(),
                'max': self.results_data['mean_reward'].max()
            }
        }

        return self.performance_metrics

    def analyze_offloading_patterns(self):
        """分析卸载决策模式"""
        if self.results_data is None:
            self.load_results()

        patterns = {
            'local_processing': self.results_data['local_ratio'].mean(),
            'edge_processing': self.results_data['edge_ratio'].mean(),
            'preprocess_edge': self.results_data['preprocess_ratio'].mean()
        }

        # 创建饼图
        plt.figure(figsize=(10, 8))
        plt.pie(patterns.values(), labels=patterns.keys(), autopct='%1.1f%%')
        plt.title('Overall Offloading Decision Distribution')
        plt.savefig(self.results_dir / 'offloading_patterns.png')
        plt.close()

        return patterns

    def analyze_priority_impact(self):
        """分析任务优先级的影响"""
        priority_data = []

        # 遍历所有实验目录
        for exp_dir in self.results_dir.glob('*'):
            if exp_dir.is_dir():
                priority_file = exp_dir / 'priority_distribution.csv'
                if priority_file.exists():
                    df = pd.read_csv(priority_file)
                    priority_data.append(df)

        if priority_data:
            combined_data = pd.concat(priority_data)

            # 创建热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(combined_data.groupby('priority').mean(), annot=True, cmap='YlOrRd')
            plt.title('Task Priority Impact on Offloading Decisions')
            plt.savefig(self.results_dir / 'priority_impact.png')
            plt.close()

            return combined_data.groupby('priority').mean()

    def analyze_convergence(self):
        """分析算法收敛性"""
        convergence_data = []

        # 遍历所有实验目录
        for exp_dir in self.results_dir.glob('*'):
            if exp_dir.is_dir():
                metrics_file = exp_dir / 'training_metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        convergence_data.append(pd.Series(data['episode_rewards']))

        if convergence_data:
            # 计算平均收敛曲线
            mean_curve = pd.concat(convergence_data, axis=1).mean(axis=1)
            std_curve = pd.concat(convergence_data, axis=1).std(axis=1)

            # 绘制收敛曲线
            plt.figure(figsize=(12, 6))
            plt.plot(mean_curve.index, mean_curve.values, label='Mean Reward')
            plt.fill_between(
                mean_curve.index,
                mean_curve.values - std_curve.values,
                mean_curve.values + std_curve.values,
                alpha=0.2
            )
            plt.title('Algorithm Convergence Analysis')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.results_dir / 'convergence_analysis.png')
            plt.close()

            return {
                'convergence_rate': self._calculate_convergence_rate(mean_curve),
                'final_performance': mean_curve.iloc[-100:].mean(),
                'stability': std_curve.iloc[-100:].mean()
            }

    def _calculate_convergence_rate(self, reward_curve):
        """计算收敛速率"""
        max_reward = reward_curve.max()
        threshold = 0.95 * max_reward

        for episode, reward in reward_curve.items():
            if reward >= threshold:
                return episode

        return len(reward_curve)

    def statistical_analysis(self):
        """进行统计分析"""
        if self.results_data is None:
            self.load_results()

        analysis_results = {}

        # 计算主要指标的相关性
        correlation = self.results_data[[
            'mean_reward', 'mean_delay', 'mean_energy',
            'local_ratio', 'edge_ratio', 'preprocess_ratio'
        ]].corr()

        # 生成相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Metric Correlation Analysis')
        plt.savefig(self.results_dir / 'correlation_analysis.png')
        plt.close()

        analysis_results['correlation'] = correlation.to_dict()

        # 进行假设检验
        baseline_comparison = {}
        for metric in ['mean_delay', 'mean_energy', 'mean_reward']:
            if 'baseline_' + metric in self.results_data.columns:
                t_stat, p_value = stats.ttest_ind(
                    self.results_data[metric],
                    self.results_data['baseline_' + metric]
                )
                baseline_comparison[metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value
                }

        analysis_results['baseline_comparison'] = baseline_comparison

        return analysis_results

    def generate_report(self, output_path):
        """生成分析报告"""
        # 确保输出目录存在
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有分析结果
        results = {
            'performance_metrics': self.calculate_performance_metrics(),
            'offloading_patterns': self.analyze_offloading_patterns(),
            'priority_impact': self.analyze_priority_impact(),
            'convergence_analysis': self.analyze_convergence(),
            'statistical_analysis': self.statistical_analysis()
        }

        # 生成报告
        report = []
        report.append("# Edge Computing Offloading Analysis Report")
        report.append("\n## 1. Performance Metrics")
        report.append(self._format_performance_metrics(results['performance_metrics']))

        report.append("\n## 2. Offloading Patterns")
        report.append(self._format_offloading_patterns(results['offloading_patterns']))

        report.append("\n## 3. Priority Impact Analysis")
        report.append("See priority_impact.png for detailed visualization")

        report.append("\n## 4. Convergence Analysis")
        report.append(self._format_convergence_analysis(results['convergence_analysis']))

        report.append("\n## 5. Statistical Analysis")
        report.append(self._format_statistical_analysis(results['statistical_analysis']))

        # 保存报告
        with open(output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))

        print(f"Analysis report has been generated at {output_dir / 'analysis_report.md'}")

    def _format_performance_metrics(self, metrics):
        """格式化性能指标"""
        lines = []
        for metric, values in metrics.items():
            lines.append(f"\n### {metric.capitalize()}")
            for stat, value in values.items():
                lines.append(f"- {stat}: {value:.4f}")
        return '\n'.join(lines)

    def _format_offloading_patterns(self, patterns):
        """格式化卸载模式"""
        lines = ["\nOffloading decision distribution:"]
        for decision, ratio in patterns.items():
            lines.append(f"- {decision}: {ratio:.2%}")
        return '\n'.join(lines)

    def _format_convergence_analysis(self, analysis):
        """格式化收敛分析"""
        lines = ["\nConvergence analysis results:"]
        for metric, value in analysis.items():
            lines.append(f"- {metric}: {value:.4f}")
        return '\n'.join(lines)

    def _format_statistical_analysis(self, analysis):
        """格式化统计分析"""
        lines = ["\nStatistical analysis results:"]

        if 'baseline_comparison' in analysis:
            lines.append("\nComparison with baseline:")
            for metric, stats in analysis['baseline_comparison'].items():
                lines.append(f"\n{metric}:")
                lines.append(f"- t-statistic: {stats['t_statistic']:.4f}")
                lines.append(f"- p-value: {stats['p_value']:.4f}")

        return '\n'.join(lines)


if __name__ == "__main__":
    # 示例使用
    analyzer = ResultAnalyzer("experiments/latest")
    analyzer.generate_report("experiments/latest/analysis")