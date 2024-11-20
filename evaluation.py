import numpy as np
import torch
from collections import defaultdict
import pandas as pd
import os
from visualize import Visualizer


class Evaluator:
    def __init__(self, agent, env, save_dir):
        self.agent = agent
        self.env = env
        self.save_dir = save_dir
        self.visualizer = Visualizer(save_dir)

    def evaluate_episode(self):
        """评估单个episode的性能"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        metrics = defaultdict(list)

        while not done:
            # 选择动作
            action, _, _ = self.agent.select_action(state)

            # 记录决策
            metrics['actions'].append(action)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 记录指标
            episode_reward += reward
            metrics['rewards'].append(reward)
            metrics['delays'].append(info.get('delay', 0))
            metrics['energies'].append(info.get('energy', 0))
            metrics['priorities'].append(info.get('priority', 0))

            state = next_state

        return episode_reward, metrics

    def evaluate(self, num_episodes=100):
        """进行完整评估"""
        total_metrics = defaultdict(list)
        episode_rewards = []

        for episode in range(num_episodes):
            reward, metrics = self.evaluate_episode()
            episode_rewards.append(reward)

            # 收集所有指标
            for key, values in metrics.items():
                total_metrics[key].extend(values)

        # 计算统计信息
        stats = self.calculate_statistics(total_metrics, episode_rewards)

        # 生成可视化
        self.generate_visualizations(total_metrics, stats)

        # 保存结果
        self.save_results(stats)

        return stats

    def calculate_statistics(self, metrics, episode_rewards):
        """计算评估统计信息"""
        stats = {}

        # 计算平均奖励和标准差
        stats['mean_reward'] = np.mean(episode_rewards)
        stats['std_reward'] = np.std(episode_rewards)

        # 计算决策分布
        actions = np.array(metrics['actions'])
        total_actions = len(actions)
        stats['local_ratio'] = np.sum(actions == 0) / total_actions
        stats['edge_ratio'] = np.sum(actions == 1) / total_actions
        stats['preprocess_ratio'] = np.sum(actions == 2) / total_actions

        # 计算性能指标统计
        stats['mean_delay'] = np.mean(metrics['delays'])
        stats['std_delay'] = np.std(metrics['delays'])
        stats['mean_energy'] = np.mean(metrics['energies'])
        stats['std_energy'] = np.std(metrics['energies'])

        # 计算任务优先级相关统计
        priority_actions = defaultdict(lambda: defaultdict(int))
        for p, a in zip(metrics['priorities'], actions):
            priority_actions[p][a] += 1

        stats['priority_distribution'] = dict(priority_actions)

        return stats

    def generate_visualizations(self, metrics, stats):
        """生成评估可视化"""
        # 绘制决策分布
        self.visualizer.plot_decision_distribution(
            stats['local_ratio'],
            stats['edge_ratio'],
            stats['preprocess_ratio']
        )

        # 绘制性能指标分布
        self.visualizer.plot_performance_metrics(
            metrics['delays'],
            metrics['energies'],
            metrics['priorities']
        )

        # 生成优先级-决策热力图数据
        priority_decision_matrix = self.calculate_priority_decision_matrix(
            metrics['priorities'],
            metrics['actions']
        )
        self.visualizer.plot_heatmap(
            priority_decision_matrix,
            'Priority-Decision Distribution',
            'Decision',
            'Priority Level'
        )

    def calculate_priority_decision_matrix(self, priorities, actions):
        """计算优先级-决策分布矩阵"""
        unique_priorities = sorted(set(priorities))
        matrix = np.zeros((len(unique_priorities), 3))  # 3种决策

        for p, a in zip(priorities, actions):
            p_idx = unique_priorities.index(p)
            matrix[p_idx, a] += 1

        # 归一化
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

        return matrix

    def save_results(self, stats):
        """保存评估结果"""
        results_file = os.path.join(self.save_dir, 'evaluation_results.csv')

        # 转换统计信息为DataFrame
        results_dict = {
            'Metric': [],
            'Value': []
        }

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                results_dict['Metric'].append(key)
                results_dict['Value'].append(value)

        df = pd.DataFrame(results_dict)
        df.to_csv(results_file, index=False)

        # 保存详细的优先级分布信息
        priority_file = os.path.join(self.save_dir, 'priority_distribution.csv')
        priority_df = pd.DataFrame.from_dict(stats['priority_distribution'], orient='index')
        priority_df.to_csv(priority_file)


class ComparativeEvaluator:
    """用于比较不同方法性能的评估器"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.visualizer = Visualizer(save_dir)

    def compare_methods(self, results_dict):
        """比较不同方法的性能"""
        # 比较平均奖励
        self.visualizer.plot_comparison(
            results_dict['baseline']['rewards'],
            results_dict['ppo']['rewards'],
            'Average Reward'
        )

        # 比较平均延迟
        self.visualizer.plot_comparison(
            results_dict['baseline']['delays'],
            results_dict['ppo']['delays'],
            'Average Delay'
        )

        # 比较平均能耗
        self.visualizer.plot_comparison(
            results_dict['baseline']['energies'],
            results_dict['ppo']['energies'],
            'Average Energy Consumption'
        )

        # 保存比较结果
        self.save_comparison_results(results_dict)

    def save_comparison_results(self, results_dict):
        """保存比较结果"""
        comparison_file = os.path.join(self.save_dir, 'method_comparison.csv')

        comparison_dict = {
            'Metric': [],
            'Method': [],
            'Value': []
        }

        # 确保数据存在
        for method_name, method_results in results_dict.items():
            if not isinstance(method_results, dict):
                print(f"Warning: Results for {method_name} is not a dictionary")
                continue

            # 获取所有可用的指标
            metrics = ['rewards', 'delays', 'energies']
            for metric in metrics:
                if metric in method_results:
                    values = method_results[metric]
                    if values:  # 确保有数据
                        mean_value = np.mean(values)
                        std_value = np.std(values)

                        # 添加均值
                        comparison_dict['Metric'].append(f'Mean {metric}')
                        comparison_dict['Method'].append(method_name)
                        comparison_dict['Value'].append(mean_value)

                        # 添加标准差
                        comparison_dict['Metric'].append(f'Std {metric}')
                        comparison_dict['Method'].append(method_name)
                        comparison_dict['Value'].append(std_value)

        # 创建并保存DataFrame
        df = pd.DataFrame(comparison_dict)
        df.to_csv(comparison_file, index=False)