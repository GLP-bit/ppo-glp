import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from config import SystemParameters, TrainingParameters
from environment import EdgeOffloadingEnv
from agent import PPOAgent
from evaluation import Evaluator
from utils import MetricLogger


class ExperimentManager:
    """实验管理类，用于进行不同参数配置的实验"""

    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        self.current_exp_dir = None
        self.results = {}

    def setup_experiment(self, exp_name):
        """设置新的实验"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_exp_dir = os.path.join(self.base_dir, f'{exp_name}_{timestamp}')
        os.makedirs(self.current_exp_dir, exist_ok=True)
        return self.current_exp_dir

    def run_parameter_study(self, parameter_grid):
        """进行参数研究"""
        results = []

        # 生成所有参数组合
        param_names = parameter_grid.keys()
        param_values = parameter_grid.values()

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # 设置实验目录
            param_str = '_'.join([f'{k}_{v}' for k, v in params.items()])
            exp_dir = self.setup_experiment(f'param_study_{param_str}')

            # 运行实验
            result = self.run_single_experiment(params, exp_dir)
            results.append({**params, **result})

        # 保存结果
        self.save_parameter_study_results(results)
        return results

    def run_single_experiment(self, params, exp_dir):
        """运行单个实验"""
        # 设置系统参数
        system_params = SystemParameters()
        training_params = TrainingParameters()

        # 更新参数
        for param_name, param_value in params.items():
            if hasattr(system_params, param_name):
                setattr(system_params, param_name, param_value)
            elif hasattr(training_params, param_name):
                setattr(training_params, param_name, param_value)

        # 创建环境和智能体
        env = EdgeOffloadingEnv(system_params)
        agent = PPOAgent(env.get_state_dim(), env.get_action_dim(), training_params)

        # 设置指标记录器
        metric_logger = MetricLogger(exp_dir)

        # 训练模型
        print(f"\nRunning experiment with parameters: {params}")
        for episode in range(training_params.num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(training_params.max_steps_per_episode):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                agent.store_transition((state, action, reward, next_state, log_prob, value))
                episode_reward += reward

                if len(agent.buffer) >= training_params.batch_size:
                    agent.update()

                state = next_state
                if done:
                    break

            # 记录指标
            metrics = {
                'episode': episode,
                'reward': episode_reward,
                **info
            }
            metric_logger.log_episode(metrics)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")

        # 评估模型
        evaluator = Evaluator(agent, env, exp_dir)
        eval_stats = evaluator.evaluate(num_episodes=50)

        # 保存结果
        self.save_experiment_results(params, eval_stats, exp_dir)

        return eval_stats

    def save_parameter_study_results(self, results):
        """保存参数研究结果"""
        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.base_dir, 'parameter_study_results.csv')
        results_df.to_csv(results_path, index=False)

        # 生成结果可视化
        self.visualize_parameter_study(results_df)

    def save_experiment_results(self, params, stats, exp_dir):
        """保存单个实验结果"""
        results = {
            'parameters': params,
            'statistics': stats
        }

        results_path = os.path.join(exp_dir, 'experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    def visualize_parameter_study(self, results_df):
        """可视化参数研究结果"""
        # 创建可视化目录
        vis_dir = os.path.join(self.base_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # 分析每个参数对性能的影响
        param_columns = [col for col in results_df.columns if col not in ['mean_reward', 'std_reward']]

        for param in param_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='mean_reward', data=results_df)
            plt.title(f'Impact of {param} on Reward')
            plt.savefig(os.path.join(vis_dir, f'{param}_impact.png'))
            plt.close()

        # 创建参数相关性热力图
        plt.figure(figsize=(12, 8))
        correlation = results_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Parameter Correlation Heatmap')
        plt.savefig(os.path.join(vis_dir, 'correlation_heatmap.png'))
        plt.close()


class SensitivityAnalysis:
    """敏感性分析类，用于分析不同参数对性能的影响"""

    def __init__(self, base_params):
        self.base_params = base_params

    def analyze_parameter_sensitivity(self, param_name, param_range):
        """分析单个参数的敏感性"""
        results = []

        for param_value in param_range:
            # 创建参数副本并修改目标参数
            params = self.base_params.copy()
            params[param_name] = param_value

            # 运行实验
            exp_manager = ExperimentManager()
            exp_dir = exp_manager.setup_experiment(f'sensitivity_{param_name}_{param_value}')
            result = exp_manager.run_single_experiment(params, exp_dir)

            results.append({
                'parameter': param_name,
                'value': param_value,
                **result
            })

        return pd.DataFrame(results)

    def plot_sensitivity_results(self, results_df, param_name, save_path):
        """绘制敏感性分析结果"""
        plt.figure(figsize=(12, 6))

        # 主要指标随参数变化的曲线
        plt.subplot(1, 2, 1)
        plt.plot(results_df['value'], results_df['mean_reward'], 'o-')
        plt.fill_between(
            results_df['value'],
            results_df['mean_reward'] - results_df['std_reward'],
            results_df['mean_reward'] + results_df['std_reward'],
            alpha=0.2
        )
        plt.xlabel(param_name)
        plt.ylabel('Mean Reward')
        plt.title(f'Sensitivity to {param_name}')

        # 标准差随参数变化的曲线
        plt.subplot(1, 2, 2)
        plt.plot(results_df['value'], results_df['std_reward'], 'o-', color='orange')
        plt.xlabel(param_name)
        plt.ylabel('Standard Deviation')
        plt.title(f'Stability with {param_name}')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def run_experiments():
    """运行一系列实验"""
    # 定义参数网格
    parameter_grid = {
        'num_ambulances': [3, 5, 7],
        'bandwidth': [10, 20, 30],
        'local_cpu_freq': [1.5, 2.0, 2.5],
        'edge_cpu_freq': [2.5, 3.0, 3.5],
        'batch_size': [16, 32, 64],
        'learning_rate': [1e-4, 3e-4, 1e-3]
    }

    # 创建实验管理器
    exp_manager = ExperimentManager()

    # 运行参数研究
    results = exp_manager.run_parameter_study(parameter_grid)

    # 进行敏感性分析
    base_params = {
        'num_ambulances': 5,
        'bandwidth': 20,
        'local_cpu_freq': 2.0,
        'edge_cpu_freq': 3.0,
        'batch_size': 32,
        'learning_rate': 3e-4
    }

    sensitivity_analyzer = SensitivityAnalysis(base_params)

    # 分析带宽敏感性
    bandwidth_range = np.linspace(5, 40, 8)
    bandwidth_results = sensitivity_analyzer.analyze_parameter_sensitivity(
        'bandwidth', bandwidth_range)
    sensitivity_analyzer.plot_sensitivity_results(
        bandwidth_results, 'bandwidth',
        os.path.join(exp_manager.base_dir, 'bandwidth_sensitivity.png'))

    # 分析计算频率敏感性
    freq_range = np.linspace(1.0, 4.0, 7)
    freq_results = sensitivity_analyzer.analyze_parameter_sensitivity(
        'edge_cpu_freq', freq_range)
    sensitivity_analyzer.plot_sensitivity_results(
        freq_results, 'edge_cpu_freq',
        os.path.join(exp_manager.base_dir, 'freq_sensitivity.png'))


if __name__ == "__main__":
    run_experiments()