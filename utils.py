import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import json
import os


class MetricLogger:
    """用于记录和保存训练指标的类"""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.reset_metrics()
        self.window_size = 100  # 用于计算移动平均

    def reset_metrics(self):
        """重置所有指标"""
        self.metrics = {
            'episode_rewards': [],
            'avg_rewards': [],
            'local_processing_ratio': [],
            'edge_processing_ratio': [],
            'preprocess_edge_ratio': [],
            'delays': [],
            'energy_consumption': [],
            'task_priorities': [],
            'losses': {
                'actor_loss': [],
                'critic_loss': [],
                'entropy_loss': [],
                'total_loss': []
            },
            'performance': {
                'mean_reward': None,
                'std_reward': None,
                'max_reward': None,
                'min_reward': None
            }
        }

    def log_episode(self, episode_metrics):
        """记录一个episode的指标"""
        # 记录基本指标
        for key, value in episode_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

        # 计算移动平均奖励
        if len(self.metrics['episode_rewards']) >= self.window_size:
            avg_reward = np.mean(self.metrics['episode_rewards'][-self.window_size:])
            self.metrics['avg_rewards'].append(avg_reward)

        # 更新性能统计
        if len(self.metrics['episode_rewards']) > 0:
            rewards = np.array(self.metrics['episode_rewards'])
            self.metrics['performance'].update({
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards))
            })

    def log_losses(self, loss_dict):
        """记录损失值"""
        for key, value in loss_dict.items():
            if key in self.metrics['losses']:
                self.metrics['losses'][key].append(value)

    def get_latest_metrics(self):
        """获取最新的指标"""
        latest = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value:
                latest[key] = value[-1]
            elif isinstance(value, dict):
                latest[key] = {k: v[-1] if isinstance(v, list) and v else v
                             for k, v in value.items()}
        return latest

    def save_metrics(self):
        """保存指标到文件"""
        metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
        try:
            # 将numpy类型转换为Python原生类型
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, list):
                    serializable_metrics[key] = [float(v) if isinstance(v, np.number) else v
                                               for v in value]
                elif isinstance(value, dict):
                    serializable_metrics[key] = {k: float(v) if isinstance(v, np.number) else v
                                               for k, v in value.items()}
                else:
                    serializable_metrics[key] = float(value) if isinstance(value, np.number) else value

            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")

    def plot_metrics(self):
        """绘制训练指标图表"""
        plt.figure(figsize=(15, 12))

        # 绘制奖励曲线
        plt.subplot(3, 2, 1)
        self._plot_rewards()

        # 绘制决策分布
        plt.subplot(3, 2, 2)
        self._plot_decision_distribution()

        # 绘制延迟分布
        plt.subplot(3, 2, 3)
        self._plot_delays()

        # 绘制能耗分布
        plt.subplot(3, 2, 4)
        self._plot_energy()

        # 绘制损失曲线
        plt.subplot(3, 2, 5)
        self._plot_losses()

        # 保存图表
        try:
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
        finally:
            plt.close()

    def _plot_rewards(self):
        """绘制奖励曲线"""
        if self.metrics['episode_rewards']:
            plt.plot(self.metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
        if self.metrics['avg_rewards']:
            plt.plot(self.metrics['avg_rewards'], label='Moving Average')
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

    def _plot_decision_distribution(self):
        """绘制决策分布"""
        if all(len(self.metrics[key]) > 0 for key in
              ['local_processing_ratio', 'edge_processing_ratio', 'preprocess_edge_ratio']):
            plt.stackplot(range(len(self.metrics['local_processing_ratio'])),
                         self.metrics['local_processing_ratio'],
                         self.metrics['edge_processing_ratio'],
                         self.metrics['preprocess_edge_ratio'],
                         labels=['Local', 'Edge', 'Preprocess+Edge'])
            plt.title('Decision Distribution')
            plt.xlabel('Episode')
            plt.ylabel('Ratio')
            plt.legend()
            plt.grid(True)

    def _plot_delays(self):
        """绘制延迟分布"""
        if self.metrics['delays']:
            plt.plot(self.metrics['delays'])
            plt.title('Average Delay per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Delay (s)')
            plt.grid(True)

    def _plot_energy(self):
        """绘制能耗分布"""
        if self.metrics['energy_consumption']:
            plt.plot(self.metrics['energy_consumption'])
            plt.title('Average Energy Consumption per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Energy (J)')
            plt.grid(True)

    def _plot_losses(self):
        """绘制损失曲线"""
        for loss_type, values in self.metrics['losses'].items():
            if values:
                plt.plot(values, label=loss_type)
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)


class ExperienceBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, log_prob, value):
        """存储一个转换"""
        self.buffer.append((state, action, reward, next_state, log_prob, value))

    def sample(self, batch_size):
        """采样一个批次的数据"""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        log_probs = []
        values = []

        for idx in batch:
            s, a, r, ns, lp, v = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            log_probs.append(lp)
            values.append(v)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(log_probs), np.array(values))

    def __len__(self):
        return len(self.buffer)


def compute_gae(rewards, values, next_values, gamma, lam):
    """计算广义优势估计(GAE)"""
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    running_advantage = 0

    for t in reversed(range(len(rewards))):
        running_advantage = deltas[t] + gamma * lam * running_advantage
        advantages[t] = running_advantage

    returns = advantages + values

    return advantages, returns


def normalize(x):
    """标准化数据"""
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x


class MovingAverage:
    """用于计算移动平均的类"""

    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    def get_average(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)