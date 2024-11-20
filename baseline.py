import numpy as np
from collections import defaultdict


class GreedyAgent:
    """贪婪策略：根据当前状态选择延迟最小的动作"""

    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        """根据当前状态选择动作"""
        delays = []

        # 计算每个动作的预期延迟
        for action in range(3):  # 3种可能的动作
            delay = self._estimate_delay(state, action)
            delays.append(delay)

        # 选择延迟最小的动作
        return np.argmin(delays)

    def _estimate_delay(self, state, action):
        """估计某个动作的延迟"""
        task_size = state[0]  # 任务大小
        task_cycles = state[1]  # 计算复杂度
        channel_gain = state[3]  # 信道增益

        if action == 0:  # 本地处理
            return task_cycles / (self.env.params.local_cpu_freq * 1000)

        elif action == 1:  # 边缘处理
            # 计算传输速率
            snr = (self.env.params.transmission_power * channel_gain) / self.env.params.noise_power
            transmission_rate = self.env.params.bandwidth * np.log2(1 + snr)

            # 传输延迟
            transmission_delay = task_size * 8 / transmission_rate
            # 处理延迟
            processing_delay = task_cycles / (self.env.params.edge_cpu_freq * 1000)

            return transmission_delay + processing_delay

        else:  # 预处理后边缘处理
            # 预处理延迟
            preprocessing_delay = self.env.params.preprocessing_cycles / (self.env.params.local_cpu_freq * 1000)

            # 预处理后的数据大小
            reduced_size = task_size * (1 - self.env.params.preprocessing_ratio)

            # 计算传输速率
            snr = (self.env.params.transmission_power * channel_gain) / self.env.params.noise_power
            transmission_rate = self.env.params.bandwidth * np.log2(1 + snr)

            # 传输延迟
            transmission_delay = reduced_size * 8 / transmission_rate
            # 处理延迟
            processing_delay = task_cycles / (self.env.params.edge_cpu_freq * 1000)

            return preprocessing_delay + transmission_delay + processing_delay


class RoundRobinAgent:
    """轮询策略：循环选择不同的动作"""

    def __init__(self):
        self.current_action = -1

    def select_action(self, state):
        self.current_action = (self.current_action + 1) % 3
        return self.current_action


class RandomAgent:
    """随机策略：随机选择动作"""

    def select_action(self, state):
        return np.random.randint(0, 3)


class PriorityBasedAgent:
    """基于优先级的策略：根据任务优先级选择不同的处理方式"""

    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        priority = state[2]  # 获取任务优先级

        if priority >= 4:  # 高优先级任务
            # 根据网络条件选择本地处理或边缘处理
            channel_gain = state[3]
            if channel_gain > 0.5:  # 信道条件好
                return 1  # 边缘处理
            else:
                return 0  # 本地处理

        elif priority >= 2:  # 中等优先级任务
            # 使用预处理+边缘处理的组合
            return 2

        else:  # 低优先级任务
            # 选择当前负载最低的处理方式
            local_cpu_usage = state[4]
            edge_cpu_usage = state[5]

            if local_cpu_usage < edge_cpu_usage:
                return 0  # 本地处理
            else:
                return 1  # 边缘处理


class AdaptiveAgent:
    """自适应策略：根据系统状态动态调整策略"""

    def __init__(self, env):
        self.env = env
        self.performance_history = defaultdict(list)
        self.learning_rate = 0.1
        self.exploration_rate = 0.1

        # 初始化每个动作的评估值
        self.action_values = np.zeros(3)

    def select_action(self, state):
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, 3)

        # 计算每个动作的得分
        scores = self._calculate_action_scores(state)
        return np.argmax(scores)

    def _calculate_action_scores(self, state):
        scores = np.zeros(3)

        for action in range(3):
            # 基础分数来自历史性能
            base_score = self.action_values[action]

            # 根据当前状态调整分数
            adjusted_score = self._adjust_score(base_score, state, action)

            scores[action] = adjusted_score

        return scores

    def _adjust_score(self, base_score, state, action):
        priority = state[2]
        channel_gain = state[3]
        local_cpu_usage = state[4]
        edge_cpu_usage = state[5]

        score = base_score

        # 根据优先级调整
        if priority >= 4 and action == 0:  # 高优先级任务倾向于本地处理
            score += 0.3

        # 根据信道条件调整
        if action in [1, 2]:  # 边缘处理相关的动作
            if channel_gain < 0.3:  # 信道条件差
                score -= 0.2
            elif channel_gain > 0.7:  # 信道条件好
                score += 0.2

        # 根据负载情况调整
        if action == 0:  # 本地处理
            score -= local_cpu_usage * 0.5
        elif action in [1, 2]:  # 边缘处理
            score -= edge_cpu_usage * 0.5

        return score

    def update(self, state, action, reward):
        """更新动作评估值"""
        self.performance_history[action].append(reward)

        # 更新动作值
        old_value = self.action_values[action]
        new_value = old_value + self.learning_rate * (reward - old_value)
        self.action_values[action] = new_value


def run_baseline(env, agent_type, num_episodes):
    """运行基准测试

    Args:
        env: 环境实例
        agent_type: 基准算法类型 ('greedy', 'round_robin', 'random', 'priority', 'adaptive')
        num_episodes: 评估回合数

    Returns:
        dict: 包含评估指标的字典
    """
    # 创建对应的基准智能体
    if agent_type == "greedy":
        agent = GreedyAgent(env)
    elif agent_type == "round_robin":
        agent = RoundRobinAgent()
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "priority":
        agent = PriorityBasedAgent(env)
    elif agent_type == "adaptive":
        agent = AdaptiveAgent(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # 初始化指标
    metrics = {
        'rewards': [],
        'delays': [],
        'energies': [],
    }

    # 评估循环
    print(f"\nEvaluating {agent_type} baseline...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        steps = 0
        max_steps = 200  # 添加最大步数限制

        # 单个episode的评估
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 记录数据
            episode_reward += reward
            episode_delay += info.get('delay', 0)
            episode_energy += info.get('energy', 0)
            steps += 1

            # 更新状态
            state = next_state

            # 如果episode结束，跳出循环
            if done:
                break

        # 确保不会除以0
        steps = max(1, steps)

        # 记录episode的指标
        metrics['rewards'].append(episode_reward)
        metrics['delays'].append(episode_delay / steps)
        metrics['energies'].append(episode_energy / steps)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['rewards'][-10:])
            print(f"Progress: {episode + 1}/{num_episodes}, Recent Avg Reward: {avg_reward:.3f}")

    # 计算并打印该方法的整体性能
    final_metrics = {
        'mean_reward': np.mean(metrics['rewards']),
        'std_reward': np.std(metrics['rewards']),
        'mean_delay': np.mean(metrics['delays']),
        'mean_energy': np.mean(metrics['energies'])
    }

    print(f"\n{agent_type.capitalize()} Baseline Results:")
    print(f"Average Reward: {final_metrics['mean_reward']:.3f} ± {final_metrics['std_reward']:.3f}")
    print(f"Average Delay: {final_metrics['mean_delay']:.3f}")
    print(f"Average Energy: {final_metrics['mean_energy']:.3f}")

    return metrics