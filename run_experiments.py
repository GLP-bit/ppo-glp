#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import numpy as np
from datetime import datetime
from config import SystemParameters, TrainingParameters
from environment import EdgeOffloadingEnv
from agent import PPOAgent
from baseline import run_baseline
from evaluation import ComparativeEvaluator
from visualize import Visualizer


def setup_experiment_dir():
    """创建实验目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiments', f'comparison_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def train_ppo_agent(env, training_params, exp_dir):
    """训练PPO智能体"""
    print("\nTraining PPO agent...")
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    agent = PPOAgent(state_dim, action_dim, training_params)

    # 训练过程
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

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.3f}")

    # 保存模型
    agent.save_model(os.path.join(exp_dir, 'ppo_model.pth'))
    return agent


def evaluate_agent(agent, env, num_episodes):
    """评估单个智能体的性能"""
    metrics = {
        'rewards': [],
        'delays': [],
        'energies': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        steps = 0
        max_steps = 200  # 添加最大步数限制

        for step in range(max_steps):  # 使用for循环替代while循环
            if hasattr(agent, 'select_action'):
                # PPO agent
                action, _, _ = agent.select_action(state)
            else:
                # Baseline agent
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_delay += info.get('delay', 0)
            episode_energy += info.get('energy', 0)
            steps += 1

            state = next_state
            if done:
                break

        # 确保不会除以0
        steps = max(1, steps)
        metrics['rewards'].append(episode_reward)
        metrics['delays'].append(episode_delay / steps)
        metrics['energies'].append(episode_energy / steps)

        # 添加进度打印
        if (episode + 1) % 10 == 0:
            print(f"Evaluation progress: {episode + 1}/{num_episodes} episodes")

    return metrics


def evaluate_all_methods(env, ppo_agent, num_episodes, exp_dir):
    """评估所有方法的性能"""
    print("\nEvaluating all methods...")
    results = {}

    # 评估PPO
    print("\nEvaluating PPO...")
    ppo_metrics = evaluate_agent(ppo_agent, env, num_episodes)
    results['PPO'] = ppo_metrics

    # 评估基准方法
    baseline_types = ['greedy', 'round_robin', 'random', 'priority', 'adaptive']
    for baseline_type in baseline_types:
        print(f"\nEvaluating {baseline_type}...")
        metrics = run_baseline(env, baseline_type, num_episodes)
        results[baseline_type.capitalize()] = metrics

        # 打印当前方法的平均性能
        avg_reward = np.mean(metrics['rewards'])
        avg_delay = np.mean(metrics['delays'])
        avg_energy = np.mean(metrics['energies'])
        print(f"{baseline_type} performance:")
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Average delay: {avg_delay:.3f}")
        print(f"Average energy: {avg_energy:.3f}")

    return results


def create_comparison_visualizations(results, exp_dir):
    """创建性能对比可视化"""
    visualizer = Visualizer(exp_dir)

    # 比较奖励
    rewards_data = [results[method]['rewards'] for method in results.keys()]
    visualizer.plot_comparison(
        data_list=rewards_data,
        labels=list(results.keys()),
        metric_name='Average Reward'
    )

    # 比较延迟
    delays_data = [results[method]['delays'] for method in results.keys()]
    visualizer.plot_comparison(
        data_list=delays_data,
        labels=list(results.keys()),
        metric_name='Average Delay'
    )

    # 比较能耗
    energies_data = [results[method]['energies'] for method in results.keys()]
    visualizer.plot_comparison(
        data_list=energies_data,
        labels=list(results.keys()),
        metric_name='Average Energy'
    )


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Run edge computing offloading experiments')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Number of episodes for training')
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 创建实验目录
    exp_dir = setup_experiment_dir()

    # 创建环境和参数配置
    system_params = SystemParameters()
    training_params = TrainingParameters()
    training_params.num_episodes = args.num_episodes

    env = EdgeOffloadingEnv(system_params)

    # 训练PPO智能体
    ppo_agent = train_ppo_agent(env, training_params, exp_dir)

    # 评估所有方法
    results = evaluate_all_methods(env, ppo_agent, args.eval_episodes, exp_dir)

    # 创建对比可视化
    create_comparison_visualizations(results, exp_dir)

    print(f"\nExperiment completed. Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()