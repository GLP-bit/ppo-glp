import numpy as np
import matplotlib.pyplot as plt
from config import SystemParameters, TrainingParameters
from environment import EdgeOffloadingEnv
from agent import PPOAgent
import os
from datetime import datetime


def train(env, agent, training_params, metric_logger):
    """
    训练函数

    Args:
        env: 环境实例
        agent: PPO智能体实例
        training_params: 训练参数
        metric_logger: 指标记录器

    Returns:
        list: 奖励历史记录
    """
    # 创建保存模型的目录
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'saved_models/{current_time}'
    os.makedirs(save_dir, exist_ok=True)

    # 训练记录
    rewards_history = []
    avg_rewards_history = []
    best_avg_reward = float('-inf')

    print("Starting training...")

    for episode in range(training_params.num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(training_params.max_steps_per_episode):
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转换
            agent.store_transition((state, action, reward, next_state, log_prob, value))

            # 更新参数
            if len(agent.buffer) >= training_params.batch_size:
                agent.update()

            episode_reward += reward
            state = next_state

            if done:
                break

        # 记录奖励
        rewards_history.append(episode_reward)

        # 记录指标
        metrics = {
            'episode_rewards': episode_reward,
            'avg_rewards': np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history),
            'delays': info.get('delay', 0),
            'energy_consumption': info.get('energy', 0),
            'task_priorities': info.get('priority', 0)
        }
        metric_logger.log_episode(metrics)

        # 计算平均奖励
        if len(rewards_history) >= 100:
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)

            # 保存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model(f'{save_dir}/best_model.pth')

        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode: {episode + 1}, Average Reward (last 10): {avg_reward:.3f}")

    # 保存最终模型
    agent.save_model(f'{save_dir}/final_model.pth')

    # 保存训练指标
    metric_logger.save_metrics()
    metric_logger.plot_metrics()

    return rewards_history


def plot_training_results(rewards, avg_rewards, save_dir):
    plt.figure(figsize=(12, 5))

    # 绘制原始奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # 绘制平滑后的平均奖励曲线
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards)
    plt.title('Average Training Rewards (over 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png')
    plt.close()


def evaluate(agent, env, num_episodes=50):
    print("\nStarting evaluation...")
    rewards = []
    local_choices = 0
    edge_choices = 0
    preprocess_choices = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(state)

            # 统计决策选择
            if action == 0:
                local_choices += 1
            elif action == 1:
                edge_choices += 1
            else:
                preprocess_choices += 1

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)

    # 计算统计信息
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    total_decisions = local_choices + edge_choices + preprocess_choices

    print(f"\nEvaluation Results:")
    print(f"Average Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"\nOffloading Decisions Distribution:")
    print(f"Local Processing: {local_choices / total_decisions * 100:.1f}%")
    print(f"Edge Processing: {edge_choices / total_decisions * 100:.1f}%")
    print(f"Preprocess + Edge: {preprocess_choices / total_decisions * 100:.1f}%")

    return mean_reward, std_reward


if __name__ == "__main__":
    # 训练模型
    trained_agent, rewards_history = train()

    # 创建新环境进行评估
    eval_env = EdgeOffloadingEnv(SystemParameters())

    # 评估模型
    evaluate(trained_agent, eval_env)