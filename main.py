import os
import torch
import numpy as np
from datetime import datetime
import random

from config import SystemParameters, TrainingParameters, get_default_parser
from environment import EdgeOffloadingEnv
from agent import PPOAgent
from baseline import run_baseline
from evaluation import Evaluator, ComparativeEvaluator
from train import train
from utils import MetricLogger
from visualize import Visualizer


def setup_experiment(args):
    """设置实验环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = os.path.join('experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 创建配置
    system_params = SystemParameters.from_args(args)
    training_params = TrainingParameters.from_args(args)

    # 保存配置
    system_params.save(os.path.join(exp_dir, 'system_params.json'))
    training_params.save(os.path.join(exp_dir, 'training_params.json'))

    return exp_dir, system_params, training_params


def train_model(args, exp_dir):
    """训练模型"""
    # 设置实验
    _, system_params, training_params = setup_experiment(args)

    # 创建环境、智能体和记录器
    env = EdgeOffloadingEnv(system_params)
    agent = PPOAgent(env.get_state_dim(), env.get_action_dim(), training_params)
    metric_logger = MetricLogger(exp_dir)

    # 训练模型
    print("\nStarting training...")
    rewards_history = train(env, agent, training_params, metric_logger)

    # 创建可视化器并生成训练过程可视化
    visualizer = Visualizer(exp_dir)
    visualizer.create_summary_plot(metric_logger.metrics)

    # 保存训练信息
    visualizer.save_training_info(
        vars(args),
        vars(training_params),
        {
            'final_reward': np.mean(rewards_history[-100:]),
            'max_reward': max(rewards_history),
            'total_episodes': len(rewards_history)
        }
    )

    return agent


def evaluate_model(args, exp_dir, agent=None):
    """评估模型"""
    # 设置实验
    _, system_params, _ = setup_experiment(args)

    # 创建环境
    env = EdgeOffloadingEnv(system_params)

    # 如果没有提供agent，则加载模型
    if agent is None:
        if args.model_path is None:
            raise ValueError("Please provide model path for evaluation")
        training_params = TrainingParameters.from_args(args)
        agent = PPOAgent(env.get_state_dim(), env.get_action_dim(), training_params)
        agent.load_model(args.model_path)

    # 创建评估器
    evaluator = Evaluator(agent, env, exp_dir)

    # 进行评估
    print("\nStarting evaluation...")
    stats = evaluator.evaluate(args.eval_episodes)

    return stats


def run_baseline_comparison(args, exp_dir):
    """运行基准方法对比"""
    # 设置实验
    _, system_params, _ = setup_experiment(args)

    # 创建环境
    env = EdgeOffloadingEnv(system_params)

    print(f"\nRunning baseline: {args.baseline_type}")
    baseline_metrics = run_baseline(env, args.baseline_type, args.eval_episodes)

    # 如果提供了PPO模型，进行对比
    if args.model_path is not None:
        ppo_stats = evaluate_model(args, exp_dir)

        # 创建对比评估器
        comparative_evaluator = ComparativeEvaluator(exp_dir)
        comparative_evaluator.compare_methods({
            'baseline': baseline_metrics,
            'ppo': ppo_stats
        })

    return baseline_metrics


def main():
    """主函数"""
    # 解析命令行参数
    parser = get_default_parser()
    args = parser.parse_args()

    # 创建实验目录
    exp_dir, _, _ = setup_experiment(args)

    try:
        if args.mode == 'train':
            # 训练模式
            agent = train_model(args, exp_dir)
            if args.eval_episodes > 0:
                evaluate_model(args, exp_dir, agent)

        elif args.mode == 'test':
            # 测试模式
            stats = evaluate_model(args, exp_dir)
            print("\nEvaluation Results:")
            for metric, value in stats.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")

        elif args.mode == 'baseline':
            # 基准测试模式
            baseline_metrics = run_baseline_comparison(args, exp_dir)
            print("\nBaseline Results:")
            for metric, value in baseline_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

    print(f"\nExperiment completed. Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()