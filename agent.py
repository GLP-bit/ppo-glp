import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from models import ActorCritic
from utils import compute_gae, normalize, ExperienceBuffer


class PPOAgent:
    def __init__(self, state_dim, action_dim, training_params):
        self.params = training_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.params.learning_rate)

        # 使用ExperienceBuffer替换简单的列表
        self.buffer = ExperienceBuffer(self.params.max_buffer_size)

        # 添加归一化统计
        self.running_delay_mean = 0
        self.running_delay_std = 1
        self.running_energy_mean = 0
        self.running_energy_std = 1
        self.update_stats_n = 0

        # 添加训练统计
        self.train_iteration = 0

    def store_transition(self, transition):
        """存储经验到buffer"""
        state, action, reward, next_state, log_prob, value = transition
        self.buffer.push(state, action, reward, next_state, log_prob, value)

    def _update_running_stats(self, delay, energy):
        """更新运行时统计值"""
        self.update_stats_n += 1
        if self.update_stats_n == 1:
            self.running_delay_mean = delay
            self.running_energy_mean = energy
            self.running_delay_std = 1
            self.running_energy_std = 1
        else:
            # Welford's online algorithm
            delay_delta = delay - self.running_delay_mean
            self.running_delay_mean += delay_delta / self.update_stats_n
            self.running_delay_std = (self.running_delay_std * (self.update_stats_n - 2) +
                                      delay_delta * (delay - self.running_delay_mean)) / (self.update_stats_n - 1)

            energy_delta = energy - self.running_energy_mean
            self.running_energy_mean += energy_delta / self.update_stats_n
            self.running_energy_std = (self.running_energy_std * (self.update_stats_n - 2) +
                                       energy_delta * (energy - self.running_energy_mean)) / (self.update_stats_n - 1)

    def update(self):
        """更新策略和价值网络"""
        if len(self.buffer) < self.params.batch_size:
            return {}

        # 采样batch数据
        state_batch, action_batch, reward_batch, next_state_batch, old_log_prob_batch, value_batch = \
            self.buffer.sample(self.params.batch_size)

        # 转换为tensor
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        old_log_prob_batch = torch.FloatTensor(old_log_prob_batch).to(self.device)
        value_batch = torch.FloatTensor(value_batch).to(self.device)

        # 计算GAE
        with torch.no_grad():
            _, next_values = self.actor_critic(next_state_batch)
            advantages, returns = compute_gae(
                reward_batch,
                value_batch,
                next_values.squeeze(),
                self.params.gamma,
                0.95  # GAE lambda parameter
            )
            advantages = normalize(advantages)

        # PPO更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0

        for _ in range(self.params.update_epochs):
            # 当前策略的动作概率和状态值
            action_probs, values = self.actor_critic(state_batch)
            distribution = Categorical(action_probs)
            new_log_probs = distribution.log_prob(action_batch)

            # 计算ratio
            ratio = torch.exp(new_log_probs - old_log_prob_batch)

            # 计算surrogate损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.params.clip_epsilon, 1 + self.params.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算critic损失
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # 计算熵损失（用于鼓励探索）
            entropy_loss = -self.params.entropy_coef * distribution.entropy().mean()

            # 总损失
            total_loss = actor_loss + self.params.critic_coef * critic_loss + entropy_loss

            # 更新网络
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.params.max_grad_norm)
            self.optimizer.step()

            # 累加损失
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy_loss.item()

        # 计算平均损失
        num_updates = self.params.update_epochs
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        # 清空buffer
        self.buffer = ExperienceBuffer(self.params.max_buffer_size)

        self.train_iteration += 1

        # 返回训练统计信息
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_actor_loss + self.params.critic_coef * avg_critic_loss + avg_entropy_loss
        }

    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def save_model(self, path):
        """保存模型"""
        state_dict = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'running_stats': {
                'delay_mean': self.running_delay_mean,
                'delay_std': self.running_delay_std,
                'energy_mean': self.running_energy_mean,
                'energy_std': self.running_energy_std,
                'update_stats_n': self.update_stats_n
            },
            'train_iteration': self.train_iteration
        }
        torch.save(state_dict, path)

    def load_model(self, path):
        """加载模型"""
        state_dict = torch.load(path)
        self.actor_critic.load_state_dict(state_dict['actor_critic_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        # 加载运行时统计
        stats = state_dict['running_stats']
        self.running_delay_mean = stats['delay_mean']
        self.running_delay_std = stats['delay_std']
        self.running_energy_mean = stats['energy_mean']
        self.running_energy_std = stats['energy_std']
        self.update_stats_n = stats['update_stats_n']

        self.train_iteration = state_dict['train_iteration']