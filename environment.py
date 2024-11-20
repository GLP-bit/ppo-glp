import numpy as np


class EdgeOffloadingEnv:
    def __init__(self, params):
        self.params = params
        self.current_state = None
        self.reset()

    def reset(self):
        # 初始化系统状态
        self.current_state = {
            'task_size': np.random.uniform(*self.params.task_size_range),
            'task_cycles': np.random.randint(*self.params.task_cpu_cycles_range),
            'task_priority': np.random.randint(*self.params.task_priority_range),
            'channel_gain': self._generate_channel_gain(),
            'local_cpu_usage': np.random.uniform(0.2, 0.8),
            'edge_cpu_usage': np.random.uniform(0.2, 0.8)
        }
        return self._get_state_vector()

    def _generate_channel_gain(self):
        distance = np.random.uniform(50, 200)  # 距离范围：50-200m
        return 1 / (distance ** self.params.path_loss_exponent)

    def _get_state_vector(self):
        return np.array([
            self.current_state['task_size'],
            self.current_state['task_cycles'],
            self.current_state['task_priority'],
            self.current_state['channel_gain'],
            self.current_state['local_cpu_usage'],
            self.current_state['edge_cpu_usage']
        ])

    def step(self, action):
        # 计算延迟和能耗
        delay, energy = self._calculate_metrics(action)

        # 计算奖励
        reward = self._calculate_reward(delay, energy)

        # 更新状态
        self.current_state = {
            'task_size': np.random.uniform(*self.params.task_size_range),
            'task_cycles': np.random.randint(*self.params.task_cpu_cycles_range),
            'task_priority': np.random.randint(*self.params.task_priority_range),
            'channel_gain': self._generate_channel_gain(),
            'local_cpu_usage': np.random.uniform(0.2, 0.8),
            'edge_cpu_usage': np.random.uniform(0.2, 0.8)
        }

        # 返回正确的info字典
        info = {
            'delay': delay,
            'energy': energy,
            'priority': self.current_state['task_priority']
        }

        return self._get_state_vector(), reward, False, info

    def _calculate_metrics(self, action):
        if action == 0:  # 本地处理
            delay = self._calculate_local_delay()
            energy = self._calculate_local_energy()
        elif action == 1:  # 边缘处理
            delay = self._calculate_edge_delay()
            energy = self._calculate_edge_energy()
        else:  # 预处理后边缘处理
            delay = self._calculate_preprocessing_edge_delay()
            energy = self._calculate_preprocessing_edge_energy()
        return delay, energy

    def _calculate_local_delay(self):
        return self.current_state['task_cycles'] / (self.params.local_cpu_freq * 1000)

    def _calculate_edge_delay(self):
        transmission_rate = self._calculate_transmission_rate()
        transmission_delay = self.current_state['task_size'] * 8 / transmission_rate
        processing_delay = self.current_state['task_cycles'] / (self.params.edge_cpu_freq * 1000)
        return transmission_delay + processing_delay

    def _calculate_preprocessing_edge_delay(self):
        preprocessing_delay = self.params.preprocessing_cycles / (self.params.local_cpu_freq * 1000)
        reduced_size = self.current_state['task_size'] * (1 - self.params.preprocessing_ratio)
        transmission_rate = self._calculate_transmission_rate()
        transmission_delay = reduced_size * 8 / transmission_rate
        processing_delay = self.current_state['task_cycles'] / (self.params.edge_cpu_freq * 1000)
        return preprocessing_delay + transmission_delay + processing_delay

    def _calculate_transmission_rate(self):
        snr = (self.params.transmission_power * self.current_state['channel_gain']) / self.params.noise_power
        return self.params.bandwidth * np.log2(1 + snr)

    def _calculate_local_energy(self):
        return self.params.local_cpu_power * self._calculate_local_delay()

    def _calculate_edge_energy(self):
        transmission_energy = self.params.transmission_power * (
                    self.current_state['task_size'] * 8 / self._calculate_transmission_rate())
        processing_energy = self.params.edge_cpu_power * (
                    self.current_state['task_cycles'] / (self.params.edge_cpu_freq * 1000))
        return transmission_energy + processing_energy

    def _calculate_preprocessing_edge_energy(self):
        preprocessing_energy = self.params.local_cpu_power * (
                    self.params.preprocessing_cycles / (self.params.local_cpu_freq * 1000))
        reduced_size = self.current_state['task_size'] * (1 - self.params.preprocessing_ratio)
        transmission_energy = self.params.transmission_power * (reduced_size * 8 / self._calculate_transmission_rate())
        processing_energy = self.params.edge_cpu_power * (
                    self.current_state['task_cycles'] / (self.params.edge_cpu_freq * 1000))
        return preprocessing_energy + transmission_energy + processing_energy

    def _calculate_reward(self, delay, energy):
        """计算奖励

        使用动态归一化的延迟和能耗值计算奖励
        """
        # 获取当前任务的优先级
        priority_factor = self.current_state['task_priority'] / self.params.task_priority_range[1]

        # 使用agent中维护的统计值进行归一化
        if hasattr(self, 'agent'):  # 如果已经设置了agent
            normalized_delay = (delay - self.agent.running_delay_mean) / (self.agent.running_delay_std + 1e-8)
            normalized_energy = (energy - self.agent.running_energy_mean) / (self.agent.running_energy_std + 1e-8)
        else:  # 否则使用一个保守的归一化方案
            normalized_delay = delay / max(1.0, self._estimate_max_delay())
            normalized_energy = energy / max(1.0, self._estimate_max_energy())

        # 计算加权奖励
        weighted_cost = (
                self.params.delay_weight * normalized_delay * priority_factor +
                self.params.energy_weight * normalized_energy
        )

        return -weighted_cost  # 转换为奖励

    def _estimate_max_delay(self):
        """估计最大可能延迟"""
        # 假设最差情况: 最大任务量、最低处理速度
        max_task_cycles = self.params.task_cpu_cycles_range[1]
        min_freq = min(self.params.local_cpu_freq, self.params.edge_cpu_freq)
        max_size = self.params.task_size_range[1]
        min_bandwidth = self.params.bandwidth

        # 考虑最差传输条件
        worst_transmission_delay = max_size * 8 / (min_bandwidth * 1e6)  # 考虑带宽单位是MHz
        worst_processing_delay = max_task_cycles / (min_freq * 1e9)  # 考虑频率单位是GHz

        return worst_transmission_delay + worst_processing_delay

    def _estimate_max_energy(self):
        """估计最大可能能耗"""
        # 假设最差情况: 最大功率、最长处理时间
        max_power = max(self.params.local_cpu_power, self.params.edge_cpu_power)
        max_delay = self._estimate_max_delay()
        max_transmission_power = self.params.transmission_power

        return max_power * max_delay + max_transmission_power * max_delay

    def get_state_dim(self):
        return 6

    def get_action_dim(self):
        return 3