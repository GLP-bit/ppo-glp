import numpy as np
from enum import Enum
from dataclasses import dataclass
import random


class TaskType(Enum):
    """任务类型枚举"""
    VITAL_SIGNS = 1  # 生命体征监测
    ECG_ANALYSIS = 2  # 心电图分析
    IMAGE_ANALYSIS = 3  # 医学图像分析
    BLOOD_ANALYSIS = 4  # 血液检测分析
    EMERGENCY_ALERT = 5  # 紧急告警


@dataclass
class Task:
    """任务数据类"""
    task_id: int
    task_type: TaskType
    size: float  # 数据大小 (MB)
    cpu_cycles: int  # 所需CPU周期数
    priority: int  # 优先级 (1-5)
    deadline: float  # 截止时间 (s)
    generation_time: float  # 生成时间
    location: tuple  # 任务生成位置 (x, y)


class TaskGenerator:
    """医疗任务生成器"""

    def __init__(self, num_ambulances=5, area_size=(1000, 1000)):
        self.num_ambulances = num_ambulances
        self.area_size = area_size
        self.task_counter = 0

        # 不同任务类型的参数配置
        self.task_configs = {
            TaskType.VITAL_SIGNS: {
                'size_range': (0.1, 0.5),
                'cycles_range': (100, 200),
                'priority_range': (1, 3),
                'deadline_range': (2, 5)
            },
            TaskType.ECG_ANALYSIS: {
                'size_range': (0.5, 2.0),
                'cycles_range': (300, 600),
                'priority_range': (2, 4),
                'deadline_range': (3, 7)
            },
            TaskType.IMAGE_ANALYSIS: {
                'size_range': (5.0, 20.0),
                'cycles_range': (800, 1500),
                'priority_range': (2, 4),
                'deadline_range': (5, 10)
            },
            TaskType.BLOOD_ANALYSIS: {
                'size_range': (0.2, 1.0),
                'cycles_range': (200, 400),
                'priority_range': (2, 3),
                'deadline_range': (4, 8)
            },
            TaskType.EMERGENCY_ALERT: {
                'size_range': (0.1, 0.3),
                'cycles_range': (50, 100),
                'priority_range': (4, 5),
                'deadline_range': (1, 3)
            }
        }

        # 任务类型的生成概率分布
        self.task_probabilities = {
            TaskType.VITAL_SIGNS: 0.4,
            TaskType.ECG_ANALYSIS: 0.2,
            TaskType.IMAGE_ANALYSIS: 0.1,
            TaskType.BLOOD_ANALYSIS: 0.2,
            TaskType.EMERGENCY_ALERT: 0.1
        }

        # 救护车位置
        self.ambulance_locations = self._initialize_ambulance_locations()

    def _initialize_ambulance_locations(self):
        """初始化救护车位置"""
        locations = []
        for _ in range(self.num_ambulances):
            x = random.uniform(0, self.area_size[0])
            y = random.uniform(0, self.area_size[1])
            locations.append((x, y))
        return locations

    def update_ambulance_locations(self, time_step):
        """更新救护车位置"""
        for i in range(self.num_ambulances):
            # 简单的随机移动模型
            dx = random.uniform(-50, 50) * time_step
            dy = random.uniform(-50, 50) * time_step

            x = max(0, min(self.area_size[0], self.ambulance_locations[i][0] + dx))
            y = max(0, min(self.area_size[1], self.ambulance_locations[i][1] + dy))

            self.ambulance_locations[i] = (x, y)

    def generate_task(self, current_time, ambulance_id=None):
        """生成单个任务"""
        # 随机选择任务类型
        task_type = np.random.choice(
            list(TaskType),
            p=[self.task_probabilities[t] for t in TaskType]
        )

        # 获取该任务类型的配置
        config = self.task_configs[task_type]

        # 生成任务参数
        size = random.uniform(*config['size_range'])
        cpu_cycles = random.randint(*config['cycles_range'])
        priority = random.randint(*config['priority_range'])
        deadline = current_time + random.uniform(*config['deadline_range'])

        # 如果指定了救护车ID，使用该救护车的位置
        if ambulance_id is not None:
            location = self.ambulance_locations[ambulance_id]
        else:
            # 随机选择一个救护车的位置
            location = random.choice(self.ambulance_locations)

        # 创建任务对象
        task = Task(
            task_id=self.task_counter,
            task_type=task_type,
            size=size,
            cpu_cycles=cpu_cycles,
            priority=priority,
            deadline=deadline,
            generation_time=current_time,
            location=location
        )

        self.task_counter += 1
        return task

    def generate_batch(self, current_time, batch_size):
        """生成一批任务"""
        return [self.generate_task(current_time) for _ in range(batch_size)]


class EmergencyScenarioGenerator:
    """紧急情况场景生成器"""

    def __init__(self, task_generator):
        self.task_generator = task_generator

    def generate_emergency_scenario(self, current_time, duration, intensity):
        """生成紧急情况场景

        Args:
            current_time: 当前时间
            duration: 紧急情况持续时间
            intensity: 紧急程度 (0-1)
        """
        # 临时调整任务生成概率
        original_probabilities = self.task_generator.task_probabilities.copy()

        # 增加紧急任务的概率
        self.task_generator.task_probabilities[TaskType.EMERGENCY_ALERT] = min(0.3, 0.1 + intensity * 0.4)
        self.task_generator.task_probabilities[TaskType.ECG_ANALYSIS] = min(0.3, 0.2 + intensity * 0.2)

        # 归一化概率
        total_prob = sum(self.task_generator.task_probabilities.values())
        for task_type in TaskType:
            self.task_generator.task_probabilities[task_type] /= total_prob

        # 生成紧急情况下的任务
        tasks = []
        time = current_time
        while time < current_time + duration:
            # 生成任务的时间间隔随强度减小
            interval = max(0.5, 2.0 * (1 - intensity))
            time += random.expovariate(1 / interval)

            if time < current_time + duration:
                task = self.task_generator.generate_task(time)
                tasks.append(task)

        # 恢复原始概率分布
        self.task_generator.task_probabilities = original_probabilities

        return tasks


class TaskQueue:
    """任务队列管理器"""

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.queue = []

    def add_task(self, task):
        """添加任务到队列"""
        if len(self.queue) < self.max_size:
            self.queue.append(task)
            # 根据优先级和截止时间排序
            self.queue.sort(key=lambda x: (-x.priority, x.deadline))
            return True
        return False

    def get_task(self):
        """获取队列中的下一个任务"""
        if self.queue:
            return self.queue.pop(0)
        return None

    def update(self, current_time):
        """更新队列，移除过期任务"""
        self.queue = [task for task in self.queue if task.deadline > current_time]


def test_task_generation():
    """测试任务生成功能"""
    generator = TaskGenerator(num_ambulances=3)
    scenario_generator = EmergencyScenarioGenerator(generator)
    queue = TaskQueue()

    # 测试普通任务生成
    print("Generating normal tasks...")
    for i in range(5):
        task = generator.generate_task(current_time=i)
        print(f"Task {task.task_id}: Type={task.task_type.name}, "
              f"Priority={task.priority}, Deadline={task.deadline:.2f}")
        queue.add_task(task)

    # 测试紧急情况场景
    print("\nGenerating emergency scenario...")
    emergency_tasks = scenario_generator.generate_emergency_scenario(
        current_time=10,
        duration=5,
        intensity=0.8
    )
    for task in emergency_tasks:
        print(f"Emergency Task {task.task_id}: Type={task.task_type.name}, "
              f"Priority={task.priority}, Deadline={task.deadline:.2f}")
        queue.add_task(task)

    # 测试队列管理
    print("\nProcessing task queue...")
    while True:
        task = queue.get_task()
        if task is None:
            break
        print(f"Processing Task {task.task_id}: Priority={task.priority}")


if __name__ == "__main__":
    test_task_generation()