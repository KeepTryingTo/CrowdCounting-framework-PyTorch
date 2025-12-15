"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/3/25-12:50
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import torch
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

"""
    参数	                 作用	                                  典型值
    max_lr	             峰值学习率	                                0.5-1.0
    total_steps	         总迭代步数	                                epochs × batches_per_epoch
    pct_start	         上升阶段占比	                                0.3（30%）
    div_factor	         初始学习率 = max_lr / div_factor	            25.0
    final_div_factor	 最终学习率下限 = max_lr / final_div_factor	1e4
"""
class CustomOneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25., final_div_factor=1e4, last_epoch=-1):

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_up = int(total_steps * pct_start)  # 上升阶段步数
        # 初始化代码同上...
        self._disable_momentum = isinstance(optimizer, torch.optim.Adam)  # 标记Adam优化器
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.step_up:
            # 余弦平滑上升（从0到π/2相位）
            progress = self.last_epoch / self.step_up
            lr = self.max_lr / self.div_factor + (self.max_lr - self.max_lr / self.div_factor) * \
                 (1 - math.cos(progress * math.pi / 2))  # 使用余弦函数
        else:
            # 余弦下降阶段（保持原逻辑）
            progress = (self.last_epoch - self.step_up) / (self.total_steps - self.step_up)
            lr = self.max_lr * (1 + math.cos(math.pi * progress)) / 2
            lr = max(lr, self.max_lr / self.final_div_factor)
        return [lr for _ in self.base_lrs]

    def get_momentum(self):
        if self._disable_momentum:
            return None
        # 动量反向调整（适用于SGD with momentum）
        if self.last_epoch <= self.step_up:
            progress = self.last_epoch / self.step_up
            momentum = 0.95 - (0.95 - 0.85) * progress  # 从0.95降到0.85
        else:
            progress = (self.last_epoch - self.step_up) / (self.total_steps - self.step_up)
            momentum = 0.85 + (0.95 - 0.85) * (1 - math.cos(math.pi * progress)) / 2  # 从0.85升到0.95
        return momentum


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from typing import Dict, Any, Optional, Union


class LRSchedulerManager:
    """
        learning ratio scheduler manager
    """

    def __init__(self, optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]):
        """
        initialize lr scheduler
        Args:
            optimizer: optimizer instance
            scheduler_config: lr scheduler hyper parameters
        """
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.scheduler = None
        self._setup_scheduler()

    def _setup_scheduler(self):
        """according to the scheduler name and select learning ratio scheduler"""
        scheduler_name = self.scheduler_config.get('name', 'StepLR').lower()

        if scheduler_name == 'steplr':
            self.scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_config.get('step_size', 30),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )

        elif scheduler_name == 'multisteplr':
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_config.get('milestones', [30, 80]),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )

        elif scheduler_name == 'exponentiallr':
            self.scheduler = lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_config.get('gamma', 0.95)
            )

        elif scheduler_name == 'cosineannealinglr':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_config.get('T_max', 50),
                eta_min=self.scheduler_config.get('eta_min', 0)
            )

        elif scheduler_name == 'cosineannealingwarmrestarts':
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_config.get('T_0', 10),
                T_mult=self.scheduler_config.get('T_mult', 2)
            )

        elif scheduler_name == 'reducelronplateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_config.get('mode', 'min'),
                factor=self.scheduler_config.get('factor', 0.1),
                patience=self.scheduler_config.get('patience', 10),
                verbose=self.scheduler_config.get('verbose', True)
            )

        elif scheduler_name == 'lambdalr':
            lambda_func = self.scheduler_config.get('lr_lambda')
            if lambda_func is None:
                lambda_func = lambda epoch: 0.95 ** epoch
            self.scheduler = lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda_func
            )

        elif scheduler_name == 'cycliclr':
            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.scheduler_config.get('base_lr', 0.001),
                max_lr=self.scheduler_config.get('max_lr', 0.01),
                step_size_up=self.scheduler_config.get('step_size_up', 2000),
                mode=self.scheduler_config.get('mode', 'triangular')
            )

        else:
            raise ValueError(f"Can not find: {scheduler_name}")

    def step(self, metrics: Optional[float] = None):
        """
        update lr
        Args:
            metrics: 用于ReduceLROnPlateau的指标值
        """
        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
            if metrics is None:
                raise ValueError("ReduceLROnPlateau需要metrics参数")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        """return current learning ratio"""
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """:return current scheduler status"""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """load current scheduler status"""
        self.scheduler.load_state_dict(state_dict)

    def get_current_lr(self) -> float:
        """获取当前学习率（标量值）"""
        lr_list = self.get_last_lr()
        return lr_list[0] if lr_list else self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # 自定义OneCycleLR
    scheduler = CustomOneCycleLR(
        optimizer,
        max_lr=0.0001,
        total_steps=1000,  # 总迭代步数（batch数×epoch数）
        pct_start=0.3,  # 30%步数用于上升
        div_factor=10.,  # 初始lr = max_lr / 25
        final_div_factor=10  # 最终lr >= max_lr / 1e4
    )

    lrs, momentums = [], []
    for _ in range(scheduler.total_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
        momentums.append(scheduler.get_momentum())

    plt.plot(lrs, label='Learning Rate')
    # plt.plot(momentums, label='Momentum')
    plt.legend()
    plt.show()