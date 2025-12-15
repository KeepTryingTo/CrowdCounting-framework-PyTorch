"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-14:50
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

scheduler_configs = {
    # StepLR配置
    'step': {
        'name': 'StepLR',
        'step_size': 30,
        'gamma': 0.1
    },

    # MultiStepLR配置
    'multistep': {
        'name': 'MultiStepLR',
        'milestones': [30, 80, 120],
        'gamma': 0.1
    },

    # ExponentialLR配置
    'exponential': {
        'name': 'ExponentialLR',
        'gamma': 0.95
    },

    # CosineAnnealingLR配置
    'cosine': {
        'name': 'CosineAnnealingLR',
        'T_max': 100,
        'eta_min': 1e-6
    },

    # CosineAnnealingWarmRestarts配置
    'cosine_warm': {
        'name': 'CosineAnnealingWarmRestarts',
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6
    },

    # ReduceLROnPlateau配置
    'plateau': {
        'name': 'ReduceLROnPlateau',
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 0,
        'eps': 1e-8,
        'verbose': True
    },
    # LambdaLR配置
    'lambda': {
        'name': 'LambdaLR',
        'lr_lambda': lambda epoch: 0.95 ** epoch
    },

    # CyclicLR配置
    'cyclic': {
        'name': 'CyclicLR',
        'base_lr': 0.001,
        'max_lr': 0.01,
        'step_size_up': 2000,
        'step_size_down': None,
        'mode': 'triangular',
        'gamma': 1.0,
        'scale_fn': None,
        'scale_mode': 'cycle',
        'cycle_momentum': True,
        'base_momentum': 0.8,
        'max_momentum': 0.9
    },

    # LinearLR配置（需要添加到类中）
    'linear': {
        'name': 'LinearLR',
        'start_factor': 1.0,
        'end_factor': 0.0,
        'total_iters': 100
    },

    # SequentialLR配置（需要添加到类中）
    'sequential': {
        'name': 'SequentialLR',
        'schedulers': [
            {'name': 'StepLR', 'step_size': 30, 'gamma': 0.1},
            {'name': 'ExponentialLR', 'gamma': 0.95}
        ],
        'milestones': [30]
    }
}
