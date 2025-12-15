"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/17-14:07
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import argparse
import os.path as osp

def get_parser():
    parser = argparse.ArgumentParser(description="Training code for Domain-general Crowd Counting in Unseen Scenarios")
    # TODO data
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--data_type', type=str, default='sha',choices=['sha', 'shb', 'nwpu', 'nqrf', 'jhu'])
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--device-ids', type=str, default='0,1',help='GPU device,example "0,1,2"')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--crop_size', type=int, default=2048)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--lr_scheduler', type=bool, default=True, help="use learning ratio scheduler")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--d_ratio', type=int, default=1, help="compare output feature, image down ratio")
    parser.add_argument('--pct-start', type=float, default=0.10)
    parser.add_argument('--div-factor', type=float, default=10)
    parser.add_argument('--final_div_factor', type=float, default=10)
    parser.add_argument('--scheduler_name', type=bool, default=False)
    parser.add_argument('--custom_lr_scheduler', type=str, default='cosineannealingwarmrestarts',
                        choices=['cosineannealingwarmrestarts', 'reducelronplateau',
                                 'lambdalr', 'cycliclr', 'cosineannealinglr',
                                 'exponentiallr', 'multisteplr', 'steplr'])

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=25)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--backbone_m', type=str, default='crowdvgg', choices=['crowdvgg', 'crowdresnet'])

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCounting-framework-PyTorch/datasets/process_data/sha/train'
                        )
    parser.add_argument('--test_data_dir', type=str,
                        default=r'/home/ff/myProject/KGT/myProjects/myProjects/CrowdCounting-framework-PyTorch/datasets/process_data/sha/test'
                        )
    parser.add_argument('--logs-dir', type=str, default=osp.join('weights', 'logs'))

    # pretrained or checkpoint model
    parser.add_argument('--resume', type=str,default=None)
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")

    # weight of loss
    parser.add_argument('--lambda_local', type=float, default=0.01, help="")
    parser.add_argument('--lambda_bound_aware', type=float, default=0.01, help="")
    parser.add_argument('--lambda_consist', type=float, default=0.1, help="")
    parser.add_argument('--weight_cc', type=float, default=0.001, help="")
    parser.add_argument('--weight_global', type=float, default=50, help="")
    parser.add_argument('--is_qnrf', type=bool, default=False, help="")

    return parser