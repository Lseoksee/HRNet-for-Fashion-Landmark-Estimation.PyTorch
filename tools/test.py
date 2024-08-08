# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate

from utils.utils import create_logger

from dataset import deepfashion2
from models import pose_hrnet


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, "valid")

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size = cfg.TEST.BATCH_SIZE_PER_GPU * torch.cuda.device_count()
        logger.info("Let's use %d GPUs!" % torch.cuda.device_count())

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = pose_hrnet.get_pose_net(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)  # False
    else:
        model_state_file = os.path.join(final_output_dir, "final_state.pth")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        cfg=cfg,
        target_type=cfg.MODEL.TARGET_TYPE,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # deepfashion2.py 에 __init__ 실행됨
    # dataset(dataset.deepfashion2) 인건데 이 뜻이 dataset.deepfashion2 를 실행하겠다는것 즉 객체 생성함
    """     valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    ) """

    valid_dataset = deepfashion2(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    logger.info("=> Start testing...")

    # evaluate on validation set
    validate(
        cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir
    )


if __name__ == "__main__":
    main()
