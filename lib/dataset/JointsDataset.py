# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def __len__(
        self,
    ):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        imgnum = db_rec["imgnum"] if "imgnum" in db_rec else ""
        category_id = db_rec["category_id"] if "category_id" in db_rec else ""
        area = db_rec["area"] if "area" in db_rec else ""

        if self.data_format == "zip":
            from utils import zipreader

            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints = db_rec["joints_3d"]
        joints_vis = db_rec["joints_3d_vis"]

        c = db_rec["center"]
        s = db_rec["scale"]
        score = db_rec["score"] if "score" in db_rec else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        
        # 이미지를 288x384로 크롭하는 코드
        y, x, _ = data_numpy.shape
        start_x = x // 2 - int(self.image_size[0]) // 2
        start_y = y // 2 - int(self.image_size[1]) // 2
        input = cv2.warpAffine(
            data_numpy,
            np.float32([[1, 0, -start_x], [0, 1, -start_y]]),
            (int(self.image_size[0]), int(self.image_size[1])),
        )

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            "image": image_file,
            "filename": filename,
            "imgnum": imgnum,
            "joints": joints,
            "joints_vis": joints_vis,
            "center": c,
            "scale": s,
            "rotation": r,
            "score": score,
            "category_id": category_id,
            "area": area,
        }
        return input, target, target_weight, meta
