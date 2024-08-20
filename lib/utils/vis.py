# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Literal

import numpy as np
import torch
import torchvision
import cv2

from core.inference import get_max_preds


# 원본이미지 mode="gt"
# 예측된 결과 이미지 mode="pred"
def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=1, padding=2, mode: Literal["gt", "pred"] = "gt"):
    """
    Args:
        mode: 원본이미지: `gt`</br>예측된 결과 이미지: `pred`
        nrow: 배치크기 보정수치 (배치크기가 재조정되면 거기에 맞게 설정 필요)
    """
    
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    scaling_factor_x = 1 
    scaling_factor_y = 1 
    
    # 예측 이미지 저장시 circle 위치가 이상한 문제 해결
    if mode == "pred":
        scaling_factor_x = ndarr.shape[0]/96
        scaling_factor_y = ndarr.shape[1]/72/nrow
    
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # 예측한 점
            joints = batch_joints[k]
            # 원본점
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint_x = x * width + padding + joint[0]*scaling_factor_x
                joint_y = y * height + padding + joint[1]*scaling_factor_y
                if  (joint[0] or joint[1]) and mode=="pred":
                    cv2.circle(ndarr, (int(joint_x), int(joint_y)), 2, [255, 0, 0], 2)
                elif (joint_vis[0] or joint_vis[1]) and mode=="gt":
                    cv2.circle(ndarr, (int(joint_x), int(joint_y)), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_image_with_joints_gt_pred(batch_image, 
                                batch_joints_gt, batch_joints_vis_gt, 
                                batch_joints_pred, batch_joints_vis_pred, 
                                file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    scaling_factor_x = ndarr.shape[0]/96
    scaling_factor_y = ndarr.shape[1]/72/nrow

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            
            # 원본 점 찍기
            joints = batch_joints_gt[k]
            joints_vis = batch_joints_vis_gt[k]
            for joint, joint_vis in zip(joints, joints_vis):
                joint_x = x * width + padding + joint[0]
                joint_y = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint_x), int(joint_y)), 2, [0, 255, 0], 2)

            # 예측 점 찍기
            joints = batch_joints_pred[k]
            joints_vis = batch_joints_vis_pred[k]
            for joint, joint_vis in zip(joints, joints_vis):
                joint_x = x * width + padding + joint[0]*scaling_factor_x
                joint_y = y * height + padding + joint[1]*scaling_factor_y
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint_x), int(joint_y)), 2, [255, 0, 0], 2)

            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_heatmaps_gt_pred(batch_image, batch_heatmaps_gt, batch_heatmaps_pred, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps_gt.size(0)
    num_joints = batch_heatmaps_gt.size(1)
    heatmap_height = batch_heatmaps_gt.size(2)
    heatmap_width = batch_heatmaps_gt.size(3)

    grid_image = np.zeros((batch_size*heatmap_height * 2,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    gts, _ = get_max_preds(batch_heatmaps_gt.detach().cpu().numpy())
    preds, _ = get_max_preds(batch_heatmaps_pred.detach().cpu().numpy())

    for i in range(batch_size):
        for idx, (kps, batch_heatmaps) in enumerate(zip([gts,preds], [batch_heatmaps_gt, batch_heatmaps_pred])):
            image = batch_image[i].mul(255)\
                                .clamp(0, 255)\
                                .byte()\
                                .permute(1, 2, 0)\
                                .cpu().numpy()
            heatmaps = batch_heatmaps[i].mul(255)\
                                        .clamp(0, 255)\
                                        .byte()\
                                        .cpu().numpy()

            resized_image = cv2.resize(image,
                                    (int(heatmap_width), int(heatmap_height)))

            height_begin = heatmap_height * (2*i + idx)
            height_end = heatmap_height * (2*i + idx + 1)
            for j in range(num_joints):
                cv2.circle(resized_image,
                        (int(kps[i][j][0]), int(kps[i][j][1])),
                        1, [0, 0, 255], 1)
                heatmap = heatmaps[j, :, :]
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                masked_image = colored_heatmap*0.7 + resized_image*0.3
                cv2.circle(masked_image,
                        (int(kps[i][j][0]), int(kps[i][j][1])),
                        1, [0, 0, 255], 1)

                width_begin = heatmap_width * (j+1)
                width_end = heatmap_width * (j+2)
                grid_image[height_begin:height_end, width_begin:width_end, :] = \
                    masked_image
                # grid_image[height_begin:height_end, width_begin:width_end, :] = \
                #     colored_heatmap*0.7 + resized_image*0.3

            grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)

def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        # 원본이미지 저장
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix), mode="gt"
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        # 예측 이미지 저장
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix), mode="pred"
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_GT_PRED:
        # 원본 + 예측 이미지 저장
        if config.TEST.USE_GT_BBOX == False:
            return
        save_batch_image_with_joints_gt_pred(input, 
        meta['joints'], meta['joints_vis'],
        torch.Tensor(joints_pred).double(), meta['joints_vis'],
        '{}_gt_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )
    