import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, patch_outputs, attn_outputs = model(samples)

            loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(cls_loss=loss.item())

            ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
            metric_logger.update(pat_loss=ploss.item())
            loss = loss + ploss

            aloss = F.multilabel_soft_margin_loss(attn_outputs, targets)
            metric_logger.update(attn_loss=aloss.item())
            loss = loss + aloss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []
    patch_mAP = []
    attn_mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output, patch_output, attn_output = model(images)

            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

            ploss = criterion(patch_output, target)
            loss += ploss
            patch_output = torch.sigmoid(patch_output)

            mAP_list = compute_mAP(target, patch_output)
            patch_mAP = patch_mAP + mAP_list
            metric_logger.meters['patch_mAP'].update(np.mean(mAP_list), n=batch_size)

            aloss = criterion(attn_output, target)
            loss += aloss
            attn_output = torch.sigmoid(attn_output)

            mAP_list = compute_mAP(target, attn_output)
            attn_mAP = attn_mAP + mAP_list
            metric_logger.meters['attn_mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        '* mAP {mAP.global_avg:.3f} patch_mAP {patch_mAP.global_avg:.3f} attn_mAP {attn_mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(mAP=metric_logger.mAP, patch_mAP=metric_logger.mAP, attn_mAP=metric_logger.mAP,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args, epoch=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(args.img_ms_list).readlines()
    index = args.rank
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
        if index >= len(img_list):
            continue
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img_name = img_list[index].strip()
        index += args.world_size

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[
                    3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                output, cams, patch_attn = model(images, return_att=True, attention_type=args.attention_type)
                patch_attn = torch.sum(patch_attn, dim=0)

                if args.patch_attn_refine:
                    cams = torch.matmul(patch_attn.unsqueeze(1),
                                                  cams.view(cams.shape[0], cams.shape[1],
                                                                      -1, 1)).reshape(cams.shape[0],
                                                                                      cams.shape[1],
                                                                                      w_featmap, h_featmap)

                cams = \
                    F.interpolate(cams, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cams = cams * target.clone().view(args.nb_classes, 1, 1)

                if s % 2 == 1:
                    cams = torch.flip(cams, dims=[-1])

                cam_list.append(cams)

            sum_cam = torch.sum(torch.stack(cam_list), dim=0)
            sum_cam = sum_cam.unsqueeze(0)
            
            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b, cls_ind] > 0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cam = sum_cam[b, cls_ind, :]

                            cam = (cam - cam.min()) / (
                                    cam.max() - cam.min() + 1e-8)
                            cam = cam.cpu().numpy()

                            cam_dict[cls_ind] = cam

                            if args.attention_dir is not None:
                                file_name = img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png'
                                fname = os.path.join(args.attention_dir, file_name)
                                show_cam_on_image(orig_images[0], cam, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
