# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import operator as op

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import json
from misc.bbox_transform import bbox_overlaps_batch
import numbers
import random
import math
from PIL import Image, ImageOps, ImageEnhance

try:
    import accimage
except ImportError:
    accimage = None
import types
import warnings
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

from tensorboardX import SummaryWriter

noc_object = ['bus', 'bottle', 'couch', 'microwave',
              'pizza', 'racket', 'suitcase', 'zebra']
noc_index = [6, 40, 58, 69, 54, 39, 29, 23]

noc_word_map = {'bus': 'car', 'bottle': 'cup',
                'couch': 'chair', 'microwave': 'oven',
                'pizza': 'cake', 'tennis racket': 'baseball bat',
                'suitcase': 'handbag', 'zebra': 'horse'}


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
"""
def decode_sequence(itow, itod, ltow, itoc, wtod, seq, bn_seq, fg_seq, vocab_size, opt):
    N, D = seq.size()

    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i,j]
            if ix > vocab_size:
                det_word = itod[fg_seq[i,j].item()]
                det_class = itoc[wtod[det_word]]
                if opt.decode_noc and det_class in noc_object:
                    det_word = det_class

                if (bn_seq[i,j] == 1) and det_word in ltow:
                    word = ltow[det_word]
                else:
                    word = det_word
                # word = '[ ' + word + ' ]'
            else:
                if ix == 0:
                    break
                else:
                    word = itow[str(ix.item())]
            txt = txt + word
        out.append(txt)
    return out
"""


def decode_sequence(itow, itod, ltow, itoc, wtod, seq, vocab_size, opt):
    N, D = seq.size()

    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            if j >= 1:
                txt = txt + ' '
            ix = seq[i, j]
            if ix == 0:
                break
            else:
                word = itow[str(ix.item())]
            txt = txt + word
        out.append(txt)
    return out


def repackage_hidden(h, batch_size):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.resize_(h.size(0), batch_size, h.size(2)).zero_())
    else:
        return tuple(repackage_hidden(v, batch_size) for v in h)


class LMCriterion(nn.Module):
    def __init__(self, opt):
        super(LMCriterion, self).__init__()
        self.vocab_size = opt.vocab_size

    def forward(self, txt_input, att2_weights, ground_weights, target, att2_target, input_seq):
        # att2_weights and ground_weights have the same target
        assert (torch.sum(target >= self.vocab_size) == 0)
        txt_mask = target.data.gt(0)  # generate the mask
        txt_mask = torch.cat(
            [txt_mask.new(txt_mask.size(0), 1).fill_(1), txt_mask[:, :-1]], 1)

        target = target.view(-1, 1)
        txt_select = torch.gather(txt_input, 1, target)
        # if isinstance(txt_input, Variable):
        #     txt_mask = Variable(txt_mask)
        txt_out = - torch.masked_select(txt_select, txt_mask.view(-1, 1))

        assert (txt_out.size(0) == torch.sum(txt_mask.data))
        loss = torch.mean(txt_out)

        # attention loss
        # assert att2_target.sum() != 0  # if there is no target, the `att2_loss` and `ground_loss` will be NaN
        if att2_target.sum() != 0:
            # att2_loss = -torch.mean(torch.masked_select(F.log_softmax(att2_weights, dim=2), att2_target.byte()))
            att2_loss = - \
                torch.mean(torch.masked_select(
                    F.log_softmax(att2_weights, dim=2), att2_target))
            # grounding loss
            # ground_loss = -torch.mean(torch.masked_select(F.log_softmax(ground_weights, dim=2), att2_target.byte()))
            ground_loss = - \
                torch.mean(torch.masked_select(
                    F.log_softmax(ground_weights, dim=2), att2_target))
        else:
            att2_loss, ground_loss = torch.zeros(1).to(
                loss.device), torch.zeros(1).to(loss.device)

        # matching loss
        # vis_mask = (input_seq > self.vocab_size)
        # vis_mask = vis_mask.unsqueeze(2).expand_as(att2_weights)

        # match_loss = F.kl_div(
        #     torch.masked_select(F.log_softmax(att2_weights, dim=2), vis_mask),
        #     torch.masked_select(F.softmax(Variable(ground_weights.data), dim=2), vis_mask))

        return loss, att2_loss, ground_loss


class LanguageCriterion(nn.Module):
    """Reshape prediction and target, compute loss"""

    def __init__(self):
        super(LanguageCriterion, self).__init__()

    def forward(self, txt_input, target):
        txt_mask = target.data.gt(0)  # generate the mask
        txt_mask = torch.cat(
            [txt_mask.new(txt_mask.size(0), 1).fill_(1), txt_mask[:, :-1]], 1)

        target = target.view(-1, 1)
        txt_select = torch.gather(txt_input, 1, target)
        txt_out = - torch.masked_select(txt_select, txt_mask.view(-1, 1))

        assert (txt_out.size(0) == torch.sum(txt_mask.data))
        loss = torch.mean(txt_out)
        return loss


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def pad(img, padding, fill=0):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)


class RandomCropWithBbox(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, proposals, bboxs):
        """
        Args:
            img (PIL Image): Image to be cropped.
            proposals, bboxs: proposals and bboxs to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        proposals[:, 1] = proposals[:, 1] - i
        proposals[:, 3] = proposals[:, 3] - i
        proposals[:, 1] = np.clip(proposals[:, 1], 0, h - 1)
        proposals[:, 3] = np.clip(proposals[:, 3], 0, h - 1)

        proposals[:, 0] = proposals[:, 0] - j
        proposals[:, 2] = proposals[:, 2] - j
        proposals[:, 0] = np.clip(proposals[:, 0], 0, w - 1)
        proposals[:, 2] = np.clip(proposals[:, 2], 0, w - 1)

        bboxs[:, 1] = bboxs[:, 1] - i
        bboxs[:, 3] = bboxs[:, 3] - i
        bboxs[:, 1] = np.clip(bboxs[:, 1], 0, h - 1)
        bboxs[:, 3] = np.clip(bboxs[:, 3], 0, h - 1)

        bboxs[:, 0] = bboxs[:, 0] - j
        bboxs[:, 2] = bboxs[:, 2] - j
        bboxs[:, 0] = np.clip(bboxs[:, 0], 0, w - 1)
        bboxs[:, 2] = np.clip(bboxs[:, 2], 0, w - 1)

        return crop(img, i, j, h, w), proposals, bboxs


def resize_bbox(bbox, width, height, rwidth, rheight):
    """
    resize the bbox from height width to rheight rwidth
    bbox: x,y,width, height.
    """
    width_ratio = rwidth / float(width)
    height_ratio = rheight / float(height)

    bbox[:, 0] = bbox[:, 0] * width_ratio
    bbox[:, 2] = bbox[:, 2] * width_ratio
    bbox[:, 1] = bbox[:, 1] * height_ratio
    bbox[:, 3] = bbox[:, 3] * height_ratio

    return bbox


def bbox_overlaps(rois, gt_box, frm_mask):
    overlaps = bbox_overlaps_batch(rois[:, :, :5], gt_box[:, :, :5], frm_mask)

    return overlaps


def sim_mat_target(overlaps, pad_gt_bboxs):
    # overlaps: B, num_rois, num_box
    # pad_gt_bboxs: B, num_box (class labels)
    B, num_rois, num_box = overlaps.size()
    assert (num_box == pad_gt_bboxs.size(1))
    masked_labels = (overlaps > 0.5).long() * pad_gt_bboxs.view(B,
                                                                1, num_box).long()  # could try a higher threshold
    return masked_labels.permute(0, 2, 1).contiguous()


def bbox_target(mask, overlaps, seq, seq_update, vocab_size):
    mask = mask.data.contiguous()
    overlaps_copy = overlaps.clone()

    max_rois = overlaps.size(1)
    batch_size = overlaps.size(0)

    overlaps_copy.masked_fill_(
        mask.view(batch_size, 1, -1).expand_as(overlaps_copy), 0)
    max_overlaps, gt_assignment = torch.max(overlaps_copy, 2)

    # get the labels.
    # labels = (max_overlaps > 0.5).float()
    labels = max_overlaps > 0.5
    no_proposal_idx = (labels.sum(1) > 0) != (seq.data[:, 2] > 0)

    # (deprecated) convert vis word to text word if there is not matched proposal
    if no_proposal_idx.sum() > 0:
        seq_update[:, 0][no_proposal_idx] = seq_update[:, 3][no_proposal_idx]
        seq_update[:, 1][no_proposal_idx] = 0
        seq_update[:, 2][no_proposal_idx] = 0

    return labels


def _affine_grid_gen(rois, input_size, grid_size):
    rois = rois.detach()
    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = input_size[0]
    width = input_size[1]

    zero = Variable(rois.data.new(rois.size(0), 1).zero_())
    theta = torch.cat([
        (x2 - x1) / (width - 1),
        zero,
        (x1 + x2 - width + 1) / (width - 1),
        zero,
        (y2 - y1) / (height - 1),
        (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size(
        (rois.size(0), 1, grid_size, grid_size)))

    return grid


def _jitter_boxes(gt_boxes, jitter=0.05):
    """
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(
        jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(
        jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes


color_pad = ['red', 'green', 'blue', 'cyan', 'brown', 'orange']


def vis_detections(ax, class_name, dets, color_i, rest_flag=0):
    """Visual debugging of detections."""
    bbox = tuple(int(np.round(x)) for x in dets[:4])
    score = dets[-1]

    if rest_flag == 0:
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,  # remove background
                lw=3,
                color=color_pad[color_i]
            )
        )

        ax.text(bbox[0] + 5, bbox[1] + 13, '%s' % (class_name), fontsize=9,
                fontweight='bold', backgroundcolor=color_pad[color_i])
    else:
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,  # remove background
                lw=2,
                color='grey'
            )
        )
        ax.text(bbox[0] + 5, bbox[1] + 13, '%s' % (class_name),
                fontsize=9, fontweight='bold', backgroundcolor='grey')
    return ax


def cbs_beam_tag(num):
    tags = []
    for i in range(num + 1):
        for tag in itertools.combinations(range(num), i):
            tags.append(tag)
    return len(tags), tags


def cmpSet(t1, t2):
    return sorted(t1) == sorted(t2)


def containSet(List1, t1):
    # List1: return the index that contain
    # t1: tupple we want to match

    if t1 == tuple([]):
        return [tag for tag in List1 if len(tag) <= 1]
    else:
        List = []
        for t in List1:
            flag = True
            for tag in t1:
                if tag not in t:
                    flag = False

            if flag == True and len(t) <= len(t1) + 1:
                List.append(t)
    return List


def set_tb_logger(log_dir, exp_name, resume):
    """ Set up TensorBoard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn(
                'Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
