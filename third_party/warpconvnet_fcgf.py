# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WarpConvNet port of the FCGF ResUNetBN2C descriptor network.

This adapter follows the maintained FCGF/WarpConvNet implementation. It is
kept here because the compatible binary wheel does not package this model
module, while the published FCGF checkpoint is still in the original
MinkowskiEngine parameter layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.activations import ReLU
from warpconvnet.nn.modules.sparse_conv import SparseConv3d
from warpconvnet.models.mink_unet import BasicBlock, ConvBlock, ConvTrBlock


class ResUNet2(nn.Module):
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]

    def __init__(self, in_channels=1, out_channels=32, bn_momentum=0.05,
                 normalize_feature=True, conv1_kernel_size=5):
        super().__init__()
        self.normalize_feature = normalize_feature
        c, tr = self.CHANNELS, self.TR_CHANNELS
        self.conv1 = ConvBlock(in_channels, c[1], kernel_size=conv1_kernel_size, stride=1, activation=None)
        self.block1 = BasicBlock(c[1], c[1])
        self.conv2 = ConvBlock(c[1], c[2], kernel_size=3, stride=2, activation=None)
        self.block2 = BasicBlock(c[2], c[2])
        self.conv3 = ConvBlock(c[2], c[3], kernel_size=3, stride=2, activation=None)
        self.block3 = BasicBlock(c[3], c[3])
        self.conv4 = ConvBlock(c[3], c[4], kernel_size=3, stride=2, activation=None)
        self.block4 = BasicBlock(c[4], c[4])
        self.conv4_tr = ConvTrBlock(c[4], tr[4], kernel_size=3, stride=2, activation=None)
        self.block4_tr = BasicBlock(tr[4], tr[4])
        self.conv3_tr = ConvTrBlock(c[3] + tr[4], tr[3], kernel_size=3, stride=2, activation=None)
        self.block3_tr = BasicBlock(tr[3], tr[3])
        self.conv2_tr = ConvTrBlock(c[2] + tr[3], tr[2], kernel_size=3, stride=2, activation=None)
        self.block2_tr = BasicBlock(tr[2], tr[2])
        self.conv1_tr = SparseConv3d(c[1] + tr[2], tr[1], kernel_size=1, stride=1, bias=False)
        self.relu = ReLU(inplace=True)
        self.final = SparseConv3d(tr[1], out_channels, kernel_size=1, stride=1, bias=True)
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = bn_momentum

    def forward(self, x: Voxels) -> Voxels:
        out_s1 = self.block1(self.conv1(x))
        out_s2 = self.block2(self.conv2(out_s1))
        out_s4 = self.block3(self.conv3(out_s2))
        out_s8 = self.block4(self.conv4(out_s4))
        out = self.block4_tr(self.conv4_tr(out_s8, out_s4))
        out = cat(out, out_s4)
        out = self.block3_tr(self.conv3_tr(out, out_s2))
        out = cat(out, out_s2)
        out = self.block2_tr(self.conv2_tr(out, out_s1))
        out = cat(out, out_s1)
        out = self.relu(self.conv1_tr(out))
        out = self.final(out)
        if self.normalize_feature:
            out = out.replace(batched_features=F.normalize(out.feature_tensor, p=2, dim=1))
        return out


class ResUNetBN2C(ResUNet2):
    pass
