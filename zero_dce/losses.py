import torch
import torch.nn as nn
from torch.nn import functional as F


class ColorConstancyLoss(nn.Module):
    """Color Constancy Loss"""

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mb - mg, 2)
        k = torch.pow(
            torch.pow(drg, 2) + torch.pow(drb, 2) + torch.pow(dgb, 2), 0.5)
        return k


class ExposureLoss(nn.Module):
    """Exposure Loss"""

    def __init__(self, patch_size, mean_val):
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        return torch.mean(torch.pow(
            mean - torch.FloatTensor([self.mean_val]).cuda(), 2
        ))


class IlluminationSmoothnessLoss(nn.Module):
    """Illumination Smoothing Loss"""

    def __init__(self, loss_weight=1):
        super(IlluminationSmoothnessLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SpatialConsistancyLoss(nn.Module):
    """Spatial Consistancy Loss"""

    def __init__(self):
        super(SpatialConsistancyLoss, self).__init__()

        kernel_left = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        d_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        d_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        d_left = torch.pow(d_org_left - d_enhance_left, 2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up = torch.pow(d_org_up - d_enhance_up, 2)
        d_down = torch.pow(d_org_down - d_enhance_down, 2)
        return d_left + d_right + d_up + d_down
