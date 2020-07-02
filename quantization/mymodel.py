import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_ultra import *

def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

class YOLOLayer(nn.Module):
    def __init__(self, anchors):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.no = 6  # number of outputs
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
    def forward(self, p, img_size):
        
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride  # 原始像素尺度

            
            torch.sigmoid_(io[..., 4:])
            
            
            return io.view(bs, -1, self.no), p

class UltraNetQua(nn.Module):
    def __init__(self):
        super(UltraNetQua, self).__init__()
        W_BIT = 4
        A_BIT = 4
        conv2d_q = conv2d_Q_fn(W_BIT)
        # act_q = activation_quantize_fn(4)

        self.layers = nn.Sequential(
            conv2d_q(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            activation_quantize_fn(A_BIT),
            nn.MaxPool2d(2, stride=2),

            conv2d_q(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            nn.MaxPool2d(2, stride=2),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),

            # conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # activation_quantize_fn(A_BIT),

            # conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            # activation_quantize_fn(A_BIT),

            conv2d_q(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            activation_quantize_fn(A_BIT),


            # nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)
            conv2d_q(64, 36, kernel_size=1, stride=1, padding=0)
            
        )
        self.yololayer = YOLOLayer([[20,20], [20,20], [20,20], [20,20], [20,20], [20,20]])
        self.yolo_layers = [self.yololayer]

    def forward(self, x):
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        x = self.layers(x)
        x = self.yololayer(x, img_size)

        yolo_out.append(x)

        if self.training:  # train
            return yolo_out
        else:  # test
            io, p = zip(*yolo_out)  # inference output, training output
            return torch.cat(io, 1), p
        return x  