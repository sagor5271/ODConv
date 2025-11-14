import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================
# STABLE ATTENTION
# ============================
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super().__init__()

        attn_ch = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, attn_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(attn_ch)
        self.relu = nn.ReLU(inplace=True)

        # channel attention
        self.fc_channel = nn.Conv2d(attn_ch, in_planes, 1)

        # filter attention
        if in_planes == groups and in_planes == out_planes:
            self.fc_filter = None
        else:
            self.fc_filter = nn.Conv2d(attn_ch, out_planes, 1)

        # spatial attention
        if kernel_size > 1:
            self.fc_spatial = nn.Conv2d(attn_ch, kernel_size * kernel_size, 1)
        else:
            self.fc_spatial = None

        # kernel attention
        if kernel_num > 1:
            self.fc_kernel = nn.Conv2d(attn_ch, kernel_num, 1)
        else:
            self.fc_kernel = None

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.size()
        x_avg = self.avg(x)
        x_fc = self.relu(self.bn(self.fc1(x_avg)))

        # channel attention
        ch = torch.sigmoid(self.fc_channel(x_fc) / (self.temperature * 2))
        ch = torch.clamp(ch, 0.01, 0.99)

        # filter attention
        if self.fc_filter is None:
            fl = 1.0
        else:
            fl = torch.sigmoid(self.fc_filter(x_fc) / (self.temperature * 2))
            fl = torch.clamp(fl, 0.01, 0.99)

        # spatial attention
        if self.fc_spatial is None:
            sp = 1.0
        else:
            sp = self.fc_spatial(x_fc)
            sp = sp.view(B, 1, 1, 1, self.kernel_size, self.kernel_size)
            sp = torch.sigmoid(sp / (self.temperature * 2))
            sp = torch.clamp(sp, 0.01, 0.99)

        # kernel attention
        if self.fc_kernel is None:
            ke = 1.0
        else:
            ke = self.fc_kernel(x_fc)
            ke = ke.view(B, self.kernel_num, 1, 1, 1, 1)
            ke = F.softmax(ke / (self.temperature * 2), dim=1)

        return ch, fl, sp, ke

# ============================
# STABLE ODConv2d
# ============================
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, reduction=0.0625, kernel_num=4):
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.attn = Attention(in_planes, out_planes, kernel_size,
                              groups=groups, reduction=reduction, kernel_num=kernel_num)

        # weights (scaled)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size) * 0.05
        )

    def forward(self, x):
        B, C, H, W = x.size()
        ch, fl, sp, ke = self.attn(x)

        # channel apply
        x = x * ch
        x = x.reshape(1, B * C, H, W)

        # convert float to tensor if necessary
        device = x.device
        dtype = x.dtype
        if isinstance(sp, float):
            sp = torch.ones(B, 1, 1, 1, self.kernel_size, self.kernel_size, device=device, dtype=dtype)
        if isinstance(ke, float):
            ke = torch.ones(B, self.kernel_num, 1, 1, 1, 1, device=device, dtype=dtype)

        # dynamic weight fusion
        agg_w = sp * ke * self.weight.unsqueeze(0)
        agg_w = agg_w.sum(1)  # sum kernels
        agg_w = agg_w / self.kernel_num  # normalize scale

        agg_w = agg_w.view(B * self.out_planes,
                           self.in_planes // self.groups,
                           self.kernel_size,
                           self.kernel_size)

        # convolution
        out = F.conv2d(
            x, agg_w, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * B
        )

        out = out.view(B, self.out_planes, out.size(-2), out.size(-1))

        # filter attention
        if isinstance(fl, float):
            fl = torch.ones_like(out, device=device, dtype=dtype)
        out = out * fl

        return out
