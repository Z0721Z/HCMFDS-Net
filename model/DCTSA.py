import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.dct_filter import DCT8x8, DCT7x7, DCT3x3


class FreConv(nn.Module):
    def __init__(self, c, reduction, k=1, p=0):
        super(FreConv, self).__init__()
        if reduction == 1:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=k, padding=p, bias=False),
            )
        else:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, c // reduction, kernel_size=k, bias=False, padding=p),
                nn.ReLU(),
                nn.Conv2d(c // reduction, 1, kernel_size=k, padding=p, bias=False)
            )

    def forward(self, x):
        return self.freq_attention(x)


class DCTSA(nn.Module):
    def __init__(self, freq_num, channel, reduction=1, groups=1, select_method='all'):
        super(DCTSA, self).__init__()
        self.freq_num = freq_num
        self.channel = channel
        self.reduction = reduction
        self.select_method = select_method
        self.groups = groups

        # 选择 DCT 变换的窗口大小
        if freq_num == 64:
            self.dct_filter = DCT8x8()
        elif freq_num == 49:
            self.dct_filter = DCT7x7()
        elif freq_num == 9:
            self.dct_filter = DCT3x3()

        self.p = int((self.dct_filter.freq_range - 1) / 2)

        # 选择 DCT 频率通道数
        if self.select_method == 'all':
            self.dct_c = self.dct_filter.freq_num
        elif 's' in self.select_method:
            self.dct_c = 1
        elif 'top' in self.select_method:
            self.dct_c = int(self.select_method.replace('top', ''))

        self.freq_attention = FreConv(self.dct_c, reduction=reduction, k=7, p=3)
        self.sigmoid = nn.Sigmoid()

        # 通道注意力
        self.avg_pool_c = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool_c = nn.AdaptiveMaxPool2d((1, 1))
        self.register_parameter('alpha', nn.Parameter(torch.FloatTensor([0.5])))
        self.register_parameter('beta', nn.Parameter(torch.FloatTensor([0.5])))

        self.register_parameter('s', nn.Parameter(torch.FloatTensor([0.5])))

    def forward(self, x):
        b, c, h, w = x.shape  # 现在输入是 (B, C, H, W)

        # 计算通道注意力
        avg_map = self.avg_pool_c(x)  # (B, C, 1, 1)
        max_map = self.max_pool_c(x)  # (B, C, 1, 1)
        map_add = self.alpha * avg_map + self.beta * max_map  # (B, C, 1, 1)
        x_c = x * map_add + x  # 通道注意力加权

        # 计算 DCT 变换的空间注意力
        if self.select_method == 'all':
            dct_weight = self.dct_filter.filter.unsqueeze(1).repeat(1, self.channel, 1, 1)
        elif 's' in self.select_method:
            filter_id = int(self.select_method.replace('s', ''))
            dct_weight = self.dct_filter.get_filter(filter_id).unsqueeze(0).unsqueeze(0).repeat(1, self.channel, 1, 1)
        elif 'top' in self.select_method:
            filter_id = self.dct_filter.get_topk(self.dct_c)
            dct_weight = self.dct_filter.get_filter(filter_id).unsqueeze(1).repeat(1, self.channel, 1, 1)

        dct_bias = torch.zeros(self.dct_c).to(dct_weight.device)
        dct_feature = F.conv2d(x_c, dct_weight, dct_bias, stride=1, padding=self.p)

        dct_feature = self.freq_attention(dct_feature)  # (B, 1, H, W)
        dct_feature = dct_feature.repeat(1, c, 1, 1)  # 扩展回通道维度
        x_s = x_c * self.sigmoid(dct_feature) + x_c  # 结合空间注意力

        x = x_s * self.s  # 最终加权
        return x


if __name__ == "__main__":
    B, C, H, W = 1, 32, 256, 256  # 四维输入
    x = torch.randn(B, C, H, W).to('cuda')  # 创建随机输入
    model = DCTSA(freq_num=9, channel=32, reduction=1, groups=1, select_method='all').to('cuda')

    print(model)
    output = model(x)  # 运行模型
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
