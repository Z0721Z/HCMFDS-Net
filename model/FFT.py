import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU(nn.Module):
    """
    ModReLU 激活函数：对于复数输入 z，
    计算公式为 |z| * ReLU(cos(angle(z) + b))，
    其中 b 是可学习的偏置参数。
    """

    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        # x 为复数张量，计算其模长并结合相位信息进行激活
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))


class FFTNetBlock2d(nn.Module):
    """
    FFTNetBlock2d 模块：针对 4D 输入 [b, c, h, w]，
    选择在 h 维度上进行 FFT 处理。

    1. 将输入重排为 [b, h, c*w]，其中 h 作为序列长度，
       c*w 为特征维度。
    2. 沿 h 维度进行一维 FFT 变换。
    3. 对 FFT 后的实部和虚部分别使用同一个全连接层进行线性变换，
       得到复数形式的过滤结果。
    4. 应用 ModReLU 激活函数。
    5. 对处理后的结果沿 h 维度进行逆 FFT（IFFT），取其实部。
    6. 将张量恢复为原始的 [b, c, h, w] 形状。
    """

    def __init__(self, channels, width):
        super().__init__()
        # 定义特征维度：channels * width
        self.feature_dim = channels * width
        self.filter = nn.Linear(self.feature_dim, self.feature_dim)
        self.modrelu = ModReLU(self.feature_dim)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.shape

        # 将张量重排为 [b, h, c, w] 并将最后两个维度合并为特征维度
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)

        # 沿 h 维度计算 FFT
        x_fft = torch.fft.fft(x, dim=1)

        # 对实部和虚部分别进行线性变换，再组合为复数结果
        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)

        # 应用 ModReLU 激活函数
        x_filtered = self.modrelu(x_filtered)

        # 沿 h 维度进行逆 FFT，并取其实部
        x_ifft = torch.fft.ifft(x_filtered, dim=1).real

        # 将张量重新恢复为 [b, c, h, w]
        x_ifft = x_ifft.reshape(b, h, c, w).permute(0, 2, 1, 3)
        return x_ifft+x


# 示例：构建输入并测试 FFTNetBlock2d 模块
if __name__ == "__main__":
    # 创建一个随机输入张量，形状为 [batch_size, channels, height, width]
    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)

    # 实例化 FFTNetBlock2d 模块
    block = FFTNetBlock2d(channels, width)

    # 前向传播
    y = block(x)

    print("输入形状:", x.shape)
    print("输出形状:", y.shape)
