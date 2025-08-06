import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))


class FFTNetBlock2d(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.feature_dim = channels * width
        self.filter = nn.Linear(self.feature_dim, self.feature_dim)
        self.modrelu = ModReLU(self.feature_dim)

    def forward(self, x):
        # x: [b, c, h, w]
        b, c, h, w = x.shape

        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)

        x_fft = torch.fft.fft(x, dim=1)

        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)

        x_filtered = self.modrelu(x_filtered)

        x_ifft = torch.fft.ifft(x_filtered, dim=1).real

        x_ifft = x_ifft.reshape(b, h, c, w).permute(0, 2, 1, 3)
        return x_ifft+x

if __name__ == "__main__":

    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)

    block = FFTNetBlock2d(channels, width)

    y = block(x)
