import torch.nn as nn
import torch


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=16):
        super(GAM_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // rate, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels // rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // rate, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x).sigmoid()
        out = x * spatial_attention
        return out


# if __name__ == '__main__':
#     x = torch.randn(3, 64, 112, 112)
#     b, c, h, w = x.shape
#     net = GAM_Attention(in_channels=c, out_channels=c)
#     y = net(x)
#     print(net)
