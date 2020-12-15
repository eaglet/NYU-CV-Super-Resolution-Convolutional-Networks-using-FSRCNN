from torch import nn
import math


class SRCNN(nn.Module):
    def __init__(self, channel_number=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size = 9, padding =9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 5, padding = 5 // 2)
        self.conv3 = nn.Conv2d(32, channel_number, kernel_size = 5, padding =5 // 2)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, channel_number=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(channel_number, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )

        layer = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            layer.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        layer.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*layer)

        self.last_part = nn.ConvTranspose2d(d, channel_number, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

class FSRCNN_S1(nn.Module):
    def __init__(self, scale_factor, channel_number=1):
        super(FSRCNN_S1, self).__init__()
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

        self.last_part = nn.ConvTranspose2d(32, channel_number, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.last_part(x)
        return x

class FSRCNN_S2(nn.Module):
    def __init__(self, scale_factor, channel_number=1):
        super(FSRCNN_S2, self).__init__()
        self.conv1 = nn.Conv2d(channel_number, 64, kernel_size=9, padding=9 // 2)
        self.relu = nn.ReLU(inplace=True)

        layer = [nn.Conv2d(64, 12, kernel_size=1), nn.PReLU(12)]
        for _ in range(4):
            layer.extend([nn.Conv2d(12, 12, kernel_size=3, padding=3 // 2), nn.PReLU(12)])
        layer.extend([nn.Conv2d(12, 64, kernel_size=1), nn.PReLU(64)])
        self.mid_part = nn.Sequential(*layer)


        self.last_part = nn.ConvTranspose2d(64, channel_number, kernel_size=9, stride=scale_factor, padding=9 // 2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.mid_part(x)
        x = self.last_part(x)
        return x