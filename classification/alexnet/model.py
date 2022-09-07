"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-09-07
"""
import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # ((224 + 4) - 11) // 4 + 1
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=(4, 4), padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            # (55 - 3 + 2*padding是0) / 2 + 1 = 27
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            # (27 + 2 * 2 -5) // 1 + 1 = 27
            nn.Conv2d(48, 128, kernel_size=(5, 5), padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            # (27 - 3) // 2 + 1 = 13
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]

            # (13 + 2 - 3) / 1 + 1 = 13
            nn.Conv2d(128, 192, kernel_size=(3, 3), padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=(3, 3), padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),

            # (13 - 3) // 2 + 1 = 6
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchinfo import summary
    print(summary(AlexNet(5), (4, 3, 224, 224)))

