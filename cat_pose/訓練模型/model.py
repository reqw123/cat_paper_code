import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_IMAGE_SIZE = 320

class BalancedMediumCNN(nn.Module):
    """
    BalancedMediumCNN (≈11~12 MB)
    -------------------------------
    - 改良通道配置: 32→64→128→256→384→512
    - 保留 BatchNorm + Dropout + 深分類頭
    - 控制模型大小 ≈ 12 MB
    """

    def __init__(self, num_classes=4):
        super(BalancedMediumCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # === 卷積主幹 ===
        self.layer1 = conv_block(3, 32)
        self.layer2 = conv_block(32, 64)
        self.layer3 = conv_block(64, 128)
        self.layer4 = conv_block(128, 256)
        self.layer5 = conv_block(256, 384)
        self.layer6 = conv_block(384, 512)  # 控制最大通道不超過512

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # === 分類頭 ===
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.layer1(x))
        x = self.maxpool(self.layer2(x))
        x = self.maxpool(self.layer3(x))
        x = self.maxpool(self.layer4(x))
        x = self.maxpool(self.layer5(x))
        x = self.layer6(x)                 # 不再 Pool，保留細節

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = BalancedMediumCNN(num_classes=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 * 1024)
    print(f"模型參數量: {total_params:,}")
    print(f"估計模型大小: {size_mb:.2f} MB")

    dummy = torch.randn(1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    out = model(dummy)
    print(f"輸出形狀: {out.shape}")
