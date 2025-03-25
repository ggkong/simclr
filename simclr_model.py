import torch.nn as nn
import timm
import torch


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', projection_dim=128,
                 local_weight_path='efficientnet_b0.pth'):
        super().__init__()

        # 创建 EfficientNet 模型（不加载在线预训练权重）
        self.encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        # 从本地文件加载权重
        state_dict = torch.load(local_weight_path, map_location='cpu')

        # 如果本地权重包含了其他模块，可能需要调整 state_dict，这里假设直接适配
        self.encoder.load_state_dict(state_dict, strict=False)

        # 定义投影头
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

