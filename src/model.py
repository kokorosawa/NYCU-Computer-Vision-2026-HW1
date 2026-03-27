from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
)


class Resnet(nn.Module):
    def __init__(self, num_classes=1000, model_name="resnet18", freeze=True):
        super(Resnet, self).__init__()
        if model_name == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == "resnet34":
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_name == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def parameter_groups(self, head_lr, backbone_lr_scale=0.1):
        head_params = list(self.model.fc.parameters())
        backbone_params = [
            param
            for name, param in self.model.named_parameters()
            if not name.startswith("fc.") and param.requires_grad
        ]

        if not backbone_params:
            return [{"params": head_params, "lr": head_lr}]

        return [
            {"params": backbone_params, "lr": head_lr * backbone_lr_scale},
            {"params": head_params, "lr": head_lr},
        ]
