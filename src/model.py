import math

import torch
from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
)


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.transpose(1, 2)
        v = self.proj(v)
        v = v.transpose(1, 2)
        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, dim, mlp_dim, seq_len, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.channel_proj_in = nn.Linear(dim, mlp_dim * 2)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(mlp_dim, seq_len)
        self.channel_proj_out = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.channel_proj_in(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.channel_proj_out(x)
        x = self.dropout(x)
        return x + residual


def _eca_kernel_size(channels, gamma=2, b=1):
    kernel_size = int(abs((math.log2(channels) + b) / gamma))
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


class ECALayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_size = _eca_kernel_size(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.activation(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class AttentionResidualBlock(nn.Module):
    def __init__(self, block, attention_layer):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = block.downsample
        self.stride = block.stride
        self.has_bottleneck = hasattr(block, "conv3")
        if self.has_bottleneck:
            self.conv3 = block.conv3
            self.bn3 = block.bn3
            out_channels = block.bn3.num_features
        else:
            out_channels = block.bn2.num_features
        self.attention = attention_layer(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.has_bottleneck:
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)

        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _replace_attention_blocks(module, attention_layer):
    for name, child in module.named_children():
        if child.__class__.__name__ in {"BasicBlock", "Bottleneck"}:
            setattr(module, name, AttentionResidualBlock(child, attention_layer))
        else:
            _replace_attention_blocks(child, attention_layer)


class Resnet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        model_name="resnet18",
        freeze=True,
        pretrained=True,
        dropout=0.3,
        image_size=256,
    ):
        super(Resnet, self).__init__()
        variant = "baseline"
        backbone_name = model_name
        for prefix in ("eca_", "se_", "gmlp_"):
            if model_name.startswith(prefix):
                variant = prefix[:-1]
                backbone_name = model_name[len(prefix) :]
                break

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.model = resnet34(weights=weights)
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
        elif backbone_name == "resnext50_32x4d":
            weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            self.model = resnext50_32x4d(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if variant == "eca":
            _replace_attention_blocks(self.model, ECALayer)
        elif variant == "se":
            _replace_attention_blocks(self.model, SELayer)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if variant == "gmlp":
            self.feature_extractor = nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool,
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            )
            self.model.fc = nn.Identity()
            feature_dim = self.model.layer4[-1].conv3.out_channels
            token_grid = max(1, math.ceil(image_size / 32))
            seq_len = token_grid * token_grid
            gmlp_dim = 512
            gmlp_hidden_dim = 1024
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, gmlp_dim),
            )
            self.gmlp_blocks = nn.Sequential(
                GMLPBlock(gmlp_dim, gmlp_hidden_dim, seq_len, dropout=dropout),
                GMLPBlock(gmlp_dim, gmlp_hidden_dim, seq_len, dropout=dropout),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(gmlp_dim),
                nn.Dropout(dropout),
                nn.Linear(gmlp_dim, num_classes),
            )
            self.variant = variant
        else:
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
            self.variant = variant

    def forward(self, x):
        if self.variant == "gmlp":
            x = self.feature_extractor(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.head(x)
            x = self.gmlp_blocks(x)
            x = x.mean(dim=1)
            return self.classifier(x)
        return self.model(x)

    def parameter_groups(self, head_lr, backbone_lr_scale=0.1):
        if self.variant == "gmlp":
            head_params = (
                list(self.head.parameters())
                + list(self.gmlp_blocks.parameters())
                + list(self.classifier.parameters())
            )
        else:
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
