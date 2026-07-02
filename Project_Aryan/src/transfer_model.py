"""Transfer-learning backbone: ResNet-18 pretrained on ImageNet.

ResNet-18 is the project's chosen backbone (over EfficientNet-B0) because its
512-dim penultimate feature vector matches the change-detection module's
required embedding size exactly, with no extra projection layer.

Two-phase fine-tuning, as specified in the brief:
  Phase 1 -- freeze everything except the new classifier head. Train 3 epochs.
  Phase 2 -- unfreeze layer3 + layer4 (the last two conv blocks), drop the
             learning rate 10x, train 5 more epochs.
"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from . import config


def build_resnet18(num_classes: int = config.NUM_EUROSAT_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: freeze every parameter except the classifier head (model.fc)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_last_blocks(model: nn.Module) -> None:
    """Phase 2: additionally unfreeze layer3 and layer4 (the last two
    residual/conv blocks), keeping the stem + layer1 + layer2 frozen."""
    for name, param in model.named_parameters():
        if name.startswith("fc.") or name.startswith("layer3.") or name.startswith("layer4."):
            param.requires_grad = True
        else:
            param.requires_grad = False


def trainable_param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    """Two LR groups: head at `head_lr`, unfrozen backbone blocks at
    `backbone_lr` (used in phase 2, backbone_lr = head_lr / 10)."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups


class EmbeddingExtractor(nn.Module):
    """Wraps a fine-tuned ResNet-18 classifier and exposes both:
      - a 512-dim global embedding (post global-avg-pool), used for the
        per-tile cosine-similarity change score and the ROC analysis.
      - a (512, 7, 7) spatial feature map (pre-pool), used to render the
        per-patch change heatmap.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.conv_stem = nn.Sequential(
            classifier.conv1, classifier.bn1, classifier.relu, classifier.maxpool,
            classifier.layer1, classifier.layer2, classifier.layer3, classifier.layer4,
        )
        self.avgpool = classifier.avgpool

    def forward(self, x):
        """Global 512-dim embedding."""
        fmap = self.conv_stem(x)
        return self.avgpool(fmap).flatten(1)

    def forward_feature_map(self, x):
        """Spatial (batch, 512, 7, 7) feature map for heatmaps."""
        return self.conv_stem(x)
