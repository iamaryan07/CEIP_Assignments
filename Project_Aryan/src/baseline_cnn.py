"""Deliverable 2 -- scratch-trained 3-layer CNN baseline.

This is intentionally simple: three conv blocks (conv -> BN -> ReLU -> maxpool)
followed by a 2-layer classifier head. No pretraining, no transfer learning.
Its job is to set the floor that the transfer-learning model in
transfer_model.py must beat.
"""
import torch.nn as nn

from . import config


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = config.NUM_EUROSAT_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
