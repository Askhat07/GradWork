# import torch
# import torch.nn as nn
#
# class BasicBlock(nn.Module):
#     def init(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).init()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out
#
# class res_net_9(nn.Module):
#     def init(self, in_channels, num_classes):
#         super(res_net_9, self).init()
#         self.prep = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.layer1 = BasicBlock(64, 128, stride=2)
#         self.layer2 = BasicBlock(128, 256, stride=2)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         x = self.prep(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ImageClassificationBase(nn.Module):
    def trainingStep(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validationStep(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validationEpochEnd(self, outputs):
        losses = [x["val_loss"] for x in outputs]
        accs = [x["val_acc"] for x in outputs]
        return {
            "val_loss": torch.stack(losses).mean(),
            "val_acc": torch.stack(accs).mean()
        }


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_one = conv_block(in_channels, 64)
        self.conv_two = conv_block(64, 128, pool=True)
        self.res_one = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv_three = conv_block(128, 256, pool=True)
        self.conv_four = conv_block(256, 512, pool=True)
        self.res_two = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv_one(xb)
        out = self.conv_two(out)
        out = self.res_one(out) + out
        out = self.conv_three(out)
        out = self.conv_four(out)
        out = self.res_two(out) + out
        out = self.classifier(out)
        return out



def load_model(weight_path, num_classes, device):
    model = ResNet9(3, num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model