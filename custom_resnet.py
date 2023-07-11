import torch.nn as nn
import torch.nn.functional as F

def apply_normalization(chennels):
  return nn.BatchNorm2d(chennels)


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        # Input Block
        drop = 0.0
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(64),
            nn.ReLU(),
        )
        # Layer1 -
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(128),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.reslayer1 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(128),
            nn.ReLU(),
        )
        # Conv 3x3 [256k]
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(256),
            nn.ReLU(),
        )
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            nn.MaxPool2d(2, 2),
            apply_normalization(512),
            nn.ReLU(),
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.reslayer2 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, stride=1, bias=False), # 3
            apply_normalization(512),
            nn.ReLU(),
        )
        self.maxpool3 = nn.MaxPool2d(4, 2)
        self.linear1 = nn.Linear(512,10)

    def forward(self,x):
        x = self.preplayer(x)
        x1 = self.convlayer1(x)
        x2 = self.reslayer1(x1)
        x = x1+x2
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x1 = self.reslayer2(x)
        x = x+x1
        x = self.maxpool3(x)
        x = x.view(-1, 512)
        x = self.linear1(x)
        return F.log_softmax(x, dim=-1)