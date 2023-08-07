import numpy as np
import torch
import torch.nn as nn
from meg_decoding.matlab_utils.load_meg import roi


class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        # 0.2-0.4は24だが0.35-0.15は23.9999
        # T = int((args.window.end - args.window.start) * args.preprocs.brain_resample_rate)
        T = int(np.round((args.window.end - args.window.start) * args.preprocs.brain_resample_rate))
        # DownSampling
        # self.down1 = nn.Sequential(nn.AvgPool2d((1, p0)))

        # Conv2d(in,out,kernel,stride,padding,bias) #k1 30
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.F1, (1, args.k1), padding="same", bias=False), nn.BatchNorm2d(args.F1)
        )
        roi_channels = roi(args)
        if "src_reconstruction" in args.keys() and args.src_reconstruction:
            num_channels = args.src_ch #449 # hard coding
        else:
            num_channels = len(roi_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                args.F1, args.D * args.F1, (num_channels, 1), groups=args.F1, bias=False
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.ELU(),
            nn.AvgPool2d((1, args.p1)), # 2
            nn.Dropout(args.dr1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (1, args.k2), # 4
                padding="same",
                groups=args.D * args.F1,
                bias=False,
            ),
            nn.Conv2d(args.D * args.F1, args.F2, (1, 1), bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(),
            nn.AvgPool2d((1, args.p2)), # 4
            nn.Dropout(args.dr2),
        )

        self.n_dim = self.compute_dim(num_channels, T)
        out_features = 512 if not 'out_features' in args.keys() else args['out_features']
        self.classifier = nn.Linear(self.n_dim, out_features, bias=True)
        # self.classifier = nn.Linear(self.n_dim, 512, bias=True)

    def forward(self, x, sbj_idxs):
        x = x.unsqueeze(1)
        # 1, 1, 128, 300
        # x = self.down1(x)
        x = self.conv1(x) # 1, 16, 128, 300
        x = self.conv2(x) # 1, 32, 1, 150
        x = self.conv3(x) # 1, 32, 1, 37
        x = x.view(-1, self.n_dim)
        x = self.classifier(x)
        return x

    def compute_dim(self, num_channels, T):
        x = torch.zeros((1, 1, num_channels, T))

        # x = self.down1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x.size()[1] * x.size()[2] * x.size()[3]
