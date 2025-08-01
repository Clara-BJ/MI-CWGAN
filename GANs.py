import torch.nn as nn
import torch.nn.functional as F
import torch


class Multi_Generator(nn.Module):

    class laten(nn.Module):
        def __init__(self, nz, eeg_out, fnirs_out):
            super().__init__()
            self.ll = nn.Linear(nz, 1024)
            self.lll = nn.Linear(1024, eeg_out + fnirs_out)
            self.lr = nn.LeakyReLU()
            nn.init.normal_(self.lll.weight.data, 0.0, 0.02)
            nn.init.normal_(self.ll.weight.data, 0.0, 0.02)

        def forward(self, x):
            x = self.ll(x)
            x = self.lr(x)
            x = self.lll(x)
            eeg = x[:,:4096]
            fnirs = x[:,4096:]
            return eeg, fnirs

    class eeg_generate(nn.Module):
        def __init__(self, ):
            super().__init__()
            self.upsample1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [64, 8, 32]
            self.conv1 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)  # [32, 8, 32]

            self.upsample2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [32, 16, 64]
            self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)  # [16, 16, 64]
            self.bn2 = nn.BatchNorm2d(16)

            self.upsample3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [16, 32, 128]
            self.conv3 = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2)  # [8, 32, 128]
            self.bn3 = nn.BatchNorm2d(8)

            self.upsample4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [8, 64, 256]
            self.conv4 = nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2)  # [4, 64, 256]
            self.bn4 = nn.BatchNorm2d(4)

            self.upsample5 = nn.Upsample(scale_factor=(1, 4), mode='bilinear', align_corners=False)  # [4, 64, 1024]
            self.conv5 = nn.Conv2d(4, 4, kernel_size=(1, 25), stride=1, padding=(0, 0))  # [1, 64, 1000]
            self.bn5 = nn.BatchNorm2d(4)

            self.conv6 = nn.Conv2d(4, 1, kernel_size=(3, 1), stride=1, padding=(0, 0))  # [1, 64, 1000]

        def forward(self, h):
            h = self.upsample1(h)  # [64, 8, 32]
            h = F.relu(self.conv1(h))  # [32, 8, 32]
            h = self.upsample2(h)  # [32, 16, 64]
            h = F.relu(self.bn2(self.conv2(h)))  # [16, 16, 64]
            h = self.upsample3(h)  # [16, 32, 128]
            h = F.relu(self.bn3(self.conv3(h)))  # [8, 32, 128]
            h = self.upsample4(h)  # [8, 64, 256]
            h = F.relu(self.bn4(self.conv4(h)))  # [4, 64, 256]
            h = self.upsample5(h)  # [4, 64, 1024]
            h = F.relu(self.bn5(self.conv5(h)))  # [1, 64, 1000]
            h = F.tanh(self.conv6(h))
            h = h.squeeze(1)
            return h

    class fnirs_generate(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [64, 8, 32]
            self.conv1 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)  # [32, 8, 32]

            self.upsample2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [32, 16, 64]
            self.conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)  # [16, 16, 64]
            self.bn2 = nn.BatchNorm2d(16)

            self.upsample3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [16, 32, 128]
            self.conv3 = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2)  # [8, 32, 128]
            self.bn3 = nn.BatchNorm2d(8)

            self.upsample4 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False)  # [8, 64, 256]
            self.conv4 = nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2)  # [4, 64, 256]
            self.bn4 = nn.BatchNorm2d(4)

            self.conv5 = nn.Conv2d(4, 4, kernel_size=(1, 10), stride=(1, 2), padding=(0, 4))  # [1, 64, 1000]
            self.bn5 = nn.BatchNorm2d(4)

            self.conv6 = nn.Conv2d(4, 1, kernel_size=(17, 1), stride=1, padding=(0, 0))  # [1, 64, 1000]
        def forward(self, h):
            h = self.upsample1(h)  # [64, 8, 32]
            h = F.relu(self.conv1(h))  # [32, 8, 32]
            h = self.upsample2(h)  # [32, 16, 64]
            h = F.relu(self.bn2(self.conv2(h)))  # [16, 16, 64]
            h = self.upsample3(h)  # [16, 32, 128]
            h = F.relu(self.bn3(self.conv3(h)))  # [8, 32, 128]
            h = self.upsample4(h)  # [8, 64, 256]
            h = F.relu(self.bn4(self.conv4(h)))  # [4, 64, 256]
            h = F.relu(self.bn5(self.conv5(h)))  # [1, 64, 1000]
            h = F.tanh(self.conv6(h))
            h = h.squeeze(1)
            return h

    def weight_init(self, mean, std):
        def normal_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(normal_init)

    def __init__(self,  n_classes=5):
        super(Multi_Generator, self).__init__()
        self.errG_array = []
        self.count = 0
        self.nz = 300 + n_classes

        self.l1 = self.laten(self.nz, 4096, 1024)

        self.eeg_g = self.eeg_generate()
        self.fnirs_g = self.fnirs_generate()

        self.weight_init(0, 0.02)

    def forward(self, x, label):
        x = torch.concat([x, label], dim=1)
        eeg, fnirs = self.l1(x)
        eeg = eeg.view(x.shape[0],64,4,16)
        fnirs = fnirs.view(x.shape[0],64,4,4)

        eeg = self.eeg_g(eeg)
        fnirs = self.fnirs_g(fnirs)

        return [eeg, fnirs]


class Multi_Discriminator(nn.Module):

    class eeg_discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.F1 = 8
            self.D = 2
            self.F2 = self.F1 * self.D

            self.kernel_length = 65
            self.avg_pool1_size = 8
            self.dconv2_size = 33
            self.avg_pool2_size = 16

            self.fc_max_norm = 0.25

            self.n_electrodes = 62
            self.n_classes = 1
            self.dropout_p = 0.5

            # Block 1
            self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernel_length),
                                           stride=1,
                                           padding=(0, self.kernel_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            self.spatial_dconv = nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D,
                                                      kernel_size=(self.n_electrodes, 1),
                                                      stride=1, padding=(0, 0), groups=self.F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
            self.elu1 = nn.ELU()
            self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, self.avg_pool1_size), stride=(1, self.avg_pool1_size))
            self.drop1 = nn.Dropout(p=self.dropout_p)

            self.dconv = nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F1 * self.D,
                                   kernel_size=(1, self.dconv2_size),
                                   stride=1,
                                   padding=(0, self.dconv2_size // 2), bias=False,
                                   groups=self.F1 * self.D)
            self.pconv = nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F2,
                                   kernel_size=(1, 1),
                                   stride=1,
                                   padding=(0, 0), bias=False,
                                   groups=1)
            self.bn3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
            self.elu2 = nn.ELU()
            self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, self.avg_pool2_size), stride=(1, self.avg_pool2_size))
            self.drop2 = nn.Dropout(p=self.dropout_p)

            self.fc = nn.Linear(self.F2 * 7, 1, bias=True)
        def forward(self, x):
            n_trials, n_channels, n_times = x.shape
            # Block 1
            y = self.temporal_conv(x.view(n_trials, 1, n_channels, n_times))
            y = self.bn1(y)
            y = self.spatial_dconv(y)
            y = self.bn2(y)
            y = self.elu1(y)
            y = self.avg_pool1(y)
            y = self.drop1(y)

            # Block 2
            y = self.dconv(y)
            y = self.pconv(y)
            y = self.bn3(y)
            y = self.elu2(y)
            y = self.avg_pool2(y)

            y1 = self.fc(y.contiguous().view(y.shape[0], -1))
            return y1 ,y

    class fnirs_discriminator(nn.Module):
        def __init__(self):
            super().__init__()

            self.F1 = 8
            self.D = 2
            self.F2 = self.F1 * self.D

            self.kernel_length = 8
            self.avg_pool1_size = 3
            self.dconv2_size = 2
            self.avg_pool2_size = 3

            self.fc_max_norm = 0.25

            self.n_electrodes = 48
            self.n_classes = 1
            self.dropout_p = 0.5

            # Block 1
            self.temporal_conv = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernel_length),
                                           stride=1,
                                           padding=(0, self.kernel_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
            self.spatial_dconv = nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D,
                                                      kernel_size=(self.n_electrodes, 1),
                                                      stride=1, padding=(0, 0), groups=self.F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
            self.elu1 = nn.ELU()
            self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, self.avg_pool1_size), stride=(1, self.avg_pool1_size))
            self.drop1 = nn.Dropout(p=self.dropout_p)

            # Block 2
            self.dconv = nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F1 * self.D,
                                   kernel_size=(1, self.dconv2_size),
                                   stride=1,
                                   padding=(0, self.dconv2_size // 2), bias=False,
                                   groups=self.F1 * self.D)
            self.pconv = nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F2,
                                   kernel_size=(1, 1),
                                   stride=1,
                                   padding=(0, 0), bias=False,
                                   groups=1)
            self.bn3 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
            self.elu2 = nn.ELU()
            self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, self.avg_pool2_size), stride=(1, self.avg_pool2_size))
            self.drop2 = nn.Dropout(p=self.dropout_p)
            self.fc = nn.Linear(self.F2 * 4, 1, bias=True)

        def forward(self, x):
            n_trials, n_channels, n_times = x.shape
            # Block 1
            y = self.temporal_conv(x.view(n_trials, 1, n_channels, n_times))
            y = self.bn1(y)
            y = self.spatial_dconv(y)
            y = self.bn2(y)
            y = self.elu1(y)
            y = self.avg_pool1(y)
            y = self.drop1(y)

            # Block 2
            y = self.dconv(y)
            y = self.pconv(y)
            y = self.bn3(y)
            y = self.elu2(y)
            y = self.avg_pool2(y)
            y1 = self.fc(y.contiguous().view(y.shape[0], -1))
            return y1, y

    def weight_init(self, mean, std):
        def normal_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(normal_init)

    class End(nn.Module):
        def __init__(self, n_classes, separate=True):
            super().__init__()
            self.separate = separate
            self.eeg_label = nn.Linear(n_classes, 10)
            self.fnirs_label = nn.Linear(n_classes, 10)
            self.joint_labl = nn.Linear(n_classes, 10)
            self.eeg_end1 = nn.Linear(112+10, 1)
            self.fnirs_end1 = nn.Linear(64+10, 1)
            self.joint_end = nn.Linear(64 + 112 + 10, 1)

        def forward(self, eeg, fnirs, label):
            eeg_label = self.eeg_label(label)
            eeg = eeg.view(eeg.shape[0], -1)

            fnirs_label = self.fnirs_label(label)
            fnirs = fnirs.view(fnirs.shape[0], -1)

            joint_label = self.joint_label(label)
            joint = torch.concat([eeg, fnirs, joint_label], dim=1)
            joint = self.joint_end(joint)

            eeg = torch.concat([eeg, eeg_label], dim=1)
            eeg = self.eeg_end1(eeg)

            fnirs = torch.concat([fnirs, fnirs_label], dim=1)
            fnirs = self.fnirs_end1(fnirs)

            return [eeg, fnirs, joint]


    def __init__(self,n_classes=5):
        super().__init__()
        self.eeg_d = self.eeg_discriminator()
        self.fnirs_d = self.fnirs_discriminator()
        self.end = self.End(n_classes=n_classes, separate=True)
        self.weight_init(0, 0.02)

    def forward(self, x, label):
        eeg, eeg_ = self.eeg_d(x[0])
        fnirs, fnirs_ = self.fnirs_d(x[1])
        return self.end(eeg_, fnirs_, label)