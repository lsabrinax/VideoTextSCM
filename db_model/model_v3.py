import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from .resnet import deformable_resnet50

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input) # T * B * (D*Hout)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class SegDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(
            inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, features, gt=None, masks=None):
        c2, c3, c4, c5 = features

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)
        fuse = torch.cat((p5, p4, p3, p2), 1)  #shaep=(1,256,h/4,w/4)

        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)
        return binary, fuse

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        self.backbone = deformable_resnet50()
        self.decoder = SegDetector(adaptive=True, in_channels=[256, 512, 1024, 2048], k=50)

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)

class SegDetectorModel(nn.Module):
    def __init__(self, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        self.model = BasicModel()

    def forward(self, data):
        data = data.float()
        pred, feat = self.model(data)
        return pred, feat

class DB_Embedding_Model(nn.Module):
    def __init__(self, nh=256, nclass=16):
        super(DB_Embedding_Model, self).__init__()
        self.db = SegDetectorModel()
        self.roi_align = RoIAlign((5, 16), 0.25, 0)
        self.extra_conv_layers = nn.Sequential(
                                        nn.Conv2d(256, 256, 3, padding=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 3, padding=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))

        self.embed_layers = nn.Sequential(nn.Linear(256*5*16,1024),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(1024,256)
                                            )

        self.seq_conv = nn.Conv2d(256,256,kernel_size=(5,3), padding=(0,1))
        self.rnn = nn.Sequential(BidirectionalLSTM(256, nh, nh),BidirectionalLSTM(nh, nh, nclass))

    def forward(self, data, all_boxes=None):
        pred, feat = self.db(data)
        feat_refine = self.extra_conv_layers(feat.detach())
        if all_boxes is None:
            return pred, feat_refine
        
        roi_feat = self.roi_align(feat_refine, all_boxes)
        ### Visual feat
        obj_num = roi_feat.shape[0]
        v_roi_feat = roi_feat.view(obj_num, -1)
        v_roi_feat = self.embed_layers(v_roi_feat)

        # Semantic feat
        s_roi_feat = self.seq_conv(roi_feat)
        s_roi_feat = s_roi_feat.squeeze(2).permute(2, 0, 1)
        s_roi_feat = self.rnn(s_roi_feat).permute(1, 2, 0).contiguous().view(obj_num, -1)
        
        roi_feat = torch.cat((v_roi_feat, s_roi_feat), -1)
        # roi_feat = s_roi_feat
        return pred, roi_feat







