from model.LocalNet.filter_estimator import FilterEstimator
from model.LocalNet.pwise_variants import PiecewiseBasis
from model.my_global_net import Global_pred_Same, Global_pred

class mobileIAT(torch.nn.Module):
    def __init__(self, in_channels=3, scale=1.0, expand_ratio=6, num_nodes=10, with_global=True):
        super(mobileIAT,self).__init__()
        
        # define enhancement module
        basis_param = num_nodes
        self.emodule = PiecewiseBasis(basis_param)
        self.with_global = with_global
        
        # define network
        input_channel = make_divisible(32 * scale)
        # 开始的一个卷积快用于映射特征
        self.conv1 = _conv_bn(in_channels=in_channels, out_channels=input_channel, kernel_size=3, 
                             stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 深度可分离卷积参数设置
        # 输入通道、输出通道，stride
        ShallowdepthSeparableConvSize = [
            # in out ,s, s
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1]
        ]
        
        DeepdepthSeparableConvSize = [
            # in out ,s, s
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2]
        ]

        conv2_sallow = []
        for i, o, s in ShallowdepthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2_sallow.append(InvertedResidualBottleNeck_S(input_channel, output_channel, 3, 
                                                    stride=s, padding=1, expand_ratio=expand_ratio))
            input_channel = output_channel

        self.conv2_sallow = nn.Sequential(*conv2_sallow)
        
        if self.with_global:
            self.params_pred = Global_pred(in_channels=output_channel, num_heads=4)
        
        conv2_deep = []
        for i, o, s in DeepdepthSeparableConvSize:
            output_channel = make_divisible(o * scale)
            # 加入可分离深度卷积层
            conv2_deep.append(InvertedResidualBottleNeck_S(input_channel, output_channel, 3, 
                                                    stride=s, padding=1, expand_ratio=expand_ratio))
            input_channel = output_channel

        self.conv2_deep = nn.Sequential(*conv2_deep)

        conv3 = []
        for i in range(5):
            # 加深度可分离卷积层，深度5
            conv3.append(InvertedResidualBottleNeck_S(input_channel, input_channel, 3, 
                                                    stride=1, padding=1, expand_ratio=expand_ratio))
        self.conv3 = nn.Sequential(*conv3)

        last_channel = make_divisible(1024 * scale)
        # 加入深度可分离卷积，深度1
        self.conv4 = nn.Sequential(
            InvertedResidualBottleNeck_S(input_channel, last_channel, 3, 
                                                    stride=2, padding=1, expand_ratio=expand_ratio),
            InvertedResidualBottleNeck_S(last_channel, last_channel, 3, 
                                                    stride=1, padding=1, expand_ratio=expand_ratio)
        )

        #池化层
        self.downsample = torch.nn.AdaptiveAvgPool2d(1)
        #线性预测层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(last_channel, last_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(last_channel, self.emodule.parameters_count)
        )
        
        self.fe = FilterEstimator(filter_type='single', kener_size=7, sigma=3)
        
        self.init_params()
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, image, applyto=None):
        x = image
        x = x - 0.5
        x =self.relu1(self.conv1(x))
        x = self.conv2_sallow(x)
        
        if self.with_global:
            gamma, color = self.params_pred(x)
        
        x = self.conv2_deep(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        applyto = (image if applyto is None else applyto)
        x = applyto + self.emodule(applyto, x)
        x = self.fe(x)
        return x, gamma, color