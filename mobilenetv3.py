from numpy.core.numeric import outer
import paddle
from paddle import nn

class DepthWise_Conv(nn.Layer):
    """深度卷积
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(DepthWise_Conv, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=in_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class PointWise_Conv(nn.Layer):
    """逐点卷积
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(PointWise_Conv, self).__init__()

        self.conv = nn.Conv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)
    
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class SEBlock(nn.Layer):
    """SE注意力机制
    """
    def __init__(self,
                 in_channels,
                 reduce=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_channels,
                             out_features=in_channels//reduce)
        
        self.fc2 = nn.Linear(in_features=in_channels//reduce,
                             out_features=in_channels)
        
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()
    
    def forward(self, inputs):
        x = self.avg_pool(inputs) # B, C, 1, 1
        x = self.flatten(x) # B, C
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hsigmoid(x)
        output = inputs * x.reshape(inputs.shape[:2]+[1, 1])
        return output


class BottleNeck(nn.Layer):
    """含SE通道注意力的瓶颈层
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 act=nn.ReLU,
                 use_se=False,
                 reduce=4):
        super(BottleNeck, self).__init__()
        self.use_se = use_se
        self.add_res = False if stride == 2 else \
                       False if in_channels!=out_channels else True

        self.in_pw = PointWise_Conv(in_channels=in_channels,
                                    out_channels=hidden_channels)
        self.in_pw_bn = nn.BatchNorm2D(hidden_channels)

        self.dw = DepthWise_Conv(in_channels=hidden_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)
        self.dw_bn = nn.BatchNorm2D(hidden_channels)

        if use_se == False:
            self.se = None
        else:
            self.se = SEBlock(in_channels=hidden_channels,
                              reduce=reduce)

        self.out_pw = PointWise_Conv(in_channels=hidden_channels,
                                     out_channels=out_channels)
        self.out_pw_bn = nn.BatchNorm2D(out_channels)

        self.act = act()

    def forward(self, inputs):
        x = self.in_pw(inputs)
        x = self.in_pw_bn(x)
        x = self.act(x)

        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.act(x)

        if self.use_se:
            x = self.se(x)

        x = self.out_pw(x)
        x = self.out_pw_bn(x)

        if self.add_res:
            x = x + inputs

        return x


class Stem(nn.Layer):
    """渐入层
    """
    def __init__(self,
                 in_channles,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0):
        super(Stem, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channles,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2D(out_channels)

        self.act = nn.Hardswish()
    
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.act(self.bn(x))
        return x


class Classifier_Head(nn.Layer):
    """分类头
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes=1000):
        super(Classifier_Head, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.conv1 = PointWise_Conv(in_channels=in_channels,
                                    out_channels=hidden_channels)
        self.conv1_act = nn.Hardswish()
        self.conv2 = PointWise_Conv(in_channels=hidden_channels,
                                    out_channels=num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
    
    def forward(self, inputs):
        x = self.avg_pool(inputs) # B, C, 1, 1
        x = self.conv1(x) # B, C, 1, 1
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.softmax(x)
        return x


class MobileNetV3_Large(nn.Layer):
    """MobileNetV3_Large实现
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 reduce=4,
                 alpha=1.0):
        super(MobileNetV3_Large, self).__init__()
        self.reduce = int(reduce / alpha)

        self.stem = Stem(in_channles=in_channels,
                         out_channels=int(alpha*16),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        
        self.bnecks_config = [
            # Params Info
            # in_channels, hidden_channels, out_channels,
            # kernel_size, stride, padding,
            # use_se, relu_type(nn.Layer)
            [16, 16, 16, 3, 1, 1, False, nn.ReLU], # 1 bneck
            [16, 64, 24, 3, 2, 1, False, nn.ReLU], # 2 bneck
            [24, 72, 24, 3, 1, 1, False, nn.ReLU],
            [24, 72, 40, 3, 2, 1, True, nn.ReLU],
            [40, 120, 40, 3, 1, 1, True, nn.ReLU],
            [40, 120, 40, 3, 1, 1, True, nn.ReLU],
            [40, 240, 80, 3, 2, 1, False, nn.Hardswish],
            [80, 200, 80, 3, 1, 1, False, nn.Hardswish],
            [80, 184, 80, 3, 1, 1, False, nn.Hardswish],
            [80, 184, 80, 3, 1, 1, False, nn.Hardswish],
            [80, 480, 112, 3, 1, 1, True, nn.Hardswish],
            [112, 672, 112, 3, 1, 1, True, nn.Hardswish],
            [112, 672, 160, 3, 2, 1, True, nn.Hardswish],
            [160, 960, 160, 3, 1, 1, True, nn.Hardswish],
            [160, 960, 160, 3, 1, 1, True, nn.Hardswish]
        ]

        self.bnecks = []
        for i in range(len(self.bnecks_config)):
            self.bnecks.append(
                BottleNeck(in_channels=int(alpha*self.bnecks_config[i][0]),
                           hidden_channels=int(alpha*self.bnecks_config[i][1]),
                           out_channels=int(alpha*self.bnecks_config[i][2]),
                           kernel_size=self.bnecks_config[i][3],
                           stride=self.bnecks_config[i][4],
                           padding=self.bnecks_config[i][5],
                           use_se=self.bnecks_config[i][6], # bool
                           act=self.bnecks_config[i][7],
                           reduce=self.reduce)
            )
        self.bnecks = nn.LayerList(self.bnecks)

        self.out_pw = PointWise_Conv(in_channels=int(alpha*self.bnecks_config[-1][2]),
                                     out_channels=960)
        
        self.head = Classifier_Head(in_channels=960,
                                    hidden_channels=1280,
                                    num_classes=num_classes)
    
    def forward(self, inputs):
        x = self.stem(inputs)
        
        for b in self.bnecks:
            x = b(x)
        
        x = self.out_pw(x)
        x = self.head(x)

        return x

"""
if __name__ == "__main__":
    model = MobileNetV3_Large(num_classes=1000,
                              in_channels=3,
                              reduce=4,
                              alpha=0.75)
    
    model = paddle.Model(model)
    model.summary(input_size=(1, 3, 224, 224))

"""
