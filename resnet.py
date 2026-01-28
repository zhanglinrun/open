import torch
from torch import  nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:#  保证输入和输出在同一维度上（通道数一致） 最后才能进行相加的操作
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm1d(ch_out)
            )


    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        # print("conv1",out.shape)
        out = self.bn2(self.conv2(out))
        # print("conv2", out.shape)
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out
class GNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNetwork, self).__init__()
        
        # Define the layers
        self.fc = nn.Linear(input_dim, output_dim)       # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(output_dim)     # Batch normalization layer
        self.relu = nn.ReLU()                            # ReLU activation layer
        
    def forward(self, x):
        # Forward pass through the network
        x = self.fc(x)            # Apply the fully connected layer
        x = self.batch_norm(x)    # Apply batch normalization
        x = self.relu(x)          # Apply the activation function (ReLU)
        return x
class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=3, padding=0),#nn.Conv1d应该是向上取整 #2, 3, 128, 128=>2, 16, 42, 42
            nn.BatchNorm1d(16)
        )
        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        self.blk1 = ResBlk(16, 32, stride=3)#2, 16, 42, 42=>2, 32, 14, 14
        # [b, 32, h, w] => [b, 64, h, w]
        self.blk2 = ResBlk(32, 64, stride=2)#2, 32, 14, 14=>2, 64, 5, 5
        # # [b, 64, h, w] => [b, 128, h, w]
        self.blk3 = ResBlk(64, 128, stride=2)#2, 64, 5, 5=>2, 128, 3, 3
        # # [b, 128, h, w] => [b, 256, h, w]
        self.blk4 = ResBlk(128, 256, stride=2)#2, 128, 3, 3=>2, 256, 2, 2
        self.posconv = GNetwork(256,256)
        # [b, 256, 2, 2]
        #self.outlayer = nn.Linear(256*2*2,10)
        # self.fc = nn.Linear(256,256)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x)) # [b,16,341]
        # print('conv1:',x.shape)

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x) #
        # print('blk1', x.shape)
        x = self.blk2(x) # [32,32,14]
        # print('blk2', x.shape)
        x = self.blk3(x)
        # print('blk3', x.shape)
        x = self.blk4(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x_neg = x.squeeze(-1)
        x_pos = self.posconv(x_neg)
        # print('blk4', x.shape)
        return x_neg,x_pos

if __name__ == '__main__':
    input = torch.randn([100,2,1024])
    model = ResNet18()
    out = model(input)
    print(out.shape)
