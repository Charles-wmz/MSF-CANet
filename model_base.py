import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN1D(nn.Module):
    """基础1D CNN分类器"""
    
    def __init__(self, n_channels=132, n_classes=2, use_feature_extractor=False, tremor_freq_band=(3,7), fs=100):
        """
        初始化基础1D CNN分类器
        
        参数:
            n_channels (int): 输入通道数，默认132
            n_classes (int): 类别数，默认2
            use_feature_extractor (bool): 是否使用特征提取器，默认False（此模型没有特征提取器）
            tremor_freq_band (tuple): 震颤频带范围 (Hz)，默认(3, 7)
            fs (int): 信号采样率 (Hz)，默认100
        """
        super(ShallowCNN1D, self).__init__()
        
        # 基础模型不使用特征提取器，但保留此参数以保持接口一致
        self.use_feature_extractor = False
        
        # 卷积层
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 自适应池化层，输出固定大小的特征图
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数据，形状为 [batch_size, n_channels, time_points]
            
        返回:
            torch.Tensor: 分类输出，形状为 [batch_size, n_classes]
        """
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x 