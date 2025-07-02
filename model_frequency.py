import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAwareModule(nn.Module):
    """频域感知模块，特别关注帕金森病的震颤频带"""
    
    def __init__(self, in_channels, out_channels, tremor_freq_band=(3, 7), fs=100):
        super(FrequencyAwareModule, self).__init__()
        self.tremor_freq_band = tremor_freq_band
        self.fs = fs
        
        # 特征映射
        self.mapping = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数据，形状为 [batch_size, in_channels, time_points]
            
        返回:
            torch.Tensor: 频域增强的特征，形状为 [batch_size, out_channels, time_points]
        """
        batch_size, channels, time_points = x.shape
        
        # 计算FFT
        x_fft = torch.fft.rfft(x, dim=2)
        freqs = torch.fft.rfftfreq(time_points, d=1.0/self.fs, device=x.device)
        
        # 创建震颤频带掩码
        mask = ((freqs >= self.tremor_freq_band[0]) & (freqs <= self.tremor_freq_band[1])).float()
        
        # 增强震颤频带
        enhanced_fft = x_fft.clone()
        enhanced_fft *= (1.0 + mask.view(1, 1, -1) * 0.5)  # 震颤频带增强50%
        
        # 转回时域
        enhanced_x = torch.fft.irfft(enhanced_fft, n=time_points, dim=2)
        
        # 特征映射
        features = self.mapping(enhanced_x)
        features = self.bn(features)
        features = F.relu(features)
        
        return features


class MSHFeatureExtractor(nn.Module):
    """
    多尺度分层特征提取器 (Multi-Scale Hierarchical Feature Extractor)
    
    该特征提取器中只启用频域感知模块，特别关注帕金森病震颤频带
    """
    
    def __init__(self, in_channels=132, out_channels=132, tremor_freq_band=(3, 7), fs=100):
        """
        初始化多尺度分层特征提取器
        
        参数:
            in_channels (int): 输入通道数，默认132
            out_channels (int): 输出通道数，默认132，保持与原始输入维度一致
            tremor_freq_band (tuple): 震颤频带范围 (Hz)，默认(3, 7)
            fs (int): 信号采样率 (Hz)，默认100
        """
        super(MSHFeatureExtractor, self).__init__()
        
        # 保存参数
        self.tremor_freq_band = tremor_freq_band
        self.fs = fs
        
        # 频域感知模块 - 专注于震颤频带
        self.freq_aware = FrequencyAwareModule(in_channels, 32, tremor_freq_band, fs)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        
        # 残差连接处理
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数据，形状为 [batch_size, in_channels, time_points]
            
        返回:
            torch.Tensor: 增强的特征表示，形状为 [batch_size, out_channels, time_points]
        """
        # 残差连接
        residual = self.residual_proj(x)
        
        # 频域感知
        feat_freq = self.freq_aware(x)
        
        # 特征融合
        fused_features = self.fusion(feat_freq)
        
        # 残差连接
        enhanced_features = F.relu(fused_features + residual)
        
        return enhanced_features


class ShallowCNN1D(nn.Module):
    """使用频域感知模块的1D CNN分类器"""
    
    def __init__(self, n_channels=132, n_classes=2, use_feature_extractor=False, 
                 tremor_freq_band=(3,7), fs=100):
        """
        初始化1D CNN分类器
        
        参数:
            n_channels (int): 输入通道数，默认132
            n_classes (int): 类别数，默认2
            use_feature_extractor (bool): 是否使用多尺度分层特征提取器，默认False
            tremor_freq_band (tuple): 震颤频带范围 (Hz)，默认(3, 7)
            fs (int): 信号采样率 (Hz)，默认100
        """
        super(ShallowCNN1D, self).__init__()
        
        self.use_feature_extractor = use_feature_extractor
        
        # 特征提取器 - 只使用频域感知模块
        if use_feature_extractor:
            self.feature_extractor = MSHFeatureExtractor(
                in_channels=n_channels, 
                out_channels=n_channels,
                tremor_freq_band=tremor_freq_band,
                fs=fs
            )
        
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
        # 应用特征提取器
        if self.use_feature_extractor:
            x = self.feature_extractor(x)
        
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