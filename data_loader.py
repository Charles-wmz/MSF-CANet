import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class PADSDataset(Dataset):
    """PADS数据集加载类"""
    
    def __init__(self, data_path, file_list, transform=None, use_boss=False, boss_params=None):
        """
        初始化PADS数据集
        
        参数:
            data_path (str): 预处理数据路径
            file_list (pandas.DataFrame): 包含文件ID和标签的DataFrame
            transform (callable, optional): 用于数据转换的函数
            use_boss (bool): 是否使用BOSS特征提取
            boss_params (dict, optional): BOSS参数，如window_sizes, n_bins, word_length等
        """
        self.data_path = data_path
        # 确保文件列表中只有标签0和1
        if set(file_list['label'].unique()) != set([0, 1]) and len(file_list['label'].unique()) > 2:
            print("警告: PADSDataset接收到包含非0/1标签的数据，将自动过滤")
            original_len = len(file_list)
            self.file_list = file_list[file_list['label'].isin([0, 1])].reset_index(drop=True)
            filtered_len = len(self.file_list)
            print(f"PADSDataset过滤: 从{original_len}个样本中移除了{original_len - filtered_len}个非0/1标签的样本，剩余{filtered_len}个样本")
        else:
            self.file_list = file_list
        self.transform = transform
        self.use_boss = use_boss
        self.boss_params = boss_params if boss_params is not None else {}
        
        # 移动数据路径
        self.mov_path = os.path.join(data_path, 'movement')
        
        # 初始化BOSS特征提取器
        self.boss_extractor = None
        if self.use_boss:
            self.boss_extractor = BossFeatureExtractor(
                window_sizes=self.boss_params.get('window_sizes', (20, 40, 80)),
                n_bins=self.boss_params.get('n_bins', 3),
                word_length=self.boss_params.get('word_length', 8),
                norm_mean=self.boss_params.get('norm_mean', True)
            )
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 从file_list获取id值
        if 'id' in self.file_list.columns:
            patient_id = self.file_list.iloc[idx]['id']
            # 确保id是三位数格式（001, 002, 等）
            file_id = f"{int(patient_id):03d}"
        else:
            # 如果没有id列，使用索引+1作为文件名
            file_id = f"{idx+1:03d}"
        
        # 获取标签
        label = self.file_list.iloc[idx]['label']
        
        # 构建文件路径
        file_path = os.path.join(self.mov_path, f"{file_id}_ml.bin")
        
        # 加载运动数据文件
        try:
            data = np.fromfile(file_path, dtype=np.float32)
        except FileNotFoundError:
            print(f"无法找到文件: {file_path}")
            # 返回一个全零的数据作为占位符
            data = np.zeros((132, 1000), dtype=np.float32).flatten()
        
        # 重塑数据形状 - 这里需要根据实际数据维度调整
        n_channels = 132  # 修正实际通道数：14个任务 * 2个手腕 * (x个轴的加速度 + y个轴的陀螺仪)
        
        # 确保数据能被重塑为正确的形状
        if len(data) % n_channels != 0:
            # 如果数据长度不是通道数的整数倍，则调整数据长度
            new_length = (len(data) // n_channels) * n_channels
            data = data[:new_length]
        
        data = data.reshape(n_channels, -1)
        
        # 转换为pytorch张量
        data_tensor = torch.FloatTensor(data)
        
        # 应用变换（如果有）
        if self.transform:
            data_tensor = self.transform(data_tensor)
        
        # 1D卷积模型使用原始形状 [通道数, 时间点]
        return data_tensor, torch.tensor(label, dtype=torch.long)

class BOSSDataset(Dataset):
    """使用BOSS特征的PADS数据集类"""
    
    def __init__(self, data_path, file_list, window_sizes=(20, 40, 80), n_bins=3, word_size=2, word_length=None, window_step=2, n_jobs=16):
        """
        初始化BOSS特征的PADS数据集
        
        参数:
            data_path (str): 预处理数据路径
            file_list (pandas.DataFrame): 包含文件ID和标签的DataFrame
            window_sizes (tuple): 滑动窗口大小
            n_bins (int): 符号化的箱数
            word_size (int): SFA单词长度
            word_length (int, 弃用): word_size的别名，保持向后兼容
            window_step (int): 滑动窗口的步长，默认为2
            n_jobs (int): 并行处理的任务数，默认为16
        """
        self.data_path = data_path
        # 确保文件列表中只有标签0和1
        if set(file_list['label'].unique()) != set([0, 1]) and len(file_list['label'].unique()) > 2:
            print("警告: BOSSDataset接收到包含非0/1标签的数据，将自动过滤")
            original_len = len(file_list)
            self.file_list = file_list[file_list['label'].isin([0, 1])].reset_index(drop=True)
            filtered_len = len(self.file_list)
            print(f"BOSSDataset过滤: 从{original_len}个样本中移除了{original_len - filtered_len}个非0/1标签的样本，剩余{filtered_len}个样本")
        else:
            self.file_list = file_list
        self.window_sizes = window_sizes
        self.n_bins = n_bins
        # 向后兼容
        if word_length is not None:
            self.word_length = word_length
        else:
            self.word_length = word_size
        self.word_size = self.word_length
        self.window_step = window_step
        self.n_jobs = n_jobs
        
        # 移动数据路径
        self.mov_path = os.path.join(data_path, 'movement')
        
        # 加载所有数据以便一次性进行BOSS特征提取
        self.data_tensors = []
        self.labels = []
        
        print(f"加载数据集: {len(file_list)}个样本...")
        for i in tqdm(range(len(file_list)), desc="加载数据", unit="样本"):
            # 从file_list获取id值
            if 'id' in self.file_list.columns:
                patient_id = self.file_list.iloc[i]['id']
                # 确保id是三位数格式（001, 002, 等）
                file_id = f"{int(patient_id):03d}"
            else:
                # 如果没有id列，使用索引+1作为文件名
                file_id = f"{i+1:03d}"
            
            # 获取标签
            label = self.file_list.iloc[i]['label']
            
            # 构建文件路径
            file_path = os.path.join(self.mov_path, f"{file_id}_ml.bin")
            
            # 加载运动数据文件
            try:
                data = np.fromfile(file_path, dtype=np.float32)
            except FileNotFoundError:
                print(f"无法找到文件: {file_path}")
                # 返回一个全零的数据作为占位符
                data = np.zeros((132, 1000), dtype=np.float32).flatten()
            
            # 重塑数据形状
            n_channels = 132
            
            # 确保数据能被重塑为正确的形状
            if len(data) % n_channels != 0:
                new_length = (len(data) // n_channels) * n_channels
                data = data[:new_length]
            
            data = data.reshape(n_channels, -1)
            
            self.data_tensors.append(data)
            self.labels.append(label)
        
        # 转换为numpy数组
        self.data_tensors = np.array(self.data_tensors)
        self.labels = np.array(self.labels)
        
        # 初始化BOSS特征提取器并提取特征
        print(f"\n开始提取BOSS特征 (窗口大小={window_sizes}, n_bins={n_bins}, word_size={self.word_size}, window_step={window_step}, n_jobs={n_jobs})...")
        self.boss_extractor = BossFeatureExtractor(
            window_sizes=window_sizes,
            n_bins=n_bins,
            word_size=self.word_size,
            window_step=window_step,
            n_jobs=n_jobs
        )
        
        # 提取BOSS特征
        self.boss_extractor.fit(self.data_tensors)
        self.boss_features = self.boss_extractor.transform(self.data_tensors)
        
        print(f"BOSS特征提取完成。特征维度: {self.boss_features.shape}")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 返回BOSS特征和标签
        boss_feature = torch.FloatTensor(self.boss_features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return boss_feature, label

def load_data(data_path="../PADS/preprocessed", batch_size=16, test_size=0.2, 
              val_size=0.2, random_state=42, use_boss=False, boss_params=None):
    """
    加载PADS数据集并分割为训练、验证和测试集
    
    参数:
        data_path (str): 预处理数据路径
        batch_size (int): 批量大小
        test_size (float): 测试集比例
        val_size (float): 验证集比例（从训练集中分割）
        random_state (int): 随机种子
        use_boss (bool): 是否使用BOSS特征提取
        boss_params (dict, optional): BOSS参数，如window_sizes, n_bins, word_length等
        
    返回:
        tuple: (train_loader, val_loader, test_loader, n_classes)
    """
    # 加载文件列表
    file_list_path = os.path.join(data_path, "file_list.csv")
    file_list = pd.read_csv(file_list_path)
    
    # 打印文件列表的列名和前几行，用于调试
    print(f"文件列表的列名: {file_list.columns.tolist()}")
    print(f"文件列表样本:\n{file_list.head()}")
    
    # 筛选出标签为0和1的数据，忽略标签为2的数据
    original_length = len(file_list)
    file_list = file_list[file_list['label'].isin([0, 1])]
    filtered_length = len(file_list)
    print(f"已过滤数据: 原始样本数 {original_length}，保留样本数 {filtered_length}，移除 {original_length - filtered_length} 个标签为2的样本")
    
    # 获取类别数量
    n_classes = len(file_list['label'].unique())
    print(f"类别数量: {n_classes}")
    print(f"标签分布: {file_list['label'].value_counts()}")
    
    # 分割数据集为训练+验证和测试
    train_val_files, test_files = train_test_split(
        file_list, test_size=test_size, random_state=random_state, stratify=file_list['label']
    )
    
    # 分割训练集为训练和验证
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size, random_state=random_state, stratify=train_val_files['label']
    )
    
    print(f"训练集大小: {len(train_files)}")
    print(f"验证集大小: {len(val_files)}")
    print(f"测试集大小: {len(test_files)}")
    
    if use_boss:
        # 使用BOSS特征提取
        print("使用BOSS特征提取...")
        print("\n=== 处理训练集 ===")
        train_dataset = BOSSDataset(
            data_path, train_files, 
            window_sizes=boss_params.get('window_sizes', (20, 40, 80)),
            n_bins=boss_params.get('n_bins', 3),
            word_size=boss_params.get('word_size', 2),
            window_step=boss_params.get('window_step', 2)
        )
        
        print("\n=== 处理验证集 ===")
        val_dataset = BOSSDataset(
            data_path, val_files,
            window_sizes=boss_params.get('window_sizes', (20, 40, 80)),
            n_bins=boss_params.get('n_bins', 3),
            word_size=boss_params.get('word_size', 2),
            window_step=boss_params.get('window_step', 2)
        )
        
        print("\n=== 处理测试集 ===")
        test_dataset = BOSSDataset(
            data_path, test_files,
            window_sizes=boss_params.get('window_sizes', (20, 40, 80)),
            n_bins=boss_params.get('n_bins', 3),
            word_size=boss_params.get('word_size', 2),
            window_step=boss_params.get('window_step', 2)
        )
    else:
        # 创建常规数据集
        train_dataset = PADSDataset(data_path, train_files)
        val_dataset = PADSDataset(data_path, val_files)
        test_dataset = PADSDataset(data_path, test_files)
    
    # 创建数据加载器 - 添加确定性设置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 单线程加载以确保确定性
        generator=torch.Generator().manual_seed(random_state),  # 确保shuffle的确定性
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 验证集不需要打乱
        num_workers=0, 
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 测试集不需要打乱
        num_workers=0, 
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, n_classes 