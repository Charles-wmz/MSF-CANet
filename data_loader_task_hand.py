import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PADSSelectiveDataset(Dataset):
    """可选择特定任务或左右手的PADS数据集加载类"""
    
    def __init__(self, data_path, file_list, selection_mode='all', selection_value=None):
        """
        初始化PADS选择性数据集
        
        参数:
            data_path (str): 预处理数据路径
            file_list (pandas.DataFrame): 包含文件ID和标签的DataFrame
            selection_mode (str): 选择模式，可以是 'all'(使用所有通道), 'task'(选择特定任务), 'hand'(选择左或右手)
            selection_value: 根据selection_mode确定的选择值
                - 'task'模式: 整数1-11，表示选择哪个任务 (例如: 1代表"Relaxed1", 2代表"Relaxed2"等)
                - 'hand'模式: 字符串'left'或'right'，表示选择哪只手
        """
        self.data_path = data_path
        # 确保文件列表中只有标签0和1
        if set(file_list['label'].unique()) != set([0, 1]) and len(file_list['label'].unique()) > 2:
            print("警告: PADSSelectiveDataset接收到包含非0/1标签的数据，将自动过滤")
            original_len = len(file_list)
            self.file_list = file_list[file_list['label'].isin([0, 1])].reset_index(drop=True)
            filtered_len = len(self.file_list)
            print(f"PADSSelectiveDataset过滤: 从{original_len}个样本中移除了{original_len - filtered_len}个非0/1标签的样本，剩余{filtered_len}个样本")
        else:
            self.file_list = file_list
        
        # 设置选择模式和值
        self.selection_mode = selection_mode
        self.selection_value = selection_value
        
        # 计算要选择的通道索引
        self.selected_channels = self._get_selected_channels()
        
        # 移动数据路径
        self.mov_path = os.path.join(data_path, 'movement')
    
    def _get_selected_channels(self):
        """获取根据选择模式和值需要保留的通道索引"""
        # 任务列表 (全部11个任务)
        tasks = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
                 "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
        
        # 所有通道总数
        n_channels = 132
        
        # 每个任务的通道数
        channels_per_task = 12
        
        # 如果选择所有通道，返回所有索引
        if self.selection_mode == 'all':
            return list(range(n_channels))
        
        # 选择特定任务的通道
        elif self.selection_mode == 'task':
            if not isinstance(self.selection_value, int) or not 1 <= self.selection_value <= 11:
                raise ValueError("'task'模式下，selection_value必须是1-11之间的整数")
            
            task_idx = self.selection_value - 1
            start_idx = task_idx * channels_per_task
            return list(range(start_idx, start_idx + channels_per_task))
        
        # 选择左手或右手的通道
        elif self.selection_mode == 'hand':
            if self.selection_value not in ['left', 'right']:
                raise ValueError("'hand'模式下，selection_value必须是'left'或'right'")
            
            selected_channels = []
            for task_idx in range(11):
                # 每个任务中的6个通道（每只手3个加速度+3个旋转）
                start_idx = task_idx * channels_per_task
                
                if self.selection_value == 'left':
                    # 左手通道 (前3个加速度 + 后3个旋转 = 6个)
                    selected_channels.extend(list(range(start_idx, start_idx + 3)))  # 左手加速度
                    selected_channels.extend(list(range(start_idx + 6, start_idx + 9)))  # 左手旋转
                else:  # 'right'
                    # 右手通道 (中间3个加速度 + 最后3个旋转 = 6个)
                    selected_channels.extend(list(range(start_idx + 3, start_idx + 6)))  # 右手加速度
                    selected_channels.extend(list(range(start_idx + 9, start_idx + 12)))  # 右手旋转
            
            return selected_channels
        
        else:
            raise ValueError("selection_mode必须是'all', 'task'或'hand'中的一个")
    
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
        
        # 重塑数据形状
        n_channels = 132
        
        # 确保数据能被重塑为正确的形状
        if len(data) % n_channels != 0:
            # 如果数据长度不是通道数的整数倍，则调整数据长度
            new_length = (len(data) // n_channels) * n_channels
            data = data[:new_length]
        
        data = data.reshape(n_channels, -1)
        
        # 只选择需要的通道
        data = data[self.selected_channels]
        
        # 转换为pytorch张量
        data_tensor = torch.FloatTensor(data)
        
        return data_tensor, torch.tensor(label, dtype=torch.long)


def load_selective_data(data_path="../PADS/preprocessed", batch_size=16, test_size=0.2, 
                        val_size=0.2, random_state=42, selection_mode='all', selection_value=None, return_dataset=False):
    """
    加载选择性PADS数据集并分割为训练、验证和测试集
    
    参数:
        data_path (str): 预处理数据路径
        batch_size (int): 批量大小
        test_size (float): 测试集比例
        val_size (float): 验证集比例（从训练集中分割）
        random_state (int): 随机种子
        selection_mode (str): 选择模式，可以是 'all', 'task', 'hand'
        selection_value: 根据selection_mode确定的选择值
        return_dataset (bool): 是否返回完整数据集对象
        
    返回:
        tuple: (train_loader, val_loader, test_loader, n_classes, n_channels)
        如果return_dataset=True: (train_loader, val_loader, test_loader, n_classes, n_channels, dataset)
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
    
    # 创建数据集
    train_dataset = PADSSelectiveDataset(data_path, train_files, selection_mode, selection_value)
    val_dataset = PADSSelectiveDataset(data_path, val_files, selection_mode, selection_value)
    test_dataset = PADSSelectiveDataset(data_path, test_files, selection_mode, selection_value)
    
    # 如果需要返回完整数据集，创建一个合并所有数据的数据集
    if return_dataset:
        full_dataset = PADSSelectiveDataset(data_path, file_list, selection_mode, selection_value)
    
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
    
    # 获取选中的通道数量
    n_channels = len(train_dataset.selected_channels)
    
    if return_dataset:
        return train_loader, val_loader, test_loader, n_classes, n_channels, full_dataset
    else:
        return train_loader, val_loader, test_loader, n_classes, n_channels

class CustomDataset(Dataset):
    """用于嵌套交叉验证的简单数据集类"""
    
    def __init__(self, data, targets):
        """
        初始化数据集
        
        参数:
            data: 数据特征
            targets: 目标标签
        """
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx] 