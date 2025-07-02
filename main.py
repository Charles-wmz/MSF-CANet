"""
基础模型
python main.py --model_type base

多尺度模型
python main.py --model_type multiscale --use_feature_extractor

频域模型
python main.py --model_type frequency --use_feature_extractor

注意力模型
python main.py --model_type attention --use_feature_extractor

完整模型
python main.py --model_type full --use_feature_extractor

完整模型进行手部评估
python run_hand_evaluation.py --model_type full --use_feature_extractor

完整模型进行任务评估
python run_task_evaluation.py --model_type full --use_feature_extractor

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from data_loader import PADSDataset
# 移除Symptom-Aware Dynamic CE损失函数引用

# 动态导入模型
import importlib

import train
from train import train_epoch, validate
from evaluate import evaluate_model, calculate_metrics

def set_seed(seed):
    """
    设置所有随机种子以确保结果可重复
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PADS数据集分类')
    
    # 路径参数
    parser.add_argument('--data_path', type=str, default='../PADS/preprocessed', help='预处理数据路径')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    
    # 交叉验证参数
    parser.add_argument('--n_splits', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    
    # 模型参数
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--model_type', type=str, default='base', 
                       choices=['base', 'multiscale', 'frequency', 'attention', 'full', 'full_noattn'], 
                       help='模型类型: base(基础模型), multiscale(多尺度模型), frequency(频域模型), attention(注意力模型), full(多尺度+频域+注意力), full_noattn(多尺度+频域,无注意力)')
    
    # 时间维度参数，默认976，根据实际数据设置
    parser.add_argument('--time_dim', type=int, default=976, help='时间维度大小')
    
    # 特征提取器参数
    parser.add_argument('--use_feature_extractor', action='store_true', help='是否使用特征提取器')
    parser.add_argument('--tremor_freq_band', type=str, default='3,7', help='震颤频带范围(Hz)，用逗号分隔')
    parser.add_argument('--sampling_rate', type=int, default=100, help='信号采样率(Hz)')
    
    return parser.parse_args()

def load_file_list(data_path):
    """
    加载文件列表
    
    参数:
        data_path: 数据路径
        
    返回:
        pandas.DataFrame: 包含文件信息的DataFrame
    """
    # 加载文件列表
    file_list_path = os.path.join(data_path, "file_list.csv")
    if os.path.exists(file_list_path):
        file_list = pd.read_csv(file_list_path)
        return file_list
    else:
        raise FileNotFoundError(f"找不到文件列表: {file_list_path}")

def prepare_data(file_list, data_path, outer_fold, inner_fold=None, is_outer=True, random_state=42, batch_size=16):
    """
    准备数据集，适用于嵌套交叉验证的不同折
    
    参数:
        file_list: 文件列表
        data_path: 数据路径
        outer_fold: 外部折的信息 (idx, (train_val_idx, test_idx)) 或 测试集文件列表
        inner_fold: 内部折的信息 (inner_idx, (train_idx, val_idx))
        is_outer: 是否为外部折划分
        random_state: 随机种子
        batch_size: 批量大小
        
    返回:
        tuple: 数据加载器和相关信息
    """
    if is_outer:
        # 处理外部折
        if isinstance(outer_fold, tuple):
            outer_idx, (train_val_idx, test_idx) = outer_fold
            test_files = file_list.iloc[test_idx]
        else:
            # 如果outer_fold已经是文件列表
            test_files = outer_fold
        
        # 创建测试数据集
        test_dataset = PADSDataset(data_path, test_files)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        # 获取样本的时间维度
        sample_data, _ = next(iter(test_loader))
        time_dim = sample_data.shape[-1]  # 最后一维是时间维度
        
        return test_loader, time_dim
    else:
        # 处理内部折
        inner_idx, (train_idx, val_idx) = inner_fold
        train_val_files = outer_fold
        
        # 分割数据
        train_files = train_val_files.iloc[train_idx].reset_index(drop=True)
        val_files = train_val_files.iloc[val_idx].reset_index(drop=True)
        
        # 创建数据集
        train_dataset = PADSDataset(data_path, train_files)
        val_dataset = PADSDataset(data_path, val_files)
        
        # 创建数据加载器，确保使用固定随机种子
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # 单线程加载以确保确定性
            generator=torch.Generator().manual_seed(random_state),  # 确保shuffle的确定性
            pin_memory=True,
            drop_last=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader

def plot_metrics_boxplot(metrics_df, metrics_list, output_dir, fold_num=None):
    """
    绘制性能指标的箱型图
    
    参数:
        metrics_df: 包含指标的DataFrame
        metrics_list: 要绘制的指标列表
        output_dir: 输出目录
        fold_num: 外部折编号，如果为None则绘制平均结果箱型图
    """
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(10, 6))
    
    # 创建指标名称映射
    metrics_map = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-score': 'f1',
        'AUC': 'roc_auc'
    }
    
    # 准备箱型图数据
    boxplot_data = []
    for metric in metrics_list:
        # 获取对应的小写列名
        if metric in metrics_map:
            column_name = metrics_map[metric]
        else:
            column_name = metric.lower()
            
        if column_name in metrics_df.columns:
            boxplot_data.append(metrics_df[column_name].values)
        else:
            # 如果找不到列，尝试使用原始名称
            boxplot_data.append(metrics_df[metric].values if metric in metrics_df.columns else [0])
    
    # 设置颜色样式
    box_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    whisker_colors = ['#2980b9', '#27ae60', '#c0392b', '#d35400', '#8e44ad']
    median_colors = ['darkblue', 'darkgreen', 'darkred', 'darkorange', 'darkviolet']
    mean_marker_colors = ['blue', 'green', 'red', 'orange', 'purple'] 
    
    # 绘制箱型图，showmeans=True显示平均值为点
    bp = plt.boxplot(boxplot_data, patch_artist=True, widths=0.6, showmeans=True, meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=8))
    
    # 自定义箱型图样式
    for i, box in enumerate(bp['boxes']):
        box_color = box_colors[i % len(box_colors)]
        box.set(color=box_color, linewidth=2)
        box.set(facecolor=box_color + '40')  # 添加透明度
    
    for i, whisker in enumerate(bp['whiskers']):
        whisker_color = whisker_colors[i // 2 % len(whisker_colors)]
        whisker.set(color=whisker_color, linewidth=2, linestyle='-')
    
    for i, cap in enumerate(bp['caps']):
        cap.set(color=whisker_colors[i // 2 % len(whisker_colors)], linewidth=2)
    
    for i, median in enumerate(bp['medians']):
        median.set(color=median_colors[i % len(median_colors)], linewidth=2)
    
    for i, flier in enumerate(bp['fliers']):
        flier.set(marker='o', markerfacecolor='white', 
                  markeredgecolor=box_colors[i % len(box_colors)],
                  markersize=8, alpha=0.7)
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 设置坐标轴
    plt.xticks(range(1, len(metrics_list) + 1), metrics_list, fontsize=12, fontname='Times New Roman')
    plt.yticks(np.arange(0.80, 1.01, 0.05), [f'{x:.2f}' for x in np.arange(0.80, 1.01, 0.05)], fontsize=11, fontname='Times New Roman')
    plt.ylim(0.80, 1.00)  # 修改纵坐标范围
    
    # 移除Y轴标签
    plt.ylabel('')
    
    # 保存图表路径
    if fold_num is not None:
        save_path = os.path.join(output_dir, f'fold_{fold_num}_boxplot.png')
    else:
        save_path = os.path.join(output_dir, 'average_boxplot.png')
    
    # 添加数值标签，保留两位小数
    for i, metric in enumerate(metrics_list):
        # 获取对应的列名
        if metric in metrics_map:
            column_name = metrics_map[metric]
        else:
            column_name = metric.lower()
            
        if column_name in metrics_df.columns:
            mean_val = np.mean(metrics_df[column_name])
            std_val = np.std(metrics_df[column_name])
            plt.text(i + 1, 0.83, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}', 
                    ha='center', va='bottom', fontsize=9, 
                    color=median_colors[i % len(median_colors)],
                    fontname='Times New Roman')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def create_fold_boxplot(fold_metrics_list, metrics_list, output_dir, fold_num):
    """
    为单个外部折生成箱型图
    
    参数:
        fold_metrics_list: 该折所有内部折和测试的指标列表
        metrics_list: 要绘制的指标列表
        output_dir: 输出目录
        fold_num: 外部折编号
    
    返回:
        str: 保存的箱型图路径
    """
    # 创建指标名称映射
    metrics_map = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-score': 'f1',
        'AUC': 'roc_auc'
    }
    
    # 标准化指标数据
    standardized_metrics = []
    for metric_dict in fold_metrics_list:
        standardized_dict = {}
        for metric in metrics_list:
            metric_key = metrics_map[metric]
            standardized_dict[metric_key] = metric_dict.get(metric_key, 0.0)
        standardized_metrics.append(standardized_dict)
    
    # 创建DataFrame
    fold_df = pd.DataFrame(standardized_metrics)
    
    # 调用通用箱型图函数
    return plot_metrics_boxplot(fold_df, metrics_list, output_dir, fold_num)

def plot_all_roc_curves(all_fold_metrics, output_dir):
    """
    在一张图中绘制所有折的ROC曲线
    
    参数:
        all_fold_metrics: 所有折的指标列表
        output_dir: 输出目录
    """
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(10, 8))
    
    # 设置颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_fold_metrics)))
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # 绘制每个折的ROC曲线
    for i, fold_metrics in enumerate(all_fold_metrics):
        if 'fpr' in fold_metrics and 'tpr' in fold_metrics and 'roc_auc' in fold_metrics:
            plt.plot(fold_metrics['fpr'], fold_metrics['tpr'], 
                    color=colors[i], 
                    label=f'Fold {i+1} (AUC = {fold_metrics["roc_auc"]:.4f})',
                    linewidth=2)
    
    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate', fontsize=12, fontname='Times New Roman')
    plt.ylabel('True Positive Rate', fontsize=12, fontname='Times New Roman')
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=10, prop={'family': 'Times New Roman'})
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    save_path = os.path.join(output_dir, 'all_roc_curves.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def nested_cross_validation(file_list, data_path, args):
    """
    执行嵌套交叉验证
    
    参数:
        file_list: 文件列表
        data_path: 数据路径
        args: 命令行参数
        
    返回:
        dict: 平均指标
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 过滤数据，只保留标签为0和1的样本
    original_len = len(file_list)
    file_list = file_list[file_list['label'].isin([0, 1])].reset_index(drop=True)
    filtered_len = len(file_list)
    print(f"数据过滤: 从{original_len}个样本中移除了{original_len - filtered_len}个非0/1标签的样本，剩余{filtered_len}个样本")
    
    # 固定二分类任务 - HC(0)和PD(1)
    n_classes = 2
    print(f"类别数量: {n_classes}")
    print(f"标签分布: \n{file_list['label'].value_counts()}")
    
    # 解析震颤频带参数
    tremor_freq_band = tuple(map(float, args.tremor_freq_band.split(',')))
    
    # 创建外部交叉验证
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    all_metrics = []
    all_fold_metrics = []  # 存储每个折的指标
    
    # 添加一个新的列表，用于存储每个外部折的内部折指标
    all_inner_fold_metrics = []
    
    # 创建用于存储每个外部折数据的列表
    outer_fold_data = [[] for _ in range(args.n_splits)]
    
    # 根据model_type动态导入相应模型
    if args.model_type == 'base':
        from model_base import ShallowCNN1D
        ModelClass = ShallowCNN1D
    elif args.model_type == 'multiscale':
        from model_multiscale import ShallowCNN1D
        ModelClass = ShallowCNN1D
    elif args.model_type == 'frequency':
        from model_frequency import ShallowCNN1D
        ModelClass = ShallowCNN1D
    elif args.model_type == 'attention':
        from model_attention import ShallowCNN1D
        ModelClass = ShallowCNN1D
    elif args.model_type == 'full_noattn':
        from model import ShallowCNN1DNoAttn
        ModelClass = ShallowCNN1DNoAttn
    else:  # 'full'
        from model import ShallowCNN1D
        ModelClass = ShallowCNN1D
    
    # 在保存ROC曲线之前，收集所有折的ROC数据
    all_roc_data = []
    
    # 外部交叉验证
    for i, (train_val_idx, test_idx) in enumerate(outer_cv.split(file_list)):
        print(f"\n---- 外部折 {i+1}/{args.n_splits} ----")
        
        # 准备测试数据
        test_files = file_list.iloc[test_idx]
        test_loader, time_dim = prepare_data(
            test_files, 
            data_path, 
            test_files, 
            None, 
            is_outer=True,
            random_state=args.random_state,
            batch_size=args.batch_size
        )
        
        # 更新时间维度参数
        if time_dim is not None:
            print(f"检测到数据时间维度: {time_dim}")
            args.time_dim = time_dim
        
        # 创建内部交叉验证
        inner_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
        train_val_files = file_list.iloc[train_val_idx]
        
        best_inner_idx = 0
        best_val_accuracy = 0.0
        best_checkpoint_dir = None
        
        # 用于存储当前外部折的所有内部折指标
        inner_fold_metrics = []
        
        # 内部交叉验证
        for j, (train_idx, val_idx) in enumerate(inner_cv.split(train_val_files)):
            print(f"\n---- 内部折 {j+1}/{args.n_splits} ----")
            
            # 准备内部折数据
            train_loader, val_loader = prepare_data(
                train_val_files, 
                data_path, 
                train_val_files, 
                (j, (train_idx, val_idx)), 
                is_outer=False,
                random_state=args.random_state,
                batch_size=args.batch_size
            )
            
            # 重设随机种子，确保模型权重初始化相同
            torch.manual_seed(args.random_state)
            torch.cuda.manual_seed(args.random_state)
            
            # 创建模型
            model = ModelClass(
                n_channels=132,
                n_classes=2,
                use_feature_extractor=args.use_feature_extractor,
                tremor_freq_band=tremor_freq_band,
                fs=args.sampling_rate
            )
                
            # 将模型移至设备
            model.to(device)
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # 学习率调度器 - 减少学习率当验证损失停止改善
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
            )
            
            # 创建内部折检查点目录
            inner_fold_dir = os.path.join(args.output_dir, f'outer_fold_{i+1}_inner_fold_{j+1}')
            os.makedirs(inner_fold_dir, exist_ok=True)
            
            # 训练模型
            model, train_losses, val_losses, val_accuracies = train.train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                device, 
                num_epochs=args.epochs, 
                patience=args.patience,
                checkpoint_dir=inner_fold_dir
            )
                
            # 加载最佳模型用于验证
            best_model_path = os.path.join(inner_fold_dir, 'best_model.pth')
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 在验证集上评估
            val_loss, val_accuracy, _, _ = validate(model, val_loader, criterion, device)
            print(f"内部折 {j+1} 最佳验证准确率: {val_accuracy:.4f}")
            
            # 计算验证集上的所有指标
            val_predictions, val_targets, val_probas = evaluate_model(model, val_loader, device)
            val_metrics = calculate_metrics(val_targets, val_predictions, val_probas)
            
            # 保存当前外部折的内部折指标
            inner_fold_metrics.append(val_metrics)
            
            # 更新最佳内部折
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_inner_idx = j
                best_checkpoint_dir = inner_fold_dir
        
        # 使用最佳内部折的模型在测试集上评估
        print(f"\n使用内部折 {best_inner_idx+1} 的最佳模型进行测试")
        
        # 重新加载最佳模型
        best_model_path = os.path.join(best_checkpoint_dir, 'best_model.pth')
        best_model = ModelClass(
            n_channels=132,
            n_classes=2,
            use_feature_extractor=args.use_feature_extractor,
            tremor_freq_band=tremor_freq_band,
            fs=args.sampling_rate
        )
        best_model.to(device)
        checkpoint = torch.load(best_model_path)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在测试集上评估
        test_predictions, test_targets, test_probas = evaluate_model(best_model, test_loader, device)
        metrics = calculate_metrics(test_targets, test_predictions, test_probas)
        
        # 计算ROC曲线数据
        fpr, tpr, _ = roc_curve(test_targets, test_probas[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # 保存ROC数据
        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        all_roc_data.append(roc_data)
        
        # 打印结果
        print(f"外部折 {i+1} 测试结果:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # 保存外部折的详细测试结果
        fold_results_dir = os.path.join(args.output_dir, f'outer_fold_{i+1}_results')
        os.makedirs(fold_results_dir, exist_ok=True)
        
        # 保存混淆矩阵
        cm = confusion_matrix(test_targets, test_predictions)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Fold {i+1} Confusion Matrix')
        plt.colorbar()
        
        # 添加文本
        thresh = cm.max() / 2.
        for i_cm in range(cm.shape[0]):
            for j_cm in range(cm.shape[1]):
                plt.text(j_cm, i_cm, format(cm[i_cm, j_cm], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i_cm, j_cm] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_results_dir, 'confusion_matrix.png'))
        plt.close()
        
        # 保存精确率-召回率曲线
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(test_targets, test_probas[:, 1])
        plt.plot(recall, precision, label=f'Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Fold {i+1} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fold_results_dir, 'precision_recall_curve.png'))
        plt.close()
    
        # 保存详细指标到CSV
        fold_metrics_df = pd.DataFrame([metrics])
        fold_metrics_df.to_csv(os.path.join(fold_results_dir, 'metrics.csv'), index=False)
        
        # 保存预测结果
        predictions_df = pd.DataFrame({
            'true_label': test_targets,
            'predicted_label': test_predictions,
            'probability_class_0': test_probas[:, 0] if test_probas.shape[1] > 1 else 1 - test_probas[:, 0],
            'probability_class_1': test_probas[:, 1] if test_probas.shape[1] > 1 else test_probas[:, 0]
        })
        predictions_df.to_csv(os.path.join(fold_results_dir, 'predictions.csv'), index=False)
        
        # 保存外部折的指标
        metrics['outer_fold'] = i+1
        all_metrics.append(metrics)
        all_fold_metrics.append(metrics.copy())  # 复制以避免引用问题
        
        # 保存当前外部折的内部折指标
        all_inner_fold_metrics.append(inner_fold_metrics)
        
        # 将测试指标也加入到此外部折的数据中
        outer_fold_data[i] = inner_fold_metrics + [metrics]
        
        # 创建外部折的箱型图
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        fold_boxplot_path = create_fold_boxplot(outer_fold_data[i], metrics_to_plot, args.output_dir, i+1)
        print(f"外部折 {i+1} 箱型图已保存至: {fold_boxplot_path}")
    
    # 计算平均指标
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        if metric != 'outer_fold':
            # 使用所有外部折的测试结果计算平均值
            avg_metrics[metric] = np.mean([fold_metric[metric] for fold_metric in all_metrics])
            std_metrics = np.std([fold_metric[metric] for fold_metric in all_metrics])
            avg_metrics[f"{metric}_std"] = std_metrics
    
    # 打印平均结果
    print("\n平均测试结果:")
    for metric_name, metric_value in avg_metrics.items():
        if not metric_name.endswith('_std'):
            std_value = avg_metrics.get(f"{metric_name}_std", 0)
            print(f"{metric_name}: {metric_value:.4f} ± {std_value:.4f}")
    
    # 创建DataFrame存储所有折的指标，用于绘图
    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    
    # 确保fold_metrics_df中包含outer_fold列
    if 'outer_fold' not in fold_metrics_df.columns:
        fold_metrics_df['outer_fold'] = range(1, len(all_fold_metrics) + 1)
    
    # 创建一个更详细的DataFrame来存储每个外部折的内部折指标数据
    detailed_fold_metrics = []
    for i, inner_fold_metrics_list in enumerate(all_inner_fold_metrics):
        for j, inner_metrics in enumerate(inner_fold_metrics_list):
            inner_metrics_copy = inner_metrics.copy()
            inner_metrics_copy['outer_fold'] = i + 1
            inner_metrics_copy['inner_fold'] = j + 1
            inner_metrics_copy['fold_type'] = 'inner'
            detailed_fold_metrics.append(inner_metrics_copy)
    
    # 添加测试结果
    for i, test_metrics in enumerate(all_fold_metrics):
        test_metrics_copy = test_metrics.copy()
        test_metrics_copy['outer_fold'] = i + 1
        test_metrics_copy['inner_fold'] = 0  # 0表示测试结果
        test_metrics_copy['fold_type'] = 'test'
        detailed_fold_metrics.append(test_metrics_copy)
    
    # 保存详细的指标数据
    if detailed_fold_metrics:
        detailed_df = pd.DataFrame(detailed_fold_metrics)
        detailed_df.to_csv(os.path.join(args.output_dir, 'fold_metrics_detailed.csv'), index=False)
    
    # 绘制每个折的性能指标
    plot_fold_metrics(fold_metrics_df, args.output_dir)
    
    # 保存所有指标到CSV
    fold_metrics_df.to_csv(os.path.join(args.output_dir, 'fold_metrics.csv'), index=False)
    
    # 创建所有外部折结果的箱型图
    all_fold_data = []
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    metrics_map = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-score': 'f1',
        'AUC': 'roc_auc'
    }
    
    # 只使用测试结果数据
    for fold_metric in all_fold_metrics:
        standardized_metrics = {}
        for metric in metrics_to_plot:
            metric_key = metrics_map[metric]
            standardized_metrics[metric_key] = fold_metric.get(metric_key, 0.0)
        all_fold_data.append(standardized_metrics)
    
    if len(all_fold_data) > 0:
        avg_boxplot_save_path = plot_metrics_boxplot(pd.DataFrame(all_fold_data), metrics_to_plot, args.output_dir)
        print(f"所有折平均性能指标箱型图已保存至: {avg_boxplot_save_path}")
    
    # 创建results.csv文件，包含每一个外部折和最终平均结果的评估指标
    results_df = pd.DataFrame()
    
    # 所需指标列表
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    metrics_map = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-score': 'f1',
        'AUC': 'roc_auc'
    }
    
    # 添加每个外部折的指标（包括标准差）
    for i, fold_metric in enumerate(all_fold_metrics):
        # 创建外部折行
        fold_row = {}
        fold_row['fold'] = f'Fold {i+1}'
        for metric in metrics_list:
            metric_key = metrics_map[metric]
            if metric_key in fold_metric:
                fold_row[metric] = fold_metric[metric_key]
            else:
                fold_row[metric] = 0.0  # 如果没有该指标，设为0
        
        # 创建对应的标准差行
        std_row = {}
        std_row['fold'] = f'Fold {i+1} Std'
        
        # 如果有内部折指标，计算标准差
        if all_inner_fold_metrics and i < len(all_inner_fold_metrics) and len(all_inner_fold_metrics[i]) > 0:
            for metric in metrics_list:
                metric_key = metrics_map[metric]
                # 提取该外部折下所有内部折的对应指标值
                inner_metric_values = [inner_metric.get(metric_key, 0) for inner_metric in all_inner_fold_metrics[i] if metric_key in inner_metric]
                
                # 只有当有足够的内部折数据时才计算标准差
                if inner_metric_values and len(inner_metric_values) > 1:
                    std_row[metric] = np.std(inner_metric_values)
                else:
                    std_row[metric] = 0.0
        else:
            # 如果没有内部折数据，所有标准差设为0
            for metric in metrics_list:
                std_row[metric] = 0.0
        
        # 添加到结果DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([fold_row]), pd.DataFrame([std_row])], ignore_index=True)
    
    # 添加平均值和标准差
    avg_row = {'fold': 'Average'}
    std_row = {'fold': 'Std'}
    
    for metric in metrics_list:
        metric_key = metrics_map[metric]
        
        # 从所有折中收集指标值
        metric_values = [fold_metric.get(metric_key, 0) for fold_metric in all_fold_metrics if metric_key in fold_metric]
        
        if metric_values and len(metric_values) > 0:
            avg_row[metric] = np.mean(metric_values)
            if len(metric_values) > 1:
                std_row[metric] = np.std(metric_values)
            else:
                std_row[metric] = 0.0
        else:
            avg_row[metric] = 0.0
            std_row[metric] = 0.0
    
    # 添加最终平均结果行
    results_df = pd.concat([results_df, pd.DataFrame([avg_row]), pd.DataFrame([std_row])], ignore_index=True)
    
    # 保存results.csv到输出目录
    results_csv_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"评估指标已保存到 {results_csv_path}")
    
    # 打印最终结果
    print("\n最终评估结果:")
    for metric in metrics_list:
        print(f"{metric}: {avg_row[metric]:.4f} ± {std_row[metric]:.4f}")
    
    # 绘制所有折的ROC曲线
    if all_roc_data:
        roc_curves_path = plot_all_roc_curves(all_roc_data, args.output_dir)
        print(f"所有折的ROC曲线已保存至: {roc_curves_path}")
    
    return avg_metrics

def plot_fold_metrics(fold_metrics_df, output_dir):
    """
    绘制每个折的性能指标
    
    参数:
        fold_metrics_df: 包含每个折指标的DataFrame
        output_dir: 输出目录
    """
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 选择要绘制的指标
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
    metrics_map = {
        'Accuracy': 'accuracy',
        'Precision': 'precision', 
        'Recall': 'recall',
        'F1-score': 'f1',
        'AUC': 'roc_auc'
    }
    
    plt.figure(figsize=(15, 10))
        
    # 设置颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_to_plot)))
        
    # 准备绘图数据
    x = fold_metrics_df['outer_fold'].values
    
    # 绘制每个指标
    for i, metric in enumerate(metrics_to_plot):
        metric_key = metrics_map[metric]
        if metric_key in fold_metrics_df.columns:
            plt.plot(x, fold_metrics_df[metric_key], marker='o', label=metric, color=colors[i], linewidth=2)
    
    # 添加图例和标签
    plt.legend(fontsize=12, prop={'family': 'Times New Roman'})
    plt.xlabel('Outer Fold', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Score', fontsize=14, fontname='Times New Roman')
    plt.title('Performance Metrics Across Folds', fontsize=16, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylim(0.8, 1.00)  # 修改纵坐标上限为1.00
    
    # 为每个点添加数值标签
    for i, metric in enumerate(metrics_to_plot):
        metric_key = metrics_map[metric]
        if metric_key in fold_metrics_df.columns:
            for j, value in enumerate(fold_metrics_df[metric_key]):
                plt.text(x[j], value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=9, 
                         rotation=45, color=colors[i], fontname='Times New Roman')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_metrics.png'), dpi=300)
    plt.close()
    
    # 计算平均指标并绘制条形图
    plt.figure(figsize=(10, 6))
    
    # 准备平均指标数据
    avg_metrics = []
    std_metrics = []
    
    for metric in metrics_to_plot:
        metric_key = metrics_map[metric]
        if metric_key in fold_metrics_df.columns:
            avg_metrics.append(fold_metrics_df[metric_key].mean())
            std_metrics.append(fold_metrics_df[metric_key].std())
        else:
            avg_metrics.append(0)
            std_metrics.append(0)
    
    bars = plt.bar(range(len(metrics_to_plot)), avg_metrics, color=colors, alpha=0.7)
    
    # 添加误差条
    plt.errorbar(range(len(metrics_to_plot)), avg_metrics, yerr=std_metrics, fmt='none', color='black', capsize=5)
    
    # 添加数值标签
    def add_labels(bars, values):
        for bar, value, std in zip(bars, values, std_metrics):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                     f'{value:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10, fontname='Times New Roman')
    
    add_labels(bars, avg_metrics)
    
    # 添加图例和标签
    plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.ylim(0.8, 1.00)  # 修改纵坐标上限为1.00
    plt.xlabel('Metrics', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Average Score', fontsize=14, fontname='Times New Roman')
    plt.title('Average Performance Metrics', fontsize=16, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_metrics.png'), dpi=300)
    plt.close()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.random_state)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 添加模型信息到输出目录名称
    model_name_part = f"{args.model_type}"
    
    # 添加特征提取器信息到输出目录名称
    fe_part = "_with_fe" if args.use_feature_extractor else ""
    
    # 将所有部分组合成输出目录名称
    args.output_dir = f"{args.output_dir}_{model_name_part}{fe_part}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"将结果保存到: {args.output_dir}")
    
    # 打印模型和特征提取器设置
    print(f"\n模型类型: {args.model_type}")
    print(f"使用特征提取器: {'是' if args.use_feature_extractor else '否'}")
    print(f"震颤频带范围: {args.tremor_freq_band} Hz")
    print(f"采样率: {args.sampling_rate} Hz\n")
    
    # 加载文件列表
    file_list = load_file_list(args.data_path)
    
    # 打印数据集信息
    print(f"数据集大小: {len(file_list)}")
    print(f"标签分布: \n{file_list['label'].value_counts()}")
    
    # 执行嵌套交叉验证
    avg_metrics = nested_cross_validation(file_list, args.data_path, args)
    
    # 将平均指标保存到文件
    metrics_df = pd.DataFrame([avg_metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, 'avg_metrics.csv'), index=False)
    
    # 额外复制一份results.csv到项目根目录，确保能够轻松找到
    if os.path.exists(os.path.join(args.output_dir, 'results.csv')):
        results_df = pd.read_csv(os.path.join(args.output_dir, 'results.csv'))
        results_df.to_csv('results.csv', index=False)
        print("已将评估指标复制到项目根目录下的 results.csv")
    
    print("训练和评估完成!")

if __name__ == "__main__":
    main() 