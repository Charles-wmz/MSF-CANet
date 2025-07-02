import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
import pandas as pd
import os
from tqdm import tqdm
from model import ShallowCNN1D

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算各种性能指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于计算AUC）
        
    返回:
        dict: 包含各种指标的字典
    """
    metrics = {}
    
    # 准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 精确度、召回率、F1分数
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    
    # 计算每个类别的特异性（Specificity）
    cm = confusion_matrix(y_true, y_pred)
    specificity_list = []
    
    for i in range(cm.shape[0]):
        # True Negatives are all elements except those in row i and column i
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm[i, :], i))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    
    # 平均特异性
    metrics['specificity'] = np.mean(specificity_list)
    
    # 如果提供了预测概率，计算AUC
    if y_prob is not None:
        # 对于多类分类，通常使用one-vs-rest或one-vs-one方法来计算AUC
        # 这里我们使用sklearn的roc_auc_score
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        
        # 获取类别数量
        n_classes = y_prob.shape[1]
        
        # 将真实标签二值化
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # 如果是二分类
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        # 如果是多分类
        else:
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    
    return metrics

def evaluate_model(model, test_loader, device):
    """
    评估模型并返回预测结果、目标和概率
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备（CPU/GPU）
        
    返回:
        tuple: (预测结果, 目标, 概率)
    """
    # 设置模型为评估模式
    model.eval()
    
    # 初始化变量
    all_preds = []
    all_targets = []
    all_probs = []
    
    # 使用tqdm显示进度
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for data, target in test_bar:
            # 将数据移动到设备上
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 获取预测结果和概率
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # 收集预测、概率和目标
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    return all_preds, all_targets, all_probs

def evaluate_model_with_metrics(model, test_loader, criterion, device, results_dir='results'):
    """
    评估模型并计算性能指标
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备（CPU/GPU）
        results_dir: 结果保存目录
        
    返回:
        dict: 包含各种指标的字典
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 初始化变量
    test_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    # 使用tqdm显示进度
    test_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for data, target in test_bar:
            # 将数据移动到设备上
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            if hasattr(criterion, 'compute_tremor_weight'):
                # 使用Symptom-Aware Dynamic CE损失函数
                loss = criterion(output, target, data)
            else:
                # 使用常规损失函数
                loss = criterion(output, target)
            
            # 累加损失
            test_loss += loss.item()
            
            # 获取预测结果和概率
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # 收集预测、概率和目标
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 更新进度条
            test_bar.set_postfix(loss=loss.item())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 计算平均损失
    test_loss /= len(test_loader)
    
    # 计算性能指标
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics['loss'] = test_loss
    
    # 打印指标
    print("\n测试结果:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    if 'roc_auc' in metrics:
        print(f"AUC: {metrics['roc_auc']:.4f}")
    
    # 保存混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加文本
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    
    # 将指标保存为CSV
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    return metrics, all_preds, all_probs, all_targets

def compare_models(test_loader, device, base_model_path, time_attn_model_path, lightweight_time_attn_model_path=None, results_dir='comparison_results', time_dim=1000):
    """
    比较不同模型的性能
    
    参数:
        test_loader: 测试数据加载器
        device: 设备（CPU/GPU）
        base_model_path: 基础CNN模型路径
        time_attn_model_path: 时间注意力CNN模型路径
        lightweight_time_attn_model_path: 轻量级时间注意力CNN模型路径（可选）
        results_dir: 结果保存目录
        time_dim: 时间维度大小
        
    返回:
        tuple: 包含各模型指标的字典
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载基础模型
    base_model = ShallowCNN1D(n_classes=2)
    base_checkpoint = torch.load(base_model_path)
    base_model.load_state_dict(base_checkpoint['model_state_dict'])
    base_model = base_model.to(device)
    
    # 加载时间注意力模型
    time_attn_model = TimeAttentionCNN1D(n_classes=2, time_dim=time_dim)
    time_attn_checkpoint = torch.load(time_attn_model_path)
    time_attn_model.load_state_dict(time_attn_checkpoint['model_state_dict'])
    time_attn_model = time_attn_model.to(device)
    
    # 如果提供了轻量级时间注意力模型路径，也加载该模型
    lightweight_time_attn_model = None
    if lightweight_time_attn_model_path:
        lightweight_time_attn_model = LightweightTimeAttentionCNN1D(n_classes=2, time_dim=time_dim)
        lightweight_time_attn_checkpoint = torch.load(lightweight_time_attn_model_path)
        lightweight_time_attn_model.load_state_dict(lightweight_time_attn_checkpoint['model_state_dict'])
        lightweight_time_attn_model = lightweight_time_attn_model.to(device)
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 评估基础模型
    print("\n=== 评估基础CNN模型 ===")
    base_results_dir = os.path.join(results_dir, 'base_model')
    base_metrics, base_preds, base_probs, targets = evaluate_model_with_metrics(
        base_model, test_loader, criterion, device, results_dir=base_results_dir
    )
    
    # 评估时间注意力模型
    print("\n=== 评估时间注意力CNN模型 ===")
    time_attn_results_dir = os.path.join(results_dir, 'time_attn_model')
    time_attn_metrics, time_attn_preds, time_attn_probs, _ = evaluate_model_with_metrics(
        time_attn_model, test_loader, criterion, device, results_dir=time_attn_results_dir
    )
    
    # 如果有轻量级时间注意力模型，评估它
    lightweight_time_attn_metrics = None
    lightweight_time_attn_probs = None
    if lightweight_time_attn_model:
        print("\n=== 评估轻量级时间注意力CNN模型 ===")
        lightweight_time_attn_results_dir = os.path.join(results_dir, 'lightweight_time_attn_model')
        lightweight_time_attn_metrics, _, lightweight_time_attn_probs, _ = evaluate_model_with_metrics(
            lightweight_time_attn_model, test_loader, criterion, device, results_dir=lightweight_time_attn_results_dir
        )
    
    # 比较关键指标
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
    comparison_data = []
    
    for metric in metrics_to_compare:
        if metric in base_metrics and metric in time_attn_metrics:
            row = {
                'Metric': metric,
                'Base Model': base_metrics[metric],
                'Time Attention Model': time_attn_metrics[metric]
            }
            
            # 计算标准时间注意力的改进
            ta_improvement = time_attn_metrics[metric] - base_metrics[metric]
            ta_improvement_percent = (ta_improvement / base_metrics[metric]) * 100 if base_metrics[metric] != 0 else float('inf')
            row['TA Improvement'] = ta_improvement
            row['TA Improvement (%)'] = ta_improvement_percent
            
            # 如果有轻量级时间注意力模型，添加其指标
            if lightweight_time_attn_metrics and metric in lightweight_time_attn_metrics:
                row['Lightweight TA Model'] = lightweight_time_attn_metrics[metric]
                
                # 计算轻量级时间注意力的改进（相对于基础模型）
                lta_improvement = lightweight_time_attn_metrics[metric] - base_metrics[metric]
                lta_improvement_percent = (lta_improvement / base_metrics[metric]) * 100 if base_metrics[metric] != 0 else float('inf')
                row['LTA Improvement'] = lta_improvement
                row['LTA Improvement (%)'] = lta_improvement_percent
                
                # 计算轻量级相对于标准注意力的差异
                lta_vs_ta = lightweight_time_attn_metrics[metric] - time_attn_metrics[metric]
                lta_vs_ta_percent = (lta_vs_ta / time_attn_metrics[metric]) * 100 if time_attn_metrics[metric] != 0 else float('inf')
                row['LTA vs TA'] = lta_vs_ta
                row['LTA vs TA (%)'] = lta_vs_ta_percent
            
            comparison_data.append(row)
    
    # 创建比较表格
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    # 绘制性能比较图
    plt.figure(figsize=(14, 8))
    
    # 准备数据
    metrics_names = comparison_df['Metric'].tolist()
    base_values = comparison_df['Base Model'].tolist()
    time_attn_values = comparison_df['Time Attention Model'].tolist()
    
    x = np.arange(len(metrics_names))
    width = 0.25  # 减小宽度以适应更多条
    
    # 绘制条形图
    plt.bar(x - width, base_values, width, label='Base CNN')
    plt.bar(x, time_attn_values, width, label='Time Attention CNN')
    
    # 如果有轻量级时间注意力模型，添加其条形
    if lightweight_time_attn_metrics:
        lightweight_time_attn_values = comparison_df['Lightweight TA Model'].tolist()
        plt.bar(x + width, lightweight_time_attn_values, width, label='Lightweight Time Attention CNN')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison: CNN Models')
    plt.xticks(x, metrics_names)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(base_values):
        plt.text(i - width, v + 0.01, f"{v:.4f}", ha='center')
    
    for i, v in enumerate(time_attn_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    if lightweight_time_attn_metrics:
        for i, v in enumerate(lightweight_time_attn_values):
            plt.text(i + width, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_comparison.png'))
    
    # 如果有AUC，绘制ROC曲线对比
    if 'roc_auc' in base_metrics and 'roc_auc' in time_attn_metrics:
        plt.figure(figsize=(8, 6))
        
        # 假设为二分类问题
        fpr_base, tpr_base, _ = roc_curve(targets, base_probs[:, 1])
        roc_auc_base = auc(fpr_base, tpr_base)
        
        fpr_time_attn, tpr_time_attn, _ = roc_curve(targets, time_attn_probs[:, 1])
        roc_auc_time_attn = auc(fpr_time_attn, tpr_time_attn)
        
        plt.plot(fpr_base, tpr_base, label=f'Base CNN (AUC = {roc_auc_base:.4f})')
        plt.plot(fpr_time_attn, tpr_time_attn, label=f'Time Attention CNN (AUC = {roc_auc_time_attn:.4f})')
        
        # 如果有轻量级时间注意力模型，添加其ROC曲线
        if lightweight_time_attn_metrics and lightweight_time_attn_probs is not None:
            fpr_lightweight, tpr_lightweight, _ = roc_curve(targets, lightweight_time_attn_probs[:, 1])
            roc_auc_lightweight = auc(fpr_lightweight, tpr_lightweight)
            plt.plot(fpr_lightweight, tpr_lightweight, label=f'Lightweight TA CNN (AUC = {roc_auc_lightweight:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, 'roc_curve_comparison.png'))
    
    print("\n=== 模型比较结果 ===")
    print(comparison_df.to_string(index=False))
    
    # 返回所有模型的指标
    result = {'base': base_metrics, 'time_attn': time_attn_metrics}
    if lightweight_time_attn_metrics:
        result['lightweight_time_attn'] = lightweight_time_attn_metrics
    
    return result 