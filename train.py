import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备（CPU/GPU）
        
    返回:
        float: 平均训练损失
    """
    model.train()
    train_loss = 0.0
    
    # 使用tqdm显示训练进度
    train_bar = tqdm(train_loader, desc="Training")
    
    for data, target in train_bar:
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 累加损失
        train_loss += loss.item()
        
        # 更新进度条
        train_bar.set_postfix(loss=loss.item())
    
    # 计算平均损失
    train_loss /= len(train_loader)
    
    return train_loss

def validate(model, val_loader, criterion, device):
    """
    验证模型
    
    参数:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备（CPU/GPU）
        
    返回:
        tuple: (平均验证损失, 验证准确率)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            # 将数据移动到设备上
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 累加损失
            val_loss += loss.item()
            
            # 获取预测结果
            _, preds = torch.max(output, 1)
            
            # 收集预测和目标
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算平均损失
    val_loss /= len(val_loader)
    
    # 计算准确率
    accuracy = accuracy_score(all_targets, all_preds)
    
    return val_loss, accuracy, all_preds, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=50, patience=10, checkpoint_dir='checkpoints'):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备（CPU/GPU）
        num_epochs: 训练轮数
        patience: 早停耐心值
        checkpoint_dir: 模型检查点保存目录
        
    返回:
        model: 训练好的模型
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化最佳验证损失和准确率
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    
    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_accuracy, _, _ = validate(model, val_loader, criterion, device)
        
        # 记录损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # 检查是否是最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            print(f"保存了新的最佳模型，验证准确率: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"早停: {patience}个epoch没有改善")
            break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_curves.png'))
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_losses, val_losses, val_accuracies 