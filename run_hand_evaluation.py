import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# 动态导入模型
import importlib

from data_loader_task_hand import load_selective_data
from train import train_epoch, validate
from evaluate import evaluate_model, calculate_metrics
from main import set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PADS dataset evaluation by hand classification')
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='../PADS/preprocessed', help='Preprocessed data path')
    parser.add_argument('--output_dir', type=str, default='results_hand_evaluation', help='Results save directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Cross-validation parameters
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='full', 
                       choices=['base', 'multiscale', 'frequency', 'attention', 'full'], 
                       help='模型类型: base(基础模型), multiscale(多尺度模型), frequency(频域模型), attention(注意力模型), full(完整模型)')
    
    # Feature extractor parameters
    parser.add_argument('--use_feature_extractor', action='store_true', help='Whether to use feature extractor')
    parser.add_argument('--tremor_freq_band', type=str, default='3,7', help='Tremor frequency band range (Hz), comma separated')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Signal sampling rate (Hz)')
    
    # Evaluation combination parameters
    parser.add_argument('--evaluate_combined', action='store_true', help='Whether to additionally evaluate combined left-right hand model')
    
    return parser.parse_args()

def train_hand_model(hand, args):
    """
    Train a specific hand model using nested cross-validation
    
    Parameters:
        hand: 'left', 'right' or 'combined'
        args: command line arguments
        
    Returns:
        dict: average test performance metrics
    """
    # Set random seed
    set_seed(args.random_state)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if hand == 'left':
        hand_name = "Left Hand"
    elif hand == 'right':
        hand_name = "Right Hand"
    else:
        hand_name = "Combined Hands"
    
    print(f"\n{'='*50}")
    print(f"Starting nested cross-validation for {hand_name} model")
    print(f"{'='*50}")
    
    # Load selective data
    selection_mode = 'hand'
    selection_value = hand
    
    # If combined model, use all channels
    if hand == 'combined':
        selection_mode = 'all'
        selection_value = None
    
    # Load data and get full dataset
    _, _, _, n_classes, n_channels, full_dataset = load_selective_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
        selection_mode=selection_mode,
        selection_value=selection_value,
        return_dataset=True
    )
    
    print(f"Number of selected channels: {n_channels}")
    print(f"Dataset size: {len(full_dataset)} samples")
    
    # Create model directory for this selection
    model_dir = os.path.join(args.output_dir, f"hand_{hand}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create outer cross-validation
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    all_fold_metrics = []
    all_y_true = []
    all_y_pred = []
    
    # 根据model_type动态导入相应模型
    if args.model_type == 'base':
        from model_base import ShallowCNN1D
    elif args.model_type == 'multiscale':
        from model_multiscale import ShallowCNN1D
    elif args.model_type == 'frequency':
        from model_frequency import ShallowCNN1D
    elif args.model_type == 'attention':
        from model_attention import ShallowCNN1D
    else:  # 'full'
        from model import ShallowCNN1D
    
    # Outer cross-validation
    for i, (train_val_idx, test_idx) in enumerate(outer_cv.split(np.arange(len(full_dataset)))):
        print(f"\n---- Outer Fold {i+1}/{args.n_splits} ----")
        
        # Split dataset
        train_val_dataset = torch.utils.data.Subset(full_dataset, train_val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        
        # Create inner cross-validation
        inner_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
        
        best_inner_idx = 0
        best_val_accuracy = 0.0
        best_checkpoint_dir = None
        
        # Inner cross-validation
        for j, (train_idx, val_idx) in enumerate(inner_cv.split(np.arange(len(train_val_dataset)))):
            print(f"\n---- Inner Fold {j+1}/{args.n_splits} ----")
            
            # Split inner fold data
            train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, 
                pin_memory=True, generator=torch.Generator().manual_seed(args.random_state)
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
            
            # Reset random seed to ensure model weight initialization is the same
            torch.manual_seed(args.random_state)
            torch.cuda.manual_seed(args.random_state)
            
            # Create model
            model = ShallowCNN1D(
                n_channels=n_channels,
                n_classes=n_classes,
                use_feature_extractor=args.use_feature_extractor,
                tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
                fs=args.sampling_rate
            ).to(device)
            
            # Create loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
            
            # Create directory for current inner fold
            inner_fold_dir = os.path.join(model_dir, f'outer_{i+1}_inner_{j+1}')
            os.makedirs(inner_fold_dir, exist_ok=True)
            
            # Train model
            best_val_acc = train_and_evaluate_inner_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, inner_fold_dir, args.patience, args.epochs)
            
            # Update best inner fold
            if best_val_acc > best_val_accuracy:
                best_val_accuracy = best_val_acc
                best_inner_idx = j
                best_checkpoint_dir = inner_fold_dir
        
        # Recreate best model for testing
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed(args.random_state)
        
        best_model = ShallowCNN1D(
            n_channels=n_channels,
            n_classes=n_classes,
            use_feature_extractor=args.use_feature_extractor,
            tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
            fs=args.sampling_rate
        ).to(device)
        
        # Load best model weights
        checkpoint = torch.load(os.path.join(best_checkpoint_dir, 'best_model.pth'))
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create directory for test results
        test_results_dir = os.path.join(model_dir, f'outer_{i+1}_test')
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Evaluate on test set
        print(f"\nEvaluating best inner model (inner fold {best_inner_idx+1}) on test set")
        y_pred, y_true, y_proba = evaluate_model(best_model, test_loader, device)
        test_metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        # Collect true labels and predictions
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Add fold index to metrics
        test_metrics['fold'] = i+1
        all_fold_metrics.append(test_metrics)
        
        # Print current fold results
        print(f"\nOuter fold {i+1} evaluation results:")
        print(f"{'='*30}")
        for k, v in test_metrics.items():
            if k != 'fold':
                print(f"{k}: {v:.4f}")
        print(f"{'='*30}")
    
    # Calculate average metrics across all folds
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # Save all fold results
    results_path = os.path.join(model_dir, "all_fold_results.csv")
    metrics_df.to_csv(results_path, index=False)
    
    # Calculate average metrics
    avg_metrics = metrics_df.mean(numeric_only=True).to_dict()
    std_metrics = metrics_df.std(numeric_only=True).to_dict()
    
    # Remove 'fold' column
    if 'fold' in avg_metrics:
        del avg_metrics['fold']
    if 'fold' in std_metrics:
        del std_metrics['fold']
    
    # Print average metrics
    print(f"\n{hand_name} cross-validation average results:")
    print("="*50)
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f} ± {std_metrics[k]:.4f}")
    print("="*50)
    
    # Save average metrics
    avg_results = {k: f"{v:.4f} ± {std_metrics[k]:.4f}" for k, v in avg_metrics.items()}
    with open(os.path.join(model_dir, "avg_results.json"), 'w') as f:
        json.dump(avg_results, f, indent=4)
    
    # Add hand information to metrics
    avg_metrics['hand'] = hand
    
    # Plot cross-validation result charts
    plot_cv_results(metrics_df, model_dir)
    
    # Merge all test set true labels and predictions for overall evaluation
    np.save(os.path.join(model_dir, 'all_y_true.npy'), np.array(all_y_true))
    np.save(os.path.join(model_dir, 'all_y_pred.npy'), np.array(all_y_pred))
    
    return avg_metrics, all_y_true, all_y_pred, None

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.random_state)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 添加模型信息到输出目录名称
    model_name_part = f"{args.model_type}"
    
    # 添加特征提取器信息到输出目录名称
    fe_part = "_with_fe" if args.use_feature_extractor else ""
    
    # 将所有部分组合成输出目录名称
    args.output_dir = f"{args.output_dir}_{model_name_part}{fe_part}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Saving results to: {args.output_dir}")
    
    # 打印模型和特征提取器设置
    print(f"\n模型类型: {args.model_type}")
    print(f"使用特征提取器: {'是' if args.use_feature_extractor else '否'}")
    print(f"震颤频带范围: {args.tremor_freq_band} Hz")
    print(f"采样率: {args.sampling_rate} Hz\n")
    
    # Save parameters
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # All evaluation results
    all_results = []
    
    # Determine which hands to evaluate
    hands = ['left', 'right']
    if args.evaluate_combined:
        hands.append('combined')
    
    hand_names = ["Left", "Right"]
    if args.evaluate_combined:
        hand_names.append("Combined")
    
    # Train and evaluate for each hand
    for hand, name in zip(hands, hand_names):
        metrics, _, _, _ = train_hand_model(hand, args)
        all_results.append(metrics)
        
        print(f"\n{name} Evaluation Results:")
        print(f"{'='*30}")
        for k, v in metrics.items():
            if k != 'hand':
                print(f"{k}: {v:.4f}")
        print(f"{'='*30}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nAll evaluation results saved to: {results_path}")
    
    # Plot comparison charts
    plot_comparison(results_df, args.output_dir, hand_names)

def plot_cv_results(metrics_df, output_dir):
    """Plot cross-validation result charts"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    if 'auc' in metrics_df.columns:
        metrics.append('auc')
    
    # Plot line chart for each metric
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        plt.plot(metrics_df['fold'], metrics_df[metric], marker='o', label=metric.capitalize())
    
    plt.title('Performance Metrics Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.xticks(metrics_df['fold'])
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'fold_metrics.png'))
    plt.close()
    
    # Plot boxplot
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    data = [metrics_df[metric] for metric in metrics]
    labels = [metric.capitalize() for metric in metrics]
    
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title('Distribution of Performance Metrics')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_boxplot.png'))
    plt.close()
    
    # Plot individual bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Get data
        values = metrics_df[metric].values
        folds = metrics_df['fold'].values
        
        # Plot bar chart
        bars = plt.bar(range(len(folds)), values, color='skyblue')
        
        # Set chart
        plt.xlabel('Fold')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Values Across Folds')
        plt.xticks(range(len(folds)), [f"Fold {int(fold)}" for fold in folds])
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean line
        mean_value = values.mean()
        plt.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.4f}')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_folds.png'))
        plt.close()
    
    print(f"Cross-validation result charts saved to: {plots_dir}")

def plot_comparison(results_df, output_dir, hand_names):
    """Plot performance comparison charts for different hands"""
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    if 'auc' in results_df.columns:
        metrics.append('auc')
    
    # Create charts directory
    plots_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Get metric values
        values = results_df[metric].values
        
        # Plot bar chart
        bars = plt.bar(range(len(hand_names)), values, color='skyblue')
        
        # Set chart
        plt.xlabel('Hand')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(range(len(hand_names)), hand_names)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # Create radar chart
    n_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the shape
    
    for i, hand in enumerate(hand_names):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Get current hand metrics
        values = results_df.iloc[i][metrics].values.tolist()
        values += values[:1]  # Close the shape
        
        # Plot radar chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=hand)
        ax.fill(angles, values, alpha=0.25)
        
        # Set radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1)
        plt.title(f'{hand} Performance Radar Chart')
        
        # Save radar chart
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{hand}_radar.png'))
        plt.close()
    
    # Create combined radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot radar chart for each hand
    for i, hand in enumerate(hand_names):
        values = results_df.iloc[i][metrics].values.tolist()
        values += values[:1]  # Close the shape
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=hand)
        ax.fill(angles, values, alpha=0.1)
    
    # Set combined radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right')
    plt.title('Hand Classification Performance Comparison Radar Chart')
    
    # Save combined radar chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_radar.png'))
    plt.close()
    
    # Create combined comparison chart
    plt.figure(figsize=(12, 8))
    
    # Set bar chart positions
    x = np.arange(len(hand_names))
    width = 0.15
    positions = [x + i*width for i in range(len(metrics))]
    
    # Plot bar chart for each metric
    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        plt.bar(positions[i], values, width, label=metric.capitalize())
    
    # Set chart
    plt.xlabel('Hand')
    plt.ylabel('Metric Value')
    plt.title('Overall Performance Metrics Comparison')
    plt.xticks(x + width*(len(metrics)-1)/2, hand_names)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_comparison.png'))
    plt.close()
    
    print(f"Comparison charts saved to: {plots_dir}")

def train_and_evaluate_inner_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, inner_fold_dir, patience, epochs):
    """
    Train and evaluate model on inner fold of nested cross-validation
    
    Parameters:
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        device: computation device
        inner_fold_dir: directory to save model checkpoints
        patience: early stopping patience
        epochs: maximum number of epochs
        
    Returns:
        best_val_acc: best validation accuracy
    """
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(inner_fold_dir, 'best_model.pth')
    
    for epoch in range(epochs):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_path)
            
            print(f"Saved new best model, validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered, best validation accuracy: {best_val_acc:.4f}")
            break
    
    return best_val_acc

if __name__ == "__main__":
    main() 