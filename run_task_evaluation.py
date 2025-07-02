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
    parser = argparse.ArgumentParser(description='PADS dataset evaluation by task classification')
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='../PADS/preprocessed', help='Preprocessed data path')
    parser.add_argument('--output_dir', type=str, default='results_task_evaluation', help='Results output directory')
    
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
    parser.add_argument('--tremor_freq_band', type=str, default='3,7', help='Tremor frequency band range (Hz), separated by comma')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Signal sampling rate (Hz)')
    
    # Evaluation mode parameters
    parser.add_argument('--eval_mode', type=str, default='task', choices=['task', 'hand'], 
                       help='Evaluation mode: task (evaluate by individual tasks), hand (evaluate by left/right hand)')
    
    return parser.parse_args()

def train_model_for_selection(selection_mode, selection_value, args):
    """
    Train model for specific selection (task or hand) using nested cross-validation
    
    Parameters:
        selection_mode: Selection mode ('task' or 'hand')
        selection_value: Selection value (task ID or hand name)
        args: Command line arguments
        
    Returns:
        dict: Average test performance metrics
    """
    # Set random seed
    set_seed(args.random_state)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Task names list
    task_names = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
                  "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
    
    if selection_mode == 'task':
        selection_name = task_names[selection_value-1]
    else:
        selection_name = f"{selection_value} Hand"
        
    print(f"\n{'='*50}")
    print(f"Starting nested cross-validation for {selection_name} model")
    print(f"{'='*50}")
    
    # Load selective data and get full dataset
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
    if selection_mode == 'task':
        selection_dir = os.path.join(args.output_dir, f"task_{selection_name}")
    else:
        selection_dir = os.path.join(args.output_dir, f"{selection_mode}_{selection_value}")
    os.makedirs(selection_dir, exist_ok=True)
    
    # Create outer cross-validation
    outer_cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    all_fold_metrics = []
    
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
        
        # Split datasets
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
            print(f"---- Inner Fold {j+1}/{args.n_splits} ----")
            
            # Split train and validation
            train_dataset = torch.utils.data.Subset(train_val_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(train_val_dataset, val_idx)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
            
            # Get single batch to determine input size
            x_sample, _ = next(iter(train_loader))
            input_size = x_sample.shape[2]
            
            # Create model
            model = ShallowCNN1D(
                n_channels=n_channels,
                n_classes=n_classes,
                use_feature_extractor=args.use_feature_extractor,
                tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
                fs=args.sampling_rate
            ).to(device)
            
            # Create criterion and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
            
            # Create inner fold directory
            inner_fold_dir = os.path.join(selection_dir, f"outer_{i+1}_inner_{j+1}")
            os.makedirs(inner_fold_dir, exist_ok=True)
            
            # Train model
            best_val_acc = 0.0
            patience_counter = 0
            best_model_path = os.path.join(inner_fold_dir, 'best_model.pth')
            
            for epoch in range(args.epochs):
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
                    
                    print(f"Inner fold {j+1} - Saved new best model, validation accuracy: {val_acc:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= args.patience:
                    print(f"Inner fold {j+1} - Early stopping triggered, best validation accuracy: {best_val_acc:.4f}")
                    break
            
            # Update best inner fold
            if best_val_acc > best_val_accuracy:
                best_val_accuracy = best_val_acc
                best_inner_idx = j
                best_checkpoint_dir = inner_fold_dir
        
        # Load best model
        best_model_path = os.path.join(best_checkpoint_dir, 'best_model.pth')
        
        # Create a new model instance
        model = ShallowCNN1D(
            n_channels=n_channels,
            n_classes=n_classes,
            use_feature_extractor=args.use_feature_extractor,
            tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
            fs=args.sampling_rate
        ).to(device)
        
        # Load best checkpoint
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        print(f"\nEvaluating best inner model (inner fold {best_inner_idx+1}) on test set")
        y_pred, y_true, y_proba = evaluate_model(model, test_loader, device)
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        metrics['fold'] = i + 1
        
        # Save metrics
        all_fold_metrics.append(metrics)
        
        # Save fold results
        fold_results_dir = os.path.join(selection_dir, f"outer_fold_{i+1}_results")
        os.makedirs(fold_results_dir, exist_ok=True)
        
        with open(os.path.join(fold_results_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Print fold results
        print(f"\nOuter Fold {i+1} Test Results:")
        for k, v in metrics.items():
            if k != 'fold':
                print(f"{k}: {v:.4f}")
    
    # Create and save metrics dataframe
    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(os.path.join(selection_dir, "fold_metrics.csv"), index=False)
    
    # Calculate average metrics
    avg_metrics = {}
    std_metrics = {}
    
    # Calculate average and std metrics
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'fold':
            avg_metrics[col] = metrics_df[col].mean()
            std_metrics[col] = metrics_df[col].std()
    
    # Save average metrics
    avg_results = {}
    for k in avg_metrics.keys():
        avg_results[k] = f"{avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}"
    
    with open(os.path.join(selection_dir, "avg_metrics.json"), 'w') as f:
        json.dump(avg_results, f, indent=4)
    
    # Print average metrics
    print(f"\n{selection_name} cross-validation average results:")
    print(f"{'='*50}")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f} ± {std_metrics[k]:.4f}")
    print(f"{'='*50}")
    
    # Add selection information to metrics
    avg_metrics['selection_mode'] = selection_mode
    avg_metrics['selection_value'] = selection_value
    
    return avg_metrics

def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.random_state)
    
    # Set results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 添加模型信息到输出目录名称
    model_name_part = f"{args.model_type}"
    
    # 添加特征提取器信息到输出目录名称
    fe_part = "_with_fe" if args.use_feature_extractor else ""
    
    # 将所有部分组合成输出目录名称
    args.output_dir = os.path.join(args.output_dir, f"{args.eval_mode}_{model_name_part}{fe_part}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Task names list
    task_names = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
                  "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
    
    # Determine selections to evaluate
    if args.eval_mode == 'task':
        selections = [(args.eval_mode, i) for i in range(1, 12)]  # Tasks 1-11
        selection_names = [f"{task_names[i-1]}" for i in range(1, 12)]
    else:  # 'hand'
        selections = [(args.eval_mode, hand) for hand in ['left', 'right']]
        selection_names = ["Left", "Right"]
    
    # Train and evaluate for each selection
    for (mode, value), name in zip(selections, selection_names):
        metrics = train_model_for_selection(mode, value, args)
        all_results.append(metrics)
        
        print(f"\n{name} Evaluation Results:")
        print(f"{'='*30}")
        for k, v in metrics.items():
            if k not in ['selection_mode', 'selection_value']:
                print(f"{k}: {v:.4f}")
        print(f"{'='*30}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(args.output_dir, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nAll evaluation results saved to: {results_path}")
    
    # Plot comparison charts
    plot_comparison(results_df, args.output_dir, selection_names)

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
    
    # Create boxplot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Set colors
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan', 'lightgray']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set chart
    plt.title('Performance Metrics Distribution')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add mean markers
    means = [metrics_df[metric].mean() for metric in metrics]
    plt.plot(range(1, len(means) + 1), means, 'rs', marker='D', markersize=8, label='Mean')
    plt.legend()
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_boxplot.png'))
    plt.close()
    
    print(f"Charts saved to: {plots_dir}")

def plot_comparison(results_df, output_dir, selection_names):
    """Plot performance comparison charts for each selection"""
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    if 'auc' in results_df.columns:
        metrics.append('auc')
    
    # Create charts directory
    plots_dir = os.path.join(output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Get metric values
        values = results_df[metric].values
        
        # Plot bar chart
        bars = plt.bar(range(len(selection_names)), values, color='skyblue')
        
        # Set chart
        plt.xlabel('Selection')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(range(len(selection_names)), selection_names, rotation=45, ha='right')
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
    
    # Create combined comparison chart
    plt.figure(figsize=(14, 8))
    
    # Set bar chart positions
    x = np.arange(len(selection_names))
    width = 0.15
    positions = [x + i*width for i in range(len(metrics))]
    
    # Plot bar chart for each metric
    for i, metric in enumerate(metrics):
        values = results_df[metric].values
        plt.bar(positions[i], values, width, label=metric.capitalize())
    
    # Set chart
    plt.xlabel('Selection')
    plt.ylabel('Metric Value')
    plt.title('Overall Performance Metrics Comparison')
    plt.xticks(x + width*(len(metrics)-1)/2, selection_names)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'overall_comparison.png'))
    plt.close()
    
    print(f"Comparison charts saved to: {plots_dir}")

if __name__ == "__main__":
    main() 