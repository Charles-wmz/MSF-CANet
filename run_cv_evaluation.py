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
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

from model import ShallowCNN1D
from data_loader import load_processed_dataset
from train import train_epoch, validate
from evaluate import evaluate_model
from main import set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PADS dataset cross-validation')
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='../PADS/preprocessed', help='Preprocessed data path')
    parser.add_argument('--output_dir', type=str, default='results_cv', help='Results output directory')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Cross-validation parameters
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Feature extractor parameters
    parser.add_argument('--use_feature_extractor', action='store_true', help='Whether to use multi-scale hierarchical feature extractor')
    parser.add_argument('--use_multiscale', action='store_true', help='Whether to use multi-scale perception module')
    parser.add_argument('--use_frequency', action='store_true', help='Whether to use frequency perception module')
    parser.add_argument('--use_channel_attention', action='store_true', help='Whether to use channel attention module')
    parser.add_argument('--tremor_freq_band', type=str, default='3,7', help='Tremor frequency band range (Hz), separated by comma')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Signal sampling rate (Hz)')
    
    # Selection mode parameters
    parser.add_argument('--selection_mode', type=str, default='', choices=['', 'task', 'hand'], 
                       help='Selection mode: "" (all data), "task" (filter by task), "hand" (filter by hand)')
    parser.add_argument('--selection_value', type=str, default='', 
                       help='Selection value: for task mode: task ID (1-11), for hand mode: "left" or "right"')
    
    return parser.parse_args()

def train_and_evaluate_inner_fold(model, train_loader, val_loader, criterion, optimizer, args, device, inner_fold_dir):
    """
    Train and evaluate model in an inner fold of nested cross-validation
    
    Parameters:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        args: Command line arguments
        device: Device to use
        inner_fold_dir: Output directory for this inner fold
        
    Returns:
        float: Best validation accuracy
        str: Path to best model file
    """
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
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
            
            print(f"Saved new best model, validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered, best validation accuracy: {best_val_acc:.4f}")
            break
    
    return best_val_acc, best_model_path

def load_full_dataset(data_path, selection_mode='', selection_value=''):
    """
    Load full dataset with optional filtering
    
    Parameters:
        data_path: Path to preprocessed data
        selection_mode: Filter mode ('', 'task', or 'hand')
        selection_value: Value to filter by (task ID or hand name)
    
    Returns:
        Complete dataset, filtered if selection mode specified
    """
    # Load all subject data
    print("Loading dataset...")
    all_data = []
    all_labels = []
    subject_ids = []
    task_ids = []
    hands = []
    
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith('.pt'):
            # Extract metadata from filename
            # Format: subject_id-task_id-hand-pd_status.pt
            parts = filename.split('-')
            subject_id = int(parts[0])
            task_id = int(parts[1])
            hand = parts[2]
            pd_status = int(parts[3].split('.')[0])
            
            # Skip if doesn't match selection criteria
            if selection_mode == 'task' and task_id != int(selection_value):
                continue
            if selection_mode == 'hand' and hand != selection_value:
                continue
            
            # Load data
            data = torch.load(os.path.join(data_path, filename))
            
            # Store data and metadata
            all_data.append(data)
            all_labels.append(pd_status)
            subject_ids.append(subject_id)
            task_ids.append(task_id)
            hands.append(hand)
    
    # Convert to tensors
    X = torch.stack(all_data)
    y = torch.tensor(all_labels)
    subject_ids = torch.tensor(subject_ids)
    task_ids = torch.tensor(task_ids)
    
    # Convert hand labels to binary
    hands_binary = torch.zeros(len(hands))
    for i, hand in enumerate(hands):
        hands_binary[i] = 0 if hand == 'left' else 1
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X, y, subject_ids, task_ids, hands_binary)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"PD samples: {sum(all_labels)}, Non-PD samples: {len(all_labels) - sum(all_labels)}")
    
    if selection_mode:
        selection_str = f"task {selection_value}" if selection_mode == 'task' else f"{selection_value} hand"
        print(f"Selected data: {selection_mode} = {selection_str}")
    
    return dataset

def main():
    """Main function for cross-validation"""
    args = parse_args()
    
    # Process selection value
    if args.selection_mode == 'task':
        args.selection_value = int(args.selection_value)
    
    # Set random seed
    set_seed(args.random_state)
    
    # Set results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add feature extractor information to output directory name
    if args.use_feature_extractor:
        fe_parts = []
        if args.use_multiscale:
            fe_parts.append("multi")
        if args.use_frequency:
            fe_parts.append("freq")
        if args.use_channel_attention:
            fe_parts.append("attn")
        
        if fe_parts:
            fe_part = "_fe_" + "_".join(fe_parts)
        else:
            fe_part = "_fe_none"  # Basic feature extractor without modules
    else:
        fe_part = ""
        
    selection_suffix = ""
    
    if args.selection_mode:
        task_names = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", 
                     "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]
        
        if args.selection_mode == 'task':
            task_name = task_names[args.selection_value - 1]
            selection_suffix = f"_task_{task_name}"
        else:
            selection_suffix = f"_{args.selection_mode}_{args.selection_value}"
    
    args.output_dir = os.path.join(args.output_dir, f"cv{fe_part}{selection_suffix}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save parameters
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = load_full_dataset(args.data_path, args.selection_mode, args.selection_value)
    
    # Extract one batch to get input dimensions
    sample_data, _, _, _, _ = full_dataset[0]
    n_channels = sample_data.shape[0]
    n_timesteps = sample_data.shape[1]
    print(f"Data shape: channels={n_channels}, timesteps={n_timesteps}")
    
    # Get all data and labels for stratification
    all_data = []
    all_labels = []
    all_subjects = []
    
    for i in range(len(full_dataset)):
        _, label, subject, _, _ = full_dataset[i]
        all_labels.append(label.item())
        all_subjects.append(subject.item())
    
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    # Store results
    fold_metrics = []
    
    # Start cross-validation
    print(f"Starting {args.n_splits}-fold cross-validation...")
    
    for fold, (train_val_idx, test_idx) in enumerate(cv.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{args.n_splits}")
        print(f"{'='*50}")
        
        # Create fold directory
        fold_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        # We need to be careful to split by subjects, not just by samples
        test_subjects = np.unique(all_subjects[test_idx])
        
        # Create masks for test set based on subjects
        test_mask = np.isin(all_subjects, test_subjects)
        train_val_mask = ~test_mask
        
        # Get actual indices
        actual_test_idx = np.where(test_mask)[0]
        actual_train_val_idx = np.where(train_val_mask)[0]
        
        # Verify split maintains PD/non-PD balance
        train_val_labels = all_labels[actual_train_val_idx]
        test_labels = all_labels[actual_test_idx]
        
        print(f"Train/Val set: {len(actual_train_val_idx)} samples, {len(np.unique(all_subjects[actual_train_val_idx]))} subjects")
        print(f"  - PD: {np.sum(train_val_labels)}, Non-PD: {len(train_val_labels) - np.sum(train_val_labels)}")
        print(f"Test set: {len(actual_test_idx)} samples, {len(test_subjects)} subjects")
        print(f"  - PD: {np.sum(test_labels)}, Non-PD: {len(test_labels) - np.sum(test_labels)}")
        
        # Create test dataset
        test_dataset = torch.utils.data.Subset(full_dataset, actual_test_idx)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )
        
        # Inner cross-validation to find best hyperparameters
        inner_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
        
        inner_fold_metrics = []
        best_inner_fold = 0
        best_val_acc = 0.0
        best_model_path = None
        best_inner_fold_dir = None
        
        # Inner labels for stratification
        inner_labels = all_labels[actual_train_val_idx]
        
        # For each inner fold
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(np.zeros(len(inner_labels)), inner_labels)):
            print(f"\n--- Inner fold {inner_fold+1}/{args.n_splits} ---")
            
            # Get actual indices
            actual_train_idx = actual_train_val_idx[train_idx]
            actual_val_idx = actual_train_val_idx[val_idx]
            
            # Create datasets and loaders
            train_dataset = torch.utils.data.Subset(full_dataset, actual_train_idx)
            val_dataset = torch.utils.data.Subset(full_dataset, actual_val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
            
            # Create inner fold directory
            inner_fold_dir = os.path.join(fold_dir, f"inner_fold_{inner_fold+1}")
            os.makedirs(inner_fold_dir, exist_ok=True)
            
            # Create model for this fold
            model = ShallowCNN1D(
                n_classes=2,
                use_feature_extractor=args.use_feature_extractor,
                use_multiscale=args.use_multiscale,
                use_frequency=args.use_frequency,
                use_channel_attention=args.use_channel_attention,
                tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
                fs=args.sampling_rate
            ).to(device)
            
            # Create criterion and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # Train and evaluate inner fold
            inner_val_acc, inner_model_path = train_and_evaluate_inner_fold(
                model, train_loader, val_loader, criterion, optimizer, args, device, inner_fold_dir
            )
            
            # Track best inner fold
            if inner_val_acc > best_val_acc:
                best_val_acc = inner_val_acc
                best_inner_fold = inner_fold
                best_model_path = inner_model_path
                best_inner_fold_dir = inner_fold_dir
        
        print(f"\nBest inner fold: {best_inner_fold+1}, validation accuracy: {best_val_acc:.4f}")
        
        # Load best model for testing
        model = ShallowCNN1D(
            n_classes=2,
            use_feature_extractor=args.use_feature_extractor,
            use_multiscale=args.use_multiscale,
            use_frequency=args.use_frequency,
            use_channel_attention=args.use_channel_attention,
            tremor_freq_band=tuple(map(float, args.tremor_freq_band.split(','))),
            fs=args.sampling_rate
        ).to(device)
        
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        criterion = nn.CrossEntropyLoss()
        test_results_dir = os.path.join(fold_dir, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)
        
        print("\nEvaluating on test set...")
        test_metrics, outputs, targets, _ = evaluate_model(
            model, test_loader, criterion, device, results_dir=test_results_dir
        )
        
        test_metrics['fold'] = fold + 1
        fold_metrics.append(test_metrics)
        
        # Print results
        print(f"\nFold {fold+1} test results:")
        for metric, value in test_metrics.items():
            if metric != 'fold':
                print(f"{metric}: {value:.4f}")
    
    # Calculate average metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "fold_metrics.csv"), index=False)
    
    # Print average results
    print("\nCross-validation average results:")
    print("="*50)
    
    avg_metrics = {}
    std_metrics = {}
    
    for column in metrics_df.columns:
        if column != 'fold':
            avg_metrics[column] = metrics_df[column].mean()
            std_metrics[column] = metrics_df[column].std()
            print(f"{column}: {avg_metrics[column]:.4f} ± {std_metrics[column]:.4f}")
    
    print("="*50)
    
    # Save average metrics
    avg_results = {k: f"{v:.4f} ± {std_metrics[k]:.4f}" for k, v in avg_metrics.items()}
    with open(os.path.join(args.output_dir, "avg_results.json"), 'w') as f:
        json.dump(avg_results, f, indent=4)
    
    # Plot results
    plot_cv_results(metrics_df, args.output_dir)

def plot_cv_results(metrics_df, output_dir):
    """Plot cross-validation result charts"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    if 'auc' in metrics_df.columns:
        metrics.append('auc')
    
    # Plot line chart with all metrics by fold
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
    
    # Plot boxplot for each metric
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    data = [metrics_df[metric] for metric in metrics]
    labels = [metric.capitalize() for metric in metrics]
    
    # Create boxplot
    box = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Set colors
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan']
    for patch, color in zip(box['boxes'], colors[:len(metrics)]):
        patch.set_facecolor(color)
    
    plt.title('Performance Metrics Distribution')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add mean markers
    means = [metrics_df[metric].mean() for metric in metrics]
    plt.plot(range(1, len(means) + 1), means, 'rs', marker='D', markersize=8, label='Mean')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_boxplot.png'))
    plt.close()
    
    # Plot confusion matrix heatmap (average across folds)
    if 'conf_mat_00' in metrics_df.columns:
        plt.figure(figsize=(8, 6))
        
        # Calculate average confusion matrix
        conf_mat = np.zeros((2, 2))
        conf_mat[0, 0] = metrics_df['conf_mat_00'].mean()
        conf_mat[0, 1] = metrics_df['conf_mat_01'].mean()
        conf_mat[1, 0] = metrics_df['conf_mat_10'].mean()
        conf_mat[1, 1] = metrics_df['conf_mat_11'].mean()
        
        # Plot
        sns.heatmap(conf_mat, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=['Non-PD', 'PD'], yticklabels=['Non-PD', 'PD'])
        plt.title('Average Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'avg_confusion_matrix.png'))
        plt.close()
    
    print(f"Result plots saved to: {plots_dir}")

if __name__ == "__main__":
    main() 