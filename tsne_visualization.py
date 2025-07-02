import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime

# 导入自定义模块
from model import ShallowCNN1D
from data_loader import PADSDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize model features using t-SNE')
    
    # Path parameters
    parser.add_argument('--data_path', type=str, default='../PADS/preprocessed', help='Path to preprocessed data')
    parser.add_argument('--output_dir', type=str, default='tsne_results', help='Directory to save visualization results')
    parser.add_argument('--model_path', type=str, default='./results_full_with_fe_20250527_091616/outer_fold_1_inner_fold_1/best_model.pth', help='Path to model weights, use random weights if None')
    
    # t-SNE parameters
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE')
    parser.add_argument('--n_iter', type=int, default=1000, help='Number of iterations for t-SNE')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--use_feature_extractor', action='store_true', help='Whether to use feature extractor')
    parser.add_argument('--tremor_freq_band', type=str, default='3,7', help='Tremor frequency band range (Hz), comma separated')
    parser.add_argument('--sampling_rate', type=int, default=100, help='Signal sampling rate (Hz)')
    
    return parser.parse_args()

def load_file_list(data_path):
    """
    Load file list
    
    Args:
        data_path: Data path
        
    Returns:
        pandas.DataFrame: DataFrame containing file information
    """
    # 加载文件列表
    file_list_path = os.path.join(data_path, "file_list.csv")
    if os.path.exists(file_list_path):
        file_list = pd.read_csv(file_list_path)
        # 过滤只保留标签为0和1的样本
        file_list = file_list[file_list['label'].isin([0, 1])].reset_index(drop=True)
        return file_list
    else:
        raise FileNotFoundError(f"File list not found: {file_list_path}")

def extract_features(model, dataloader, device, extract_type='both'):
    """
    Extract raw features and model's intermediate features
    
    Args:
        model: Neural network model
        dataloader: Data loader
        device: Computing device
        extract_type: Extraction type, can be 'input', 'feature', or 'both'
        
    Returns:
        tuple: (raw_features, model_features, labels)
    """
    model.eval()  # Set to evaluation mode
    
    # Create storage arrays
    raw_features = []
    model_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Extracting features"):
            # Move data to device
            data = data.to(device)
            
            # Store raw input data
            if extract_type in ['input', 'both']:
                # Get average features of raw data as simplified representation
                avg_features = torch.mean(data, dim=2).cpu().numpy()  # [batch_size, channels]
                raw_features.append(avg_features)
            
            # Extract model intermediate features
            if extract_type in ['feature', 'both']:
                # Forward pass
                if model.use_feature_extractor:
                    # First through feature extractor
                    enhanced_features = model.feature_extractor(data)
                    
                    # Through first convolutional layer
                    x = model.conv1(enhanced_features)
                    x = model.bn1(x)
                    x = nn.functional.relu(x)
                    x = model.pool1(x)
                    
                    # Through second convolutional layer
                    x = model.conv2(x)
                    x = model.bn2(x)
                    x = nn.functional.relu(x)
                    x = model.pool2(x)
                    
                    # Through third convolutional layer
                    x = model.conv3(x)
                    x = model.bn3(x)
                    x = nn.functional.relu(x)
                    x = model.pool3(x)
                    
                    # Global pooling to get features
                    x = model.global_pool(x)
                    x = x.view(x.size(0), -1)  # [batch_size, 256]
                    
                    # Fully connected layer for final feature representation
                    features = model.fc1(x)  # [batch_size, 128]
                else:
                    # Directly through model's convolutional layers
                    x = model.conv1(data)
                    x = model.bn1(x)
                    x = nn.functional.relu(x)
                    x = model.pool1(x)
                    
                    # Through second convolutional layer
                    x = model.conv2(x)
                    x = model.bn2(x)
                    x = nn.functional.relu(x)
                    x = model.pool2(x)
                    
                    # Through third convolutional layer
                    x = model.conv3(x)
                    x = model.bn3(x)
                    x = nn.functional.relu(x)
                    x = model.pool3(x)
                    
                    # Global pooling to get features
                    x = model.global_pool(x)
                    x = x.view(x.size(0), -1)  # [batch_size, 256]
                    
                    # Fully connected layer for final feature representation
                    features = model.fc1(x)  # [batch_size, 128]
                
                model_features.append(features.cpu().numpy())
            
            # Store labels
            all_labels.append(labels.numpy())
    
    # Merge batch data
    if extract_type in ['input', 'both']:
        raw_features = np.vstack(raw_features)
    else:
        raw_features = None
        
    if extract_type in ['feature', 'both']:
        model_features = np.vstack(model_features)
    else:
        model_features = None
        
    all_labels = np.concatenate(all_labels)
    
    return raw_features, model_features, all_labels

def apply_tsne(features, labels, perplexity=30.0, n_iter=1000, random_state=42):
    """
    Apply t-SNE dimensionality reduction
    
    Args:
        features: Feature array, shape [n_samples, n_features]
        labels: Label array, shape [n_samples]
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        random_state: Random seed
        
    Returns:
        numpy.ndarray: Reduced features, shape [n_samples, 2]
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, verbose=1)
    tsne_result = tsne.fit_transform(scaled_features)
    
    return tsne_result

def visualize_tsne(tsne_result, labels, title, save_path):
    """
    Visualize t-SNE results
    
    Args:
        tsne_result: t-SNE dimensionality reduction result, shape [n_samples, 2]
        labels: Label array, shape [n_samples]
        title: Image title
        save_path: Save path
    """
    plt.figure(figsize=(12, 10))
    
    # Define class names and colors
    class_names = ['HC', 'PD']
    colors = ['#2ca02c', '#d62728']  # Green for healthy controls, red for Parkinson's
    markers = ['o', 's']  # Circle for healthy controls, square for Parkinson's
    
    # Draw scatter plot for each class
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        plt.scatter(
            tsne_result[idx, 0], 
            tsne_result[idx, 1],
            c=colors[i],
            marker=markers[i],
            label=class_names[i],
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
    
    # Add legend but remove title and axis labels
    plt.legend(loc='best', fontsize=20, frameon=True, facecolor='white', edgecolor='gray')
    
    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load file list
    file_list = load_file_list(args.data_path)
    print(f"Dataset size: {len(file_list)}, Label distribution: {file_list['label'].value_counts().to_dict()}")
    
    # Create dataset and data loader
    dataset = PADSDataset(args.data_path, file_list)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # No need to shuffle
        num_workers=0,
        pin_memory=True
    )
    
    # Parse tremor frequency band parameter
    tremor_freq_band = tuple(map(float, args.tremor_freq_band.split(',')))
    
    # Create model
    model = ShallowCNN1D(
        n_classes=2,  # Binary classification task: HC(0) and PD(1)
        use_feature_extractor=args.use_feature_extractor,
        tremor_freq_band=tremor_freq_band,
        fs=args.sampling_rate
    )
    
    # Load pre-trained model (if path is specified)
    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"Loading pre-trained model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_state = "pretrained"
    else:
        print("Using randomly initialized model")
        model_state = "random"
    
    # Move model to device
    model.to(device)
    
    # Extract raw features and model features
    print("Starting feature extraction...")
    raw_features, model_features, labels = extract_features(model, dataloader, device)
    
    print(f"Raw features shape: {raw_features.shape}, Model features shape: {model_features.shape}, Labels shape: {labels.shape}")
    
    # Apply t-SNE dimensionality reduction to raw features
    print("Applying t-SNE to raw features...")
    raw_tsne = apply_tsne(raw_features, labels, args.perplexity, args.n_iter, args.random_state)
    
    # Apply t-SNE dimensionality reduction to model features
    print("Applying t-SNE to model features...")
    model_tsne = apply_tsne(model_features, labels, args.perplexity, args.n_iter, args.random_state)
    
    # Visualize t-SNE results of raw features
    raw_title = ""  # Empty title
    raw_save_path = os.path.join(args.output_dir, f"raw_features_tsne_{model_state}.png")
    visualize_tsne(raw_tsne, labels, raw_title, raw_save_path)
    
    # Visualize t-SNE results of model features
    model_title = ""  # Empty title
    model_save_path = os.path.join(args.output_dir, f"model_features_tsne_{model_state}.png")
    visualize_tsne(model_tsne, labels, model_title, model_save_path)
    
    # Create comparison figure
    plt.figure(figsize=(18, 8))
    
    # Define class names and colors
    class_names = ['HC', 'PD']
    colors = ['#2ca02c', '#d62728']  # Green for healthy controls, red for Parkinson's
    markers = ['o', 's']  # Circle for healthy controls, square for Parkinson's
    
    # Raw features subplot
    plt.subplot(1, 2, 1)
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        plt.scatter(
            raw_tsne[idx, 0], 
            raw_tsne[idx, 1],
            c=colors[i],
            marker=markers[i],
            label=class_names[i],
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
    # Add legend but remove title and axis labels
    plt.legend(loc='best', fontsize=20)
    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Model features subplot
    plt.subplot(1, 2, 2)
    for i, label in enumerate(np.unique(labels)):
        idx = labels == label
        plt.scatter(
            model_tsne[idx, 0], 
            model_tsne[idx, 1],
            c=colors[i],
            marker=markers[i],
            label=class_names[i],
            alpha=0.7,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
    # Add legend but remove title and axis labels
    plt.legend(loc='best', fontsize=20)
    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"comparison_tsne_{model_state}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization results saved to: {args.output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds") 