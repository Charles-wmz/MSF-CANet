import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from glob import glob
from matplotlib.ticker import MaxNLocator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='比较多个交叉验证结果')
    
    parser.add_argument('--results_dirs', nargs='+', required=True, 
                        help='要比较的结果目录列表，可以指定多个目录')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='比较结果保存目录')
    parser.add_argument('--labels', nargs='+', 
                        help='各结果目录的标签，如果不提供，将使用目录名')
    parser.add_argument('--title', type=str, default='交叉验证结果比较',
                        help='图表标题')
    parser.add_argument('--figsize', type=str, default='12,8',
                        help='图表尺寸，格式为"宽,高"')
    parser.add_argument('--dpi', type=int, default=100,
                        help='图表DPI')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='seaborn样式')
    parser.add_argument('--palette', type=str, default='Set1',
                        help='颜色调色板')
    
    return parser.parse_args()

def load_results(results_dir):
    """加载单个结果目录中的数据"""
    # 加载参数
    args_path = os.path.join(results_dir, 'args.json')
    if not os.path.exists(args_path):
        print(f"警告: {results_dir} 中找不到args.json文件")
        config = {}
    else:
        with open(args_path, 'r') as f:
            config = json.load(f)
    
    # 加载所有折的结果
    results_path = os.path.join(results_dir, 'all_fold_results.csv')
    if not os.path.exists(results_path):
        print(f"警告: {results_dir} 中找不到all_fold_results.csv文件")
        return None, config
    
    results_df = pd.read_csv(results_path)
    
    # 提取平均值和标准差
    avg_results_path = os.path.join(results_dir, 'avg_results.json')
    if os.path.exists(avg_results_path):
        with open(avg_results_path, 'r') as f:
            avg_results = json.load(f)
    else:
        # 如果没有平均结果文件，手动计算
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        avg_results = {}
        for col in numeric_cols:
            if col != 'fold':
                mean = results_df[col].mean()
                std = results_df[col].std()
                avg_results[col] = f"{mean:.4f} ± {std:.4f}"
    
    return results_df, config, avg_results

def get_result_label(results_dir, label=None, config=None):
    """为结果生成标签"""
    if label is not None:
        return label
    
    # 尝试从目录名生成标签
    dir_name = os.path.basename(results_dir.rstrip('/'))
    
    # 如果配置中有相关信息，添加到标签中
    if config:
        parts = []
        
        # 添加评估模式和值
        if 'eval_mode' in config:
            if config['eval_mode'] != 'all':
                parts.append(f"{config['eval_mode']}")
                if 'value' in config and config['value'] is not None:
                    parts.append(f"{config['value']}")
        
        # 添加特征提取器信息
        if 'use_feature_extractor' in config:
            fe_info = "with_fe" if config['use_feature_extractor'] else "without_fe"
            parts.append(fe_info)
        
        if parts:
            return " ".join(parts)
    
    return dir_name

def compare_metrics(results_list, labels, output_dir, title="交叉验证结果比较", figsize=(12, 8), dpi=100):
    """比较多个结果的性能指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 寻找共有的指标
    common_metrics = set()
    first = True
    
    for result_df, _, _ in results_list:
        if result_df is not None:
            metrics = set(result_df.columns) - {'fold'}
            if first:
                common_metrics = metrics
                first = False
            else:
                common_metrics = common_metrics.intersection(metrics)
    
    if not common_metrics:
        print("错误: 没有找到共有的指标")
        return
    
    # 常见的性能指标列表，用于排序
    standard_metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc']
    
    # 按标准顺序排序指标
    metrics = [m for m in standard_metrics_order if m in common_metrics]
    # 添加任何其他未在标准列表中的指标
    metrics.extend([m for m in common_metrics if m not in standard_metrics_order])
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    # 1. 比较平均性能的条形图
    plt.figure(figsize=figsize, dpi=dpi)
    
    # 准备数据
    avg_data = []
    
    for i, (_, _, avg_results) in enumerate(results_list):
        if avg_results:
            for metric in metrics:
                if metric in avg_results:
                    # 解析平均值和标准差
                    value_str = avg_results[metric]
                    if "±" in value_str:
                        mean_str, std_str = value_str.split("±")
                        mean = float(mean_str.strip())
                        std = float(std_str.strip())
                    else:
                        mean = float(value_str)
                        std = 0.0
                    
                    avg_data.append({
                        'Metric': metric.capitalize(),
                        'Mean': mean,
                        'Std': std,
                        'Method': labels[i]
                    })
    
    if not avg_data:
        print("警告: 没有平均数据可以绘制")
    else:
        avg_df = pd.DataFrame(avg_data)
        
        # 绘制条形图
        ax = plt.figure(figsize=figsize, dpi=dpi)
        ax = sns.barplot(x='Metric', y='Mean', hue='Method', data=avg_df)
        
        # 添加误差条
        for i, bar in enumerate(ax.patches):
            if i < len(avg_data):
                std = avg_data[i]['Std']
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(x, y, yerr=std, fmt='none', color='black', capsize=3)
        
        plt.title(f'{title} - 平均性能比较')
        plt.ylim(0, 1.05)
        plt.ylabel('性能指标值')
        plt.xlabel('')
        plt.xticks(rotation=45)
        plt.legend(title='')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, 'average_metrics_comparison.png'))
        plt.close()
    
    # 2. 为每个指标生成箱线图比较
    for metric in metrics:
        # 准备数据
        data = []
        for i, (result_df, _, _) in enumerate(results_list):
            if result_df is not None and metric in result_df.columns:
                for _, row in result_df.iterrows():
                    data.append({
                        'Method': labels[i],
                        'Value': row[metric],
                        'Fold': int(row['fold'])
                    })
        
        if not data:
            print(f"警告: 指标 {metric} 没有数据可以绘制")
            continue
            
        boxplot_df = pd.DataFrame(data)
        
        # 绘制箱线图
        plt.figure(figsize=figsize, dpi=dpi)
        ax = sns.boxplot(x='Method', y='Value', data=boxplot_df)
        
        # 添加散点图展示各折的值
        sns.stripplot(x='Method', y='Value', data=boxplot_df, 
                     size=7, jitter=True, alpha=0.6, color='black')
        
        plt.title(f'{title} - {metric.capitalize()} 分布比较')
        plt.ylim(0, 1.05)
        plt.ylabel(f'{metric.capitalize()} 值')
        plt.xlabel('')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'{metric}_distribution_comparison.png'))
        plt.close()
        
        # 为每个指标绘制折线图，比较不同方法在各折上的表现
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 查找最大折数
        max_fold = boxplot_df['Fold'].max()
        
        for method in labels:
            method_data = boxplot_df[boxplot_df['Method'] == method]
            if not method_data.empty:
                # 确保数据按折排序
                method_data = method_data.sort_values('Fold')
                plt.plot(method_data['Fold'], method_data['Value'], 'o-', label=method, linewidth=2, markersize=8)
        
        plt.title(f'{title} - {metric.capitalize()} 各折表现比较')
        plt.xlabel('交叉验证折')
        plt.ylabel(f'{metric.capitalize()} 值')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴刻度为整数
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'{metric}_folds_comparison.png'))
        plt.close()
    
    # 3. 生成汇总比较表
    summary_data = []
    
    for i, (_, config, avg_results) in enumerate(results_list):
        method_info = {'Method': labels[i]}
        
        # 添加配置信息
        if config:
            for key in ['eval_mode', 'value', 'use_feature_extractor', 'n_splits', 'stratified']:
                if key in config:
                    method_info[key] = config[key]
        
        # 添加性能指标
        if avg_results:
            for metric in metrics:
                if metric in avg_results:
                    method_info[metric] = avg_results[metric]
        
        summary_data.append(method_info)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        summary_path = os.path.join(output_dir, 'comparison_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"比较汇总表已保存到: {summary_path}")
        
        # 保存为更美观的HTML
        html_path = os.path.join(output_dir, 'comparison_summary.html')
        summary_df.to_html(html_path, index=False)
        print(f"比较汇总HTML已保存到: {html_path}")
    
    print(f"所有比较图表已保存到: {output_dir}")

def main():
    args = parse_args()
    
    # 解析图表尺寸
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # 设置seaborn样式
    sns.set_style(args.style)
    sns.set_palette(args.palette)
    
    # 验证结果目录
    valid_dirs = []
    for directory in args.results_dirs:
        if os.path.isdir(directory):
            valid_dirs.append(directory)
        else:
            print(f"警告: {directory} 不是有效目录，将被跳过")
    
    if not valid_dirs:
        print("错误: 没有找到有效的结果目录")
        return
    
    # 设置标签
    if args.labels and len(args.labels) == len(valid_dirs):
        labels = args.labels
    else:
        labels = []
        for directory in valid_dirs:
            # 尝试自动生成标签
            dir_name = os.path.basename(directory.rstrip('/'))
            labels.append(dir_name)
    
    # 加载所有结果
    results_list = []
    for i, directory in enumerate(valid_dirs):
        result_df, config, avg_results = load_results(directory)
        if result_df is not None:
            # 更新标签
            labels[i] = get_result_label(directory, labels[i], config)
            results_list.append((result_df, config, avg_results))
        else:
            print(f"跳过 {directory} - 无法加载结果")
    
    if not results_list:
        print("错误: 没有找到有效的结果数据")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行比较
    compare_metrics(
        results_list=results_list,
        labels=labels[:len(results_list)],
        output_dir=args.output_dir,
        title=args.title,
        figsize=figsize,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main() 