import torch
import model

# 实例化模型（启用特征提取器，即 use_feature_extractor=True）
model_full = model.ShallowCNN1D(n_channels=132, n_classes=2, use_feature_extractor=True, tremor_freq_band=(3, 7), fs=100)

# 计算总参数量（即所有参数的元素个数之和）
total_params = sum(p.numel() for p in model_full.parameters())

print("模型（ShallowCNN1D，含特征提取器）的总参数量为：", total_params, "个参数。") 