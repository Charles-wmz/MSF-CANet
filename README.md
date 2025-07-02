# 基于BOSS的时间序列特征提取与CNN分类

本项目实现了一个时间序列分类系统，该系统使用Bag of SFA Symbols (BOSS)进行特征提取，并结合CNN进行分类。

## 项目背景

Bag of SFA Symbols (BOSS)是一种强大的时间序列特征提取方法，它将时间序列转换为符号序列，然后使用词袋模型统计这些符号的频率分布。BOSS特征能够捕捉时间序列的局部结构，并对噪声有较强的鲁棒性。本项目将BOSS特征提取作为CNN的前置处理步骤，以提高分类性能。

## 主要组件

### 1. SFA (Symbolic Fourier Approximation)

SFA是BOSS的核心组件，它通过以下步骤将时间序列转换为符号表示：
- 应用离散傅里叶变换(DFT)进行低通滤波
- 使用Multiple Coefficient Binning (MCB)方法进行离散化
- 生成符号序列表示时间序列

### 2. BOSS (Bag of SFA Symbols)

BOSS算法工作流程：
- 使用滑动窗口从时间序列中提取子序列
- 将每个子序列通过SFA转换为符号表示
- 统计符号序列的频率，创建词袋表示
- 生成特征向量用于分类

### 3. BOSS特征提取器

本项目实现了一个BOSS特征提取器，具有以下特点：
- 支持多尺度窗口大小(20, 40, 80)，捕捉不同时间尺度的模式
- 为每个通道创建单独的BOSS变换器
- 将所有特征连接成一个特征向量
- 对特征进行L2归一化

### 4. CNN分类器

实现了两种使用BOSS特征的分类器：
- BOSSCNNClassifier: 使用1D卷积网络处理BOSS特征
- BOSSMLPClassifier: 使用多层感知机处理BOSS特征

## 使用方法

通过命令行参数可以控制BOSS特征提取和CNN分类的各个方面：

```
python main.py --model_type boss_cnn --use_boss \
    --boss_window_sizes 20,40,80 --boss_n_bins 3 --boss_word_length 8
```

参数说明：
- `--model_type`: 选择模型类型(boss_cnn或boss_mlp)
- `--use_boss`: 启用BOSS特征提取
- `--boss_window_sizes`: 滑动窗口大小，多个值用逗号分隔
- `--boss_n_bins`: 符号化的箱数(默认3)
- `--boss_word_length`: SFA单词长度(默认8)

## 项目结构

```
|-- boss_transform.py    # BOSS特征提取实现
|-- data_loader.py       # 数据加载器，支持BOSS特征
|-- model.py             # 模型定义，包括BOSS CNN和MLP分类器
|-- main.py              # 主程序，训练和评估模型
|-- train.py             # 训练函数
|-- evaluate.py          # 评估函数
```

## 优势

1. **特征提取与分类分离**：BOSS特征提取作为独立步骤，可以更好地捕捉时间序列结构。
2. **多尺度特征**：使用多个窗口大小提取特征，捕捉不同时间尺度的模式。
3. **噪声鲁棒性**：SFA的傅里叶变换和离散化步骤能够有效过滤噪声。
4. **通道独立处理**：为每个通道单独提取特征，保留通道特异性信息。
5. **灵活的模型选择**：可以选择CNN或MLP分类器处理BOSS特征。

# 帕金森病诊断模型架构可视化工具

本项目包含三种不同的模型架构可视化工具，用于生成帕金森病诊断模型的结构图，适用于期刊论文和学术报告。

## 依赖安装

```bash
pip install matplotlib numpy torch graphviz tensorboard
```

## 可视化工具说明

### 1. Matplotlib绘制静态图 (draw_model_architecture.py)

使用Matplotlib库绘制模型架构图，生成PNG和PDF格式的静态图像。

**特点：**
- 完全自定义布局和样式
- 适合精确控制图形位置和比例
- 生成高质量栅格图(PNG)和矢量图(PDF)

**使用方法：**
```bash
python draw_model_architecture.py
```

### 2. Graphviz绘制期刊图 (draw_model_architecture_journal.py)

使用Graphviz库绘制符合期刊要求的模型架构流程图，自动布局，生成专业的矢量图。

**特点：**
- 自动计算布局，简化复杂结构的绘制
- 符合期刊排版要求的清晰结构图
- 生成高质量矢量图(PDF)和栅格图(PNG)
- 更专业的节点和边的样式

**使用方法：**
```bash
python draw_model_architecture_journal.py
```

### 3. TensorBoard可视化 (draw_model_structure_tensorboard.py)

使用PyTorch和TensorBoard的工具直接从模型定义生成交互式结构图，可动态探索。

**特点：**
- 直接从PyTorch模型定义生成
- 准确反映实际模型结构和数据流
- 可交互探索和缩放
- 适合调试和详细了解模型内部结构

**使用方法：**
```bash
python draw_model_structure_tensorboard.py
# 然后使用TensorBoard查看生成的图
tensorboard --logdir=./tensorboard_logs
# 在浏览器中访问: http://localhost:6006/
```

## 模型结构说明

该模型是专为帕金森病诊断设计的深度学习架构，主要包含以下组件：

1. **多尺度分层特征提取器 (MSHFeatureExtractor)**
   - 多尺度卷积模块：使用不同核大小(3,7,15)的卷积层捕获不同时间尺度的特征
   - 频域感知模块：特别关注帕金森病震颤频带(3-7Hz)
   - 通道注意力模块：学习不同传感器通道的重要性

2. **基础CNN结构**
   - 三层卷积层：依次提取64、128、256通道的特征
   - 池化层和标准化层：降维和标准化
   - 全连接层：特征融合和分类

3. **创新点**
   - 频域感知：针对帕金森病震颤特征的频域增强
   - 多尺度特征提取：同时捕获不同时间尺度的运动模式
   - 通道注意力：关注最相关的传感器信号

## 示例图像

生成的图像将保存在项目根目录下，包括：
- `model_architecture.png/pdf`：Matplotlib静态图
- `model_architecture_journal.pdf`：Graphviz期刊图
- `tensorboard_logs/`：TensorBoard交互式可视化数据 