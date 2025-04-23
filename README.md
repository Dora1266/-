# 人员异常行为检测系统

基于深度学习的视频异常行为检测系统，结合3D CNN和双向LSTM进行时空特征提取和异常模式识别。

## 项目简介

本项目实现了一个端到端的人员异常行为检测系统，适用于安防监控、工业安全、医疗护理和零售分析等多种场景。系统采用无监督学习方法，通过仅使用正常行为数据进行训练，学习正常行为模式，从而检测出偏离正常模式的异常行为。

## 系统架构

系统主要由以下几个部分组成：

- **数据预处理**：视频帧提取、调整大小和归一化
- **特征提取**：使用预训练的3D CNN（ResNet-18 3D）提取时空特征
- **时序建模**：利用双向LSTM捕获长时间依赖关系
- **注意力机制**：自动关注关键时刻的特征
- **异常评分**：输出每个视频片段的异常概率分数

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- OpenCV 4.5+
- NumPy
- scikit-learn
- matplotlib
- tqdm

## 安装指南

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
```

2. 创建虚拟环境（推荐）：

```bash
conda create -n anomaly python=3.8
conda activate anomaly
```

3. 安装依赖：

```bash
pip install torch torchvision opencv-python numpy scikit-learn matplotlib tqdm
```

## 数据集结构

系统要求数据集按以下结构组织：

```
dataset/
├── train/
│   └── normal/           # 训练集仅包含正常行为视频
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
└── test/
    ├── normal/           # 测试集中的正常行为视频
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    └── abnormal/         # 测试集中的异常行为视频
        ├── video1.mp4
        ├── video2.mp4
        └── ...
```

## 配置说明

在代码中的`Config`类中可以调整以下配置参数：

- **CLIP_LENGTH**：每个视频片段的帧数
- **FRAME_SIZE**：调整后的帧大小
- **BATCH_SIZE**：批处理大小
- **HIDDEN_SIZE**：LSTM隐藏层大小
- **NUM_LAYERS**：LSTM层数
- **DROPOUT**：Dropout比例
- **LEARNING_RATE**：学习率
- **WEIGHT_DECAY**：权重衰减
- **EPOCHS**：训练轮数
- **DEVICE**：计算设备（GPU或CPU）
- **DATA_ROOT**：数据集根目录路径
- **MODEL_SAVE_PATH**：模型保存路径

## 使用指南

1. 修改`Config`类中的`DATA_ROOT`为您的数据集路径。

2. 运行训练程序：

```bash
python anomaly_detection.py
```

3. 训练完成后，最佳模型将保存在`MODEL_SAVE_PATH`指定的目录中。

4. 对新视频进行异常检测：

```python
# 加载训练好的模型
model = AnomalyDetector().to(device)
checkpoint = torch.load('saved_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
threshold = checkpoint['threshold']

# 对视频进行异常检测
scores, time_axis = visualize_anomalies(model, 'path/to/video.mp4', threshold, device)
```

## 评估指标

系统使用以下指标评估异常检测性能：

- **AUC（曲线下面积）**：评估模型区分正常与异常行为的能力
- **F1分数**：在最佳阈值下的精确率和召回率的调和平均
- **最佳阈值**：自动选择最优化F1分数的异常判断阈值

## 可视化

系统提供了异常分数随时间变化的可视化功能：

- 生成异常分数随时间变化的曲线图
- 标记超过阈值的异常区域
- 将可视化结果保存为图像文件

## 模型详解

1. **3D CNN特征提取器**：
   - 使用预训练的R3D-18模型
   - 去除最后的分类层，仅保留特征提取功能

2. **双向LSTM**：
   - 处理从3D CNN提取的特征序列
   - 捕获长时间依赖关系

3. **注意力机制**：
   - 关注重要时刻和特征
   - 生成上下文向量用于异常检测

4. **分类器**：
   - 多层感知机结构
   - 输出异常分数（0-1之间）

## 扩展方向

1. **模型轻量化**：
   - 知识蒸馏
   - 模型剪枝
   - 量化

2. **多模态融合**：
   - 结合音频特征
   - 整合姿态估计

3. **实时处理优化**：
   - 流式处理
   - 模型加速

## 致谢

感谢以下开源项目和研究工作的贡献：

- PyTorch
- OpenCV
- torchvision

## 许可证

[MIT License](LICENSE)
