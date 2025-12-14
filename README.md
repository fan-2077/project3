### 鸢尾花数据分类与可视化项目

## 项目概述

本项目系统地比较了多种机器学习分类器在鸢尾花数据集上的表现，通过创新的可视化技术深入分析了分类器的决策边界和概率分布。项目包含四个渐进的任务，从简单的二维特征分类到复杂的四维特征空间分析，全面展示了机器学习分类问题的核心概念和技术。

## 数据集信息

• 数据集：Iris鸢尾花数据集

• 样本数量：150个样本（3个类别，每个类别50个样本）

• 特征数量：4个特征

  • 萼片长度（Sepal Length）

  • 萼片宽度（Sepal Width） 

  • 花瓣长度（Petal Length）

  • 花瓣宽度（Petal Width）

• 类别：Setosa, Versicolor, Virginica（3个类别）

## 任务说明

# 任务一：二维特征空间的三分类可视化

• 目标：比较5个分类器在花瓣特征（Petal Length, Petal Width）上的表现

• 分类器：逻辑回归、SVM（RBF核）、决策树、K近邻、随机森林

• 可视化：决策边界 + 概率分布图

• 输出文件：

  • task1_classifiers_comparison.png - 分类器决策边界对比

  • task1_summary_probability_maps.png - 汇总概率分布图

# 任务二：三维特征空间的二分类决策边界

• 目标：在三维特征空间中可视化Setosa vs Virginica的二分类问题

• 特征：Sepal Width, Petal Length, Petal Width

• 技术：交互式3D决策边界可视化

• 输出文件：task2_result.html - 交互式3D决策边界

# 任务三：三维概率图可视化

• 目标：展示分类器在三维空间中的概率分布和分类置信度

• 技术：多概率水平等值面可视化

• 特点：动态颜色映射反映分类置信度

• 输出文件：task3_result.html - 交互式3D概率图

# 任务四：四维特征空间的综合分析与创新

• 目标：使用全部四个特征进行三分类，结合先进的可视化技术

• 创新点：

  • 原始空间训练 + PCA可视化

  • 5折交叉验证评估

  • 特征重要性分析

• 输出文件：

  • task4_decision_boundary.html - 改进版3D决策边界

  • task4_probability_maps.html - 改进版概率图

  • task4_performance.html - 性能对比分析

## 安装要求

# Python环境要求

Python 3.7+


# 必需库安装

pip install numpy matplotlib scikit-learn plotly pandas


# 可选库（用于高级可视化）

pip install seaborn plotly-express


## 快速开始

# 运行单个任务

运行任务一：二维特征分类器比较
python task_1.py

运行任务二：三维决策边界可视化  
python task_2&3.py

运行任务三：三维概率图可视化
python task_2&3.py

运行任务四：改进的四维特征分析
python task_4.py


## 查看结果

• 静态图像（PNG格式）：直接打开查看

• 交互式可视化（HTML格式）：在浏览器中打开HTML文件

## 分类器性能比较

本项目系统比较了以下5个经典分类器：

| 分类器 | 优点 | 缺点 | 适用场景 |
|--------|--------|-------|------|
| 逻辑回归 | 计算效率高，可解释性强 | 只能学习线性边界 | 线性可分数据 |
| SVM（RBF核） | 能够处理复杂非线性边界 | 参数调优复杂，计算成本高 | 小到中型数据集 | 
| 决策树 | 直观易懂，无需特征缩放 | 容易过拟合，对噪声敏感 | 需要可解释性的场景
| K近邻 | 简单易实现，无需训练 | 预测速度慢，对不平衡数据敏感 | 小型数据集，局部模式重要
| 随机森林 | 抗过拟合，可评估特征重要性 | 黑盒模型，计算资源消耗大 | 各类数据集，需要稳定性能

## 技术特点

# 可视化技术创新

1. 多维数据降维可视化：使用PCA将高维数据投影到3D/2D空间
2. 交互式3D图形：支持旋转、缩放的多角度观察
3. 概率等值面可视化：显示分类置信度分布
4. 决策边界动态展示：实时查看不同分类器的决策区域

# 分析方法创新

1. 原始空间训练 + 降维可视化：保持模型精度的同时实现有效可视化
2. 交叉验证评估：提供更可靠的性能估计
3. 特征重要性分析：理解模型决策依据
4. 多角度性能对比：训练准确率、测试准确率、交叉验证结果

## 结果解读指南

# 决策边界图解读

• 彩色区域：表示不同类别的决策区域

• 数据点颜色：表示真实类别标签

• 数据点形状：圆形表示训练集，菱形表示测试集

• 边界平滑度：反映模型的复杂度和泛化能力

# 概率图解读

• 颜色深浅：表示分类置信度（颜色越深置信度越高）

• 等值面：连接相同概率值的点，形成概率轮廓

• 透明度：反映概率值的大小（越不透明概率越高）

• 过渡区域：模型不确定的区域，通常位于类别边界

# 性能指标解读

• 训练准确率：模型在训练集上的表现（可能过拟合）

• 测试准确率：模型在未见数据上的泛化能力

• 交叉验证均值：更稳健的性能估计

• 交叉验证标准差：模型性能的稳定性

# 扩展应用

## 本项目的方法可以扩展到其他分类问题：

# 更换数据集

from sklearn.datasets import load_wine, load_breast_cancer

#使用葡萄酒数据集
wine = load_wine()
X, y = wine.data, wine.target

#使用乳腺癌数据集
cancer = load_breast_cancer() 
X, y = cancer.data, cancer.target


# 添加新分类器

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#添加新分类器
classifiers['Gaussian Naive Bayes'] = GaussianNB()
classifiers['QDA'] = QuadraticDiscriminantAnalysis()


# 自定义特征组合

#使用不同的特征组合
feature_combinations = {
    'sepal_only': [0, 1],      # 只使用萼片特征
    'petal_only': [2, 3],      # 只使用花瓣特征  
    'all_features': [0, 1, 2, 3]  # 使用所有特征
}


常见问题解答

Q1: 为什么选择鸢尾花数据集？

A: 鸢尾花数据集是机器学习入门的经典数据集，具有适中的规模、清晰的类别分离和良好的可解释性，非常适合教学和算法比较。

Q2: 如何解释概率图中的等值面？

A: 等值面连接了具有相同预测概率的点。比如0.5的等值面表示模型对该类别有50%置信度的边界，等值面越密集表示概率变化越剧烈。

Q3: 为什么有些分类器在训练集上表现完美但测试集不佳？

A: 这通常是过拟合的迹象，表明模型过度学习了训练数据的噪声而非一般规律。决策树和K近邻容易出现这种情况。

Q4: 如何选择最适合自己数据的分类器？

A: 建议依次考虑：数据规模、特征数量、需要可解释性还是准确率、计算资源限制。本项目的比较方法可以帮助您做出选择。

贡献指南

欢迎贡献代码、报告问题或提出改进建议：

1. Fork本项目
2. 创建特性分支 (git checkout -b feature/AmazingFeature)
3. 提交更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 开启Pull Request


最后更新：2024年1月 

注意：本项目主要用于教育和研究目的，实际应用时请根据具体数据特点进行调整和验证。
