"""
特征相关性分析模块 / Feature Correlation Analysis Module
==========================================================
提供 Spearman 相关性分析与可视化，用于分析ESM2特征维度之间的相关性。
Provides Spearman correlation analysis and visualization for analyzing
correlations between ESM2 feature dimensions.

主要功能 / Main Functions:
    - spear_correlation.plot_spearman_heatmap: 绘制Spearman相关热图
                                               Plot Spearman correlation heatmap
    - spear_correlation.feature_crossover_reg: 绘制两特征间的回归散点图（带相关系数）
                                               Plot regression scatter plot between two features

特征名格式约定 / Feature name format convention:
    - 交叉特征: "feature1*_feature2" 格式，用 '*_' 分隔两个特征名
      Cross features: "feature1*_feature2" format, split by '*_'

依赖 / Dependencies: scipy >= 1.7.3, seaborn, matplotlib
"""

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from packaging import version

# 检查scipy版本 / Check scipy version
try:
    import scipy
    if version.parse(scipy.__version__) < version.parse('1.7.3'):
        warnings.warn("Your scipy version is older than 1.7.3 and may not operate correctly.")
    from scipy import stats  # 用于Spearman相关计算 / For Spearman correlation calculation
except ImportError:
    warnings.warn("Scipy is not installed. Some features may not work as expected.")


class spear_correlation():
    """
    Spearman 相关性分析与可视化类 / Spearman correlation analysis and visualization class.

    属性 / Attributes:
        X_data (pd.DataFrame): 特征矩阵 / Feature matrix
        y_data: 标签数据（目前未使用）/ Label data (currently unused)
    """
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    @staticmethod
    def plot_spearman_heatmap(X_data, save_path_completed, draw_heatmap=True, figure_size=(10, 10)):
        """
        计算并绘制 Spearman 相关系数热图
        Compute and plot Spearman correlation coefficient heatmap.

        参数 / Parameters:
            X_data (pd.DataFrame): 特征矩阵 / Feature matrix
            save_path_completed (str): 完整保存路径（含文件名）/ Full save path (including filename)
            draw_heatmap (bool): 是否绘制热图 / Whether to plot heatmap
            figure_size (tuple): 图像尺寸 / Figure size

        返回 / Returns:
            (df_corr, df_p): Spearman相关系数DataFrame 和 p值DataFrame
                             Spearman correlation DataFrame and p-value DataFrame
        """
        # 计算Spearman相关系数和p值 / Compute Spearman correlation and p-values
        corrs, p_values = stats.spearmanr(X_data)
        # 创建上三角掩码（只显示下三角）/ Create upper triangle mask (show lower triangle only)
        mask = np.zeros_like(corrs)
        mask[np.triu_indices_from(mask)] = True
        if draw_heatmap:
            fig, ax = plt.subplots(figsize=figure_size)
            # 绘制热图（颜色范围 -1 到 1，中心为 0）/ Plot heatmap (color range -1 to 1, centered at 0)
            fig = sns.heatmap(
                corrs, vmin=-1, vmax=1, center=0, mask=mask, square=True,
                cmap='viridis',
                xticklabels=X_data.columns,
                yticklabels=X_data.columns)
            fig.set_title('Spearman Correlation Heatmap')
            fig.figure.savefig(save_path_completed, dpi=1000, bbox_inches="tight")
            plt.close(fig.figure)
        # 将相关系数和p值转换为DataFrame / Convert to DataFrames
        df_corr = pd.DataFrame(corrs, index=X_data.columns, columns=X_data.columns)
        df_p = pd.DataFrame(p_values, index=X_data.columns, columns=X_data.columns)
        return df_corr, df_p

    def feature_crossover_reg(self, feature_cross_over, save_path, figure_size=(13, 6)):
        """
        对指定特征对绘制回归散点图，并标注Spearman相关系数与p值
        Plot regression scatter plot for feature pairs with Spearman correlation annotation.

        参数 / Parameters:
            feature_cross_over (list): 特征对列表，格式为 "feat1*_feat2"
                                       List of feature pair strings in "feat1*_feat2" format
            save_path (str): 输出目录路径 / Output directory path
            figure_size (tuple): 图像尺寸 / Figure size
        """
        for i_name in feature_cross_over:
            # 解析特征对名称（用'*_'分隔）/ Parse feature pair names (split by '*_')
            feat1, feat2 = i_name.split('*_')[0], i_name.split('*_')[1]
            # 计算Spearman相关系数和p值 / Compute Spearman correlation and p-value
            corr, p_value = stats.spearmanr(self.X_data[feat1], self.X_data[feat2])
            # 绘制回归散点图 / Plot regression scatter plot
            fig, axs = plt.subplots(figsize=figure_size)
            fig = sns.regplot(x=self.X_data[feat1], y=self.X_data[feat2],
                              ax=axs, scatter_kws={'alpha': 0.3},
                              line_kws={'color': 'g'})
            axs.set_xlabel(feat1, fontsize=13)
            axs.set_ylabel(feat2, fontsize=13)
            # 标题中显示相关系数和p值 / Show correlation and p-value in title
            fig.set_title('Feature Crossover Regression Analysis\nspearman: {:.3f}, p-value: {:.4f}'.format(corr, p_value))
            fig.figure.savefig(save_path + feat1 + 'vs_' + feat2 + ' crossover_reg.pdf',
                               dpi=1000, bbox_inches="tight")
            plt.close(fig.figure)
