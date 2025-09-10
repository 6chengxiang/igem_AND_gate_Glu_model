#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iGEM 建模工具包 - 主分析脚本 (iGEM Modeling Toolkit - Main Analysis Script)

功能:
- 演示如何使用 `models` 目录中的模块化模型。
- 运行并可视化每个核心模型的分析结果：
  1. AND门逻辑 (AND Gate Logic)
  2. 谷氨酸代谢 (Glutamate Metabolism)
  3. 整合治疗模型 (Integrated Therapy Model)
- 生成分析图表并保存到 `results/` 目录。

如何运行:
- 确保已安装所有依赖 (`pip install -r requirements.txt`)。
- 直接从命令行运行此脚本: `python run_analysis.py`
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 导入新的模块化模型

from models.and_gate import SimpleANDGate
from models.glu_metabolism import GluMetabolismModel
from models.integrated_model import IntegratedTherapyModel

# 设置全局绘图样式和输出目录
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DIR = Path("results")
FIGURE_DIR.mkdir(exist_ok=True)

def run_and_gate_analysis():
    """
    分析并可视化AND门的响应面。
    """
    print("--- 1. 正在运行 AND门 响应面分析 ---")
    start_time = time.time()
    
    gate = SimpleANDGate()
    
    # 创建输入网格
    O2_levels = np.linspace(0, 25, 50)
    Temp_levels = np.linspace(35, 45, 50)
    O2_grid, Temp_grid = np.meshgrid(O2_levels, Temp_levels)
    
    # 计算T7活性
    t7_activity = gate.get_t7_activity(O2_grid, Temp_grid)
    
    # 绘图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(O2_grid, Temp_grid, t7_activity, cmap='viridis', edgecolor='none')
    
    ax.set_title('AND Gate Response Surface (T7 Activity)', fontsize=16)
    ax.set_xlabel('Oxygen Level (%)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_zlabel('T7 Activity (AU)', fontsize=12)
    fig.colorbar(surf, shrink=0.5, aspect=10, label='T7 Activity')
    
    # 保存图像
    output_path = FIGURE_DIR / "and_gate_response.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    duration = time.time() - start_time
    print(f"✅ AND门分析完成 ({duration:.2f}s)。图像已保存至: {output_path}")

def run_glutamate_analysis():
    """
    模拟并可视化谷氨酸的生产和分泌。
    """
    print("\n--- 2. 正在运行谷氨酸生产动力学模拟 ---")
    start_time = time.time()
    
    # 假设AND门处于"ON"状态，T7活性较高
    t7_activity_on = 1000.0 
    model = GluMetabolismModel()
    
    # 运行模拟
    t, solution = model.simulate(t7_activity_on, t_end=48, dt=0.1)
    
    # 提取结果
    glu_intra = solution[:, 0]
    glu_extra = solution[:, 1]
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(t, glu_intra, label='Intracellular Glutamate', color='blue')
    plt.plot(t, glu_extra, label='Extracellular Glutamate', color='red', linestyle='--')
    
    plt.title('Glutamate Production and Secretion Dynamics', fontsize=16)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Glutamate Concentration (mM)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    output_path = FIGURE_DIR / "glutamate_production.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    duration = time.time() - start_time
    print(f"✅ 谷氨酸分析完成 ({duration:.2f}s)。图像已保存至: {output_path}")

def run_integrated_therapy_analysis():
    """
    模拟并可视化整合治疗模型的效果。
    """
    print("\n--- 3. 正在运行整合治疗效果模拟 ---")
    start_time = time.time()
    
    model = IntegratedTherapyModel()
    
    # --- 条件1: 治疗开启 (低氧, 高温) ---
    env_therapy = {'O2_percent': 1.0, 'Temp_C': 42.0}
    t_therapy, sol_therapy = model.simulate(env_therapy, t_end=100, with_neurotox=False)[:2]
    
    # --- 条件2: 对照组 (正常氧, 正常体温) ---
    env_control = {'O2_percent': 5.0, 'Temp_C': 37.0}
    t_control, sol_control = model.simulate(env_control, t_end=100, with_neurotox=False)[:2]
    
    # --- 绘图比较 ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle('Integrated Therapy Simulation: Therapy vs. Control', fontsize=18)
    
    # 图1: 肿瘤细胞数
    ax1.plot(t_therapy, sol_therapy[:, 0], label='Therapy ON', color='red')
    ax1.plot(t_control, sol_control[:, 0], label='Control', color='blue', linestyle='--')
    ax1.set_title('Tumor Cell Response')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Tumor Cell Count')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    # 图2: 工程细胞数
    ax2.plot(t_therapy, sol_therapy[:, 4], label='Therapy ON', color='green')
    ax2.plot(t_control, sol_control[:, 4], label='Control', color='orange', linestyle='--')
    ax2.set_title('Engineered Cell Population')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Engineered Cell Count')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    # 图3: 细胞外谷氨酸浓度
    ax3.plot(t_therapy, sol_therapy[:, 3], label='Therapy ON', color='purple')
    ax3.plot(t_control, sol_control[:, 3], label='Control', color='gray', linestyle='--')
    ax3.set_title('Extracellular Glutamate')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Glutamate (mM)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图像
    output_path = FIGURE_DIR / "integrated_therapy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    duration = time.time() - start_time
    print(f"✅ 整合治疗分析完成 ({duration:.2f}s)。图像已保存至: {output_path}")

def main():
    """
    运行所有分析。
    """
    print("="*70)
    print("====== iGEM 建模工具包 - 开始分析 ======")
    print("="*70)
    
    run_and_gate_analysis()
    run_glutamate_analysis()
    run_integrated_therapy_analysis()
    
    print("\n" + "="*70)
    print("====== 所有分析已完成 ======")
    print(f"所有结果图表已保存到 '{FIGURE_DIR.resolve()}' 目录。")
    print("="*70)

if __name__ == '__main__':
    main()