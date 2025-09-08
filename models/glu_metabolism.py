#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iGEM工程菌治疗性谷氨酸代谢模型 (Therapeutic Glutamate Metabolism Model)

基于HTML文档的完整生物学机理实现：
- T7聚合酶驱动的ICD/GDH过表达系统
- NADPH供应与谷氨酸合成的偶联代谢
- 野生型vs工程菌的差异化代谢特性
- 热激激活的两阶段外排控制系统
- 符合体内治疗应用的动力学参数

核心治疗目标：
- 野生型：胞内Glu稳态10-30mM，胞外接近0
- 工程菌热激后：胞内Glu堆积≥50mM，胞外峰值≥30mM，最终回归~20mM

生物学机理：
- ICD催化：ICIT → AKG + CO2 + NADPH
- GDH催化：AKG + NH4+ + NADPH → Glu + NADP+
- 动态外排：累积阶段抑制 + 恢复阶段强化
"""

import numpy as np
from scipy.integrate import odeint

class TherapeuticGluModel:
    """
    iGEM工程菌治疗性谷氨酸代谢的完整ODE模型
    
    基于HTML文档的生物学机理，包含：
    1. T7聚合酶驱动的ICD/GDH过表达调控
    2. 完整的谷氨酸代谢通路：Glc → ICIT → AKG → Glu
    3. NADPH供应系统：PPP + ICD-NADPH
    4. 野生型vs工程菌的差异化特性
    5. 动态两阶段外排控制
    
    状态变量：
    - [Glc_ext]: 胞外葡萄糖浓度 (mM)
    - [NH4_ext]: 胞外铵浓度 (mM)
    - [ICIT]: 胞内异柠檬酸浓度 (mM)
    - [AKG]: 胞内α-酮戊二酸浓度 (mM)
    - [Glu_in]: 胞内谷氨酸浓度 (mM)
    - [NADPH]: 胞内NADPH浓度 (mM)
    - [X]: 生物量浓度 (gDW/L)
    - [Glu_ext]: 胞外谷氨酸浓度 (mM)
    - [fold_ICD]: ICD过表达倍数（动态）
    - [fold_GDH]: GDH过表达倍数（动态）
    """
    
    def __init__(self, strain_type='engineered', **params):
        """
        初始化治疗性谷氨酸代谢模型
        
        Parameters:
        -----------
        strain_type : str
            菌株类型: 'wildtype' 或 'engineered'
        **params : dict
            模型参数，基于HTML文档的参数表
        """
        self.strain_type = strain_type
        
        # === 基础代谢参数（基于HTML文档参数表）===
        # 葡萄糖摄取
        self.V_max_glc = params.get('V_max_glc', 10.0)  # mmol·gDW⁻¹·h⁻¹
        self.K_m_glc = params.get('K_m_glc', 1.0)       # mM
        
        # TCA分流
        self.f_TCA = params.get('f_TCA', 0.6)           # 进入ICIT池的碳分流比
        
        # ICD酶动力学
        self.V_max_base_ICD = params.get('V_max_base_ICD', 20.0)  # mmol·gDW⁻¹·h⁻¹
        self.K_m_ICD = params.get('K_m_ICD', 0.029)              # mM
        
        # GDH酶动力学（多底物米氏）
        self.V_max_base_GDH = params.get('V_max_base_GDH', 25.0)  # mmol·gDW⁻¹·h⁻¹
        self.K_m_AKG = params.get('K_m_AKG', 0.64)               # mM
        self.K_m_NH4 = params.get('K_m_NH4', 1.1)                # mM
        self.K_m_NADPH = params.get('K_m_NADPH', 0.04)           # mM
        
        # NADPH供应
        self.k_PPP = params.get('k_PPP', 0.3)                    # mol NADPH / mol Glc
        self.y_ICD_NADPH = params.get('y_ICD_NADPH', 1.0)        # mol NADPH / mol AKG
        
        # 外排与维持
        self.k_sec_base = params.get('k_sec_base', 0.5)          # h⁻¹
        self.Y_X_Glc = params.get('Y_X_Glc', 0.20)               # gDW·mmol⁻¹
        
        # 非生长耗项
        self.k_drain_AKG = params.get('k_drain_AKG', 0.0)        # h⁻¹
        self.k_drain_Glu = params.get('k_drain_Glu', 0.0)        # h⁻¹
        
        # NADPH平衡调节
        self.lambda_NADPH = params.get('lambda_NADPH', 1.0)       # h⁻¹
        self.NADPH_set = params.get('NADPH_set', 0.10)            # mM
        
        # === T7驱动的酶表达动力学参数 ===
        self.K_T7 = params.get('K_T7', 800.0)                    # T7半最大激活常数 (AU)
        self.n_hill = params.get('n_hill', 3.0)                  # Hill系数
        self.tau_enzyme = params.get('tau_enzyme', 0.05)          # 酶表达时间常数 (h)
        
        # === 菌株特异性参数 ===
        if strain_type == 'wildtype':
            # 野生型：强制稳态控制，无过表达
            self.fold_ICD_max = 1.0
            self.fold_GDH_max = 1.0
            self.homeostasis_strength = params.get('homeostasis_strength', 5.0)
            self.Glu_target = params.get('Glu_target', 20.0)  # mM
        else:
            # 工程菌：强力过表达，但保持基础稳态调节
            self.fold_ICD_max = params.get('fold_ICD_max', 1000.0)
            self.fold_GDH_max = params.get('fold_GDH_max', 1500.0)
            # 工程菌也有轻微的稳态调节，确保正常情况下维持20mM
            self.homeostasis_strength = params.get('homeostasis_strength_eng', 1.0)  # 比野生型弱
            self.Glu_target = params.get('Glu_target', 20.0)  # mM
            
        # === 动态外排控制参数 ===
        if strain_type == 'engineered':
            # 累积阶段参数
            self.accum_threshold = params.get('accum_threshold', 55.0)     # mM
            self.export_accum_suppression = params.get('export_accum_suppression', 0.05)
            
            # 恢复阶段参数  
            self.recovery_threshold = params.get('recovery_threshold', 20.0)  # mM
            self.postshock_export_boost = params.get('postshock_export_boost', 10.0)
        
        # === 维持成本与稀释 ===
        self.k_maintenance = params.get('k_maintenance', 0.1)     # h⁻¹
        self.mu_max = params.get('mu_max', 0.5)                   # h⁻¹
        
    def calculate_growth_rate(self, Glc_ext):
        """计算基于葡萄糖的生长速率"""
        return self.mu_max * Glc_ext / (self.K_m_glc + Glc_ext)
    
    def calculate_glucose_uptake(self, Glc_ext, X):
        """计算葡萄糖摄取速率"""
        q_glc = self.V_max_glc * Glc_ext / (self.K_m_glc + Glc_ext)
        return q_glc
    
    def calculate_enzyme_expression(self, t7_activity, current_fold):
        """计算T7驱动的酶表达动力学"""
        if self.strain_type == 'wildtype':
            return 0.0  # 野生型无过表达
            
        # T7信号转导（Hill函数）
        t7_signal = (t7_activity**self.n_hill) / (self.K_T7**self.n_hill + t7_activity**self.n_hill)
        
        # 目标表达水平
        if 'ICD' in str(current_fold):  # 判断是ICD还是GDH
            target = 1.0 + (self.fold_ICD_max - 1.0) * t7_signal
        else:
            target = 1.0 + (self.fold_GDH_max - 1.0) * t7_signal
            
        # 一阶动力学逼近目标
        return (target - current_fold) / self.tau_enzyme
    
    def calculate_export_rate(self, Glu_in, fold_GDH, t_current):
        """计算动态外排速率"""
        base_export = self.k_sec_base
        
        if self.strain_type == 'wildtype':
            # 野生型：极弱外排
            return base_export * 0.1
        
        # 工程菌：动态两阶段控制
        if fold_GDH > 10.0 and Glu_in < self.accum_threshold:
            # 累积阶段：强力抑制外排
            k_sec = base_export * self.export_accum_suppression
        elif Glu_in > self.recovery_threshold:
            # 恢复阶段：强化外排
            k_sec = base_export * self.postshock_export_boost
        else:
            # 基线外排
            k_sec = base_export
            
        return k_sec
    
    def apply_homeostasis(self, Glu_in, fold_GDH):
        """稳态控制机制：工程菌和野生型都有，但强度不同"""
        # 基础稳态调节
        deviation = Glu_in - self.Glu_target
        base_correction = -self.homeostasis_strength * deviation
        
        # 工程菌额外逻辑：热激期间降低稳态调节，允许堆积
        if self.strain_type == 'engineered' and fold_GDH > 10.0:
            # 热激期间大幅降低稳态调节强度
            heat_shock_suppression = 0.1  # 降低到10%
            return base_correction * heat_shock_suppression
        else:
            return base_correction

    
    def dydt(self, y, t, t7_activity):
        """
        定义完整的谷氨酸代谢ODE系统
        
        状态变量y = [Glc_ext, NH4_ext, ICIT, AKG, Glu_in, NADPH, X, Glu_ext, fold_ICD, fold_GDH]
        
        Parameters:
        -----------
        y : array
            状态变量向量
        t : float
            时间 (h)
        t7_activity : float or function
            T7聚合酶活性 (AU)
        
        Returns:
        --------
        dydt : array
            状态变量的时间导数
        """
        # 解包状态变量
        Glc_ext, NH4_ext, ICIT, AKG, Glu_in, NADPH, X, Glu_ext, fold_ICD, fold_GDH = y
        
        # 处理T7活性（可以是函数或常数）
        if callable(t7_activity):
            T7_current = t7_activity(t)
        else:
            T7_current = t7_activity
        
        # 1. 生长速率
        mu = self.calculate_growth_rate(Glc_ext)
        
        # 2. 葡萄糖摄取速率
        q_glc = self.calculate_glucose_uptake(Glc_ext, X)
        
        # 3. 进入TCA的碳流
        v_TCAin = self.f_TCA * q_glc
        
        # 4. ICD反应速率 (ICIT → AKG + NADPH)
        V_max_ICD = self.V_max_base_ICD * fold_ICD
        v_ICD = V_max_ICD * ICIT / (self.K_m_ICD + ICIT)
        
        # 5. GDH反应速率 (AKG + NH4+ + NADPH → Glu) - 多底物米氏
        V_max_GDH = self.V_max_base_GDH * fold_GDH
        f_AKG = AKG / (self.K_m_AKG + AKG)
        f_NH4 = NH4_ext / (self.K_m_NH4 + NH4_ext)
        f_NADPH = NADPH / (self.K_m_NADPH + NADPH)
        v_GDH = V_max_GDH * f_AKG * f_NH4 * f_NADPH
        
        # 6. NADPH产生速率
        v_NADPH_production = (self.k_PPP * q_glc + self.y_ICD_NADPH * v_ICD) * X
        
        # 7. NADPH平衡调节
        v_relax = self.lambda_NADPH * (self.NADPH_set - NADPH)
        
        # 8. 外排速率
        k_sec = self.calculate_export_rate(Glu_in, fold_GDH, t)
        v_sec = k_sec * Glu_in
        
        # 9. 稳态控制（传入fold_GDH用于工程菌热激判断）
        homeostasis_term = self.apply_homeostasis(Glu_in, fold_GDH)
        
        # 10. 酶表达动力学
        dfold_ICD_dt = self.calculate_enzyme_expression(T7_current, fold_ICD)
        dfold_GDH_dt = self.calculate_enzyme_expression(T7_current, fold_GDH)
        
        # === ODE系统（基于HTML文档）===
        
        # 胞外葡萄糖
        dGlc_ext_dt = -q_glc * X
        
        # 胞外铵
        dNH4_ext_dt = -v_GDH * X
        
        # 胞内异柠檬酸
        dICIT_dt = (v_TCAin - v_ICD) * X - mu * ICIT
        
        # 胞内α-酮戊二酸
        dAKG_dt = (v_ICD - v_GDH) * X - self.k_drain_AKG * AKG - mu * AKG
        
        # 胞内谷氨酸
        dGlu_in_dt = (v_GDH * X - v_sec * X - self.k_drain_Glu * Glu_in 
                      - mu * Glu_in + homeostasis_term)
        
        # 胞内NADPH
        dNADPH_dt = (v_NADPH_production - v_GDH * X + v_relax)
        
        # 生物量
        dX_dt = mu * X - self.k_maintenance * X
        
        # 胞外谷氨酸
        dGlu_ext_dt = v_sec * X
        
        return [dGlc_ext_dt, dNH4_ext_dt, dICIT_dt, dAKG_dt, dGlu_in_dt, 
                dNADPH_dt, dX_dt, dGlu_ext_dt, dfold_ICD_dt, dfold_GDH_dt]
    
    def simulate(self, t7_activity_func, t_end=48.0, dt=0.1, initial_conditions=None):
        """
        运行完整的治疗性谷氨酸代谢模拟
        
        Parameters:
        -----------
        t7_activity_func : float or function
            T7聚合酶活性。可以是常数或时间函数
        t_end : float
            模拟时间 (h)
        dt : float
            时间步长 (h)  
        initial_conditions : dict or None
            初始条件。如果为None，使用默认值
            
        Returns:
        --------
        t : array
            时间数组
        solution : array
            状态变量解 [Glc_ext, NH4_ext, ICIT, AKG, Glu_in, NADPH, X, Glu_ext, fold_ICD, fold_GDH]
        """
        t = np.arange(0, t_end, dt)
        
        # 默认初始条件
        if initial_conditions is None:
            if self.strain_type == 'wildtype':
                y0 = [
                    50.0,   # Glc_ext (mM) - 充足葡萄糖
                    10.0,   # NH4_ext (mM) - 充足铵
                    0.1,    # ICIT (mM) - 基线
                    0.5,    # AKG (mM) - 基线  
                    20.0,   # Glu_in (mM) - 野生型稳态
                    0.10,   # NADPH (mM) - 基线
                    0.1,    # X (gDW/L) - 初始生物量
                    0.0,    # Glu_ext (mM) - 无外排
                    1.0,    # fold_ICD - 无过表达
                    1.0     # fold_GDH - 无过表达
                ]
            else:
                y0 = [
                    50.0,   # Glc_ext (mM)
                    10.0,   # NH4_ext (mM)
                    0.1,    # ICIT (mM)
                    0.5,    # AKG (mM)
                    20.0,   # Glu_in (mM) - 工程菌正常情况下也是20mM
                    0.10,   # NADPH (mM)
                    0.1,    # X (gDW/L)
                    0.0,    # Glu_ext (mM)
                    1.0,    # fold_ICD - 基线
                    1.0     # fold_GDH - 基线
                ]
        else:
            y0 = [
                initial_conditions.get('Glc_ext', 50.0),
                initial_conditions.get('NH4_ext', 10.0),
                initial_conditions.get('ICIT', 0.1),
                initial_conditions.get('AKG', 0.5),
                initial_conditions.get('Glu_in', 20.0),  # 无论野生型还是工程菌正常都是20mM
                initial_conditions.get('NADPH', 0.10),
                initial_conditions.get('X', 0.1),
                initial_conditions.get('Glu_ext', 0.0),
                initial_conditions.get('fold_ICD', 1.0),
                initial_conditions.get('fold_GDH', 1.0)
            ]
        
        # 积分ODE系统
        solution = odeint(self.dydt, y0, t, args=(t7_activity_func,))
        
        return t, solution
    
    def analyze_therapeutic_performance(self, t, solution):
        """
        分析治疗性能指标
        
        Returns:
        --------
        metrics : dict
            关键治疗指标
        """
        # 提取关键变量
        Glu_in = solution[:, 4]
        Glu_ext = solution[:, 7]
        fold_GDH = solution[:, 9]
        
        # 找到热激期间（GDH表达>10倍）
        heat_shock_mask = fold_GDH > 10.0
        
        metrics = {
            'max_intracellular_glu': np.max(Glu_in),
            'max_extracellular_glu': np.max(Glu_ext),
            'final_intracellular_glu': Glu_in[-1],
            'final_extracellular_glu': Glu_ext[-1],
            'heat_shock_duration': np.sum(heat_shock_mask) * (t[1] - t[0]),
            'therapeutic_target_met': {
                'intracellular_peak_50mM': np.max(Glu_in) >= 50.0,
                'extracellular_peak_30mM': np.max(Glu_ext) >= 30.0,
                'final_recovery_20mM': abs(Glu_in[-1] - 20.0) <= 8.0
            }
        }
        
        return metrics

# === 便捷函数 ===
def create_heat_shock_protocol(shock_start=8.0, shock_duration=4.0, 
                             t7_low=50.0, t7_high=3000.0):
    """
    创建热激协议的T7活性函数
    
    Parameters:
    -----------
    shock_start : float
        热激开始时间 (h)
    shock_duration : float
        热激持续时间 (h)  
    t7_low : float
        基线T7活性
    t7_high : float
        热激T7活性
        
    Returns:
    --------
    function
        T7活性的时间函数
    """
    def t7_function(t):
        if shock_start <= t <= shock_start + shock_duration:
            return t7_high
        else:
            return t7_low
    return t7_function

# === 向后兼容的别名 ===
GluMetabolismModel = TherapeuticGluModel

# === 示例：治疗性工程菌模型验证 ===
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # 解决中文字体显示问题
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    except Exception as e:
        print(f"设置中文字体失败: {e}")

    print("=" * 60)
    print("    iGEM工程菌治疗性谷氨酸代谢模型验证")
    print("=" * 60)
    
    # === 1. 定义治疗性优化参数 ===
    therapeutic_params = {
        # 酶表达参数（基于文档要求）
        'fold_ICD_max': 1000.0,        # ICD过表达1000倍
        'fold_GDH_max': 1500.0,        # GDH过表达1500倍
        'K_T7': 800.0,                 # T7激活阈值
        'tau_enzyme': 0.05,             # 快速响应
        'n_hill': 3.0,                 # Hill系数
        
        # 外排控制参数
        'accum_threshold': 55.0,        # 累积阈值55mM
        'export_accum_suppression': 0.05,  # 强力抑制外排
        'recovery_threshold': 20.0,     # 恢复阈值20mM
        'postshock_export_boost': 10.0, # 10倍外排增强
        
        # 基础代谢参数优化
        'V_max_base_GDH': 30.0,        # 提高GDH基础活性
        'V_max_base_ICD': 25.0,        # 提高ICD基础活性
        'k_sec_base': 0.8,             # 基础外排速率
        'k_maintenance': 0.08,         # 降低维持成本
        
        # NADPH供应优化
        'k_PPP': 0.4,                  # 增强PPP通量
        'lambda_NADPH': 2.0,           # 快速NADPH平衡
        'NADPH_set': 0.15,             # 提高NADPH靶值
    }
    
    # === 2. 创建工程菌和野生型模型 ===
    engineered_model = TherapeuticGluModel(strain_type='engineered', **therapeutic_params)
    wildtype_model = TherapeuticGluModel(strain_type='wildtype', **therapeutic_params)
    
    print("✓ 模型已创建 (工程菌 & 野生型)")
    
    # === 3. 定义热激协议 ===
    heat_shock_protocol = create_heat_shock_protocol(
        shock_start=8.0,    # 8小时后开始热激
        shock_duration=4.0, # 持续4小时
        t7_low=50.0,        # 基线T7
        t7_high=3000.0      # 热激T7
    )
    
    print("✓ 热激协议已定义 (8-12h, T7: 50→3000)")
    
    # === 4. 运行模拟 ===
    print("\n正在运行模拟...")
    
    # 工程菌热激模拟
    t_eng, sol_eng = engineered_model.simulate(
        t7_activity_func=heat_shock_protocol,
        t_end=48.0,
        dt=0.1
    )
    
    # 野生型对照
    t_wt, sol_wt = wildtype_model.simulate(
        t7_activity_func=50.0,  # 恒定低T7
        t_end=48.0,
        dt=0.1
    )
    
    print("✓ 模拟完成")
    
    # === 5. 分析治疗性能 ===
    metrics_eng = engineered_model.analyze_therapeutic_performance(t_eng, sol_eng)
    metrics_wt = wildtype_model.analyze_therapeutic_performance(t_wt, sol_wt)
    
    print("\n" + "=" * 40)
    print("         治疗性能分析结果")
    print("=" * 40)
    
    print(f"\n【工程菌治疗指标】")
    print(f"  胞内Glu峰值: {metrics_eng['max_intracellular_glu']:.2f} mM")
    print(f"  胞外Glu峰值: {metrics_eng['max_extracellular_glu']:.2f} mM")
    print(f"  最终胞内Glu: {metrics_eng['final_intracellular_glu']:.2f} mM")
    print(f"  热激持续时间: {metrics_eng['heat_shock_duration']:.1f} h")
    
    print(f"\n【野生型对照指标】")
    print(f"  胞内Glu峰值: {metrics_wt['max_intracellular_glu']:.2f} mM")
    print(f"  胞外Glu峰值: {metrics_wt['max_extracellular_glu']:.2f} mM")
    print(f"  最终胞内Glu: {metrics_wt['final_intracellular_glu']:.2f} mM")
    
    print(f"\n【治疗目标达成情况】")
    targets = metrics_eng['therapeutic_target_met']
    print(f"  ✓ 胞内Glu≥50mM: {'成功' if targets['intracellular_peak_50mM'] else '失败'}")
    print(f"  ✓ 胞外Glu≥30mM: {'成功' if targets['extracellular_peak_30mM'] else '失败'}")
    print(f"  ✓ 最终回归~20mM: {'成功' if targets['final_recovery_20mM'] else '失败'}")
    
    # === 6. 生成分析图 ===
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('iGEM工程菌治疗性谷氨酸代谢模型分析', fontsize=16, fontweight='bold')
        
        # 胞内谷氨酸
        axes[0,0].plot(t_eng, sol_eng[:, 4], 'r-', linewidth=2, label='工程菌')
        axes[0,0].plot(t_wt, sol_wt[:, 4], 'b--', linewidth=2, label='野生型')
        axes[0,0].axhline(y=50, color='orange', linestyle=':', alpha=0.7, label='目标≥50mM')
        axes[0,0].axhline(y=20, color='green', linestyle=':', alpha=0.7, label='回归~20mM')
        axes[0,0].set_xlabel('时间 (h)')
        axes[0,0].set_ylabel('胞内Glu (mM)')
        axes[0,0].set_title('胞内谷氨酸动力学')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 胞外谷氨酸
        axes[0,1].plot(t_eng, sol_eng[:, 7], 'r-', linewidth=2, label='工程菌')
        axes[0,1].plot(t_wt, sol_wt[:, 7], 'b--', linewidth=2, label='野生型')
        axes[0,1].axhline(y=30, color='orange', linestyle=':', alpha=0.7, label='目标≥30mM')
        axes[0,1].set_xlabel('时间 (h)')
        axes[0,1].set_ylabel('胞外Glu (mM)')
        axes[0,1].set_title('胞外谷氨酸动力学')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 酶表达倍数
        axes[0,2].plot(t_eng, sol_eng[:, 8], 'g-', linewidth=2, label='ICD倍数')
        axes[0,2].plot(t_eng, sol_eng[:, 9], 'm-', linewidth=2, label='GDH倍数')
        axes[0,2].axvspan(8, 12, alpha=0.2, color='red', label='热激期')
        axes[0,2].set_xlabel('时间 (h)')
        axes[0,2].set_ylabel('过表达倍数')
        axes[0,2].set_title('酶表达动力学')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 代谢中间产物
        axes[1,0].plot(t_eng, sol_eng[:, 3], 'purple', linewidth=2, label='AKG')
        axes[1,0].plot(t_eng, sol_eng[:, 5], 'brown', linewidth=2, label='NADPH')
        axes[1,0].set_xlabel('时间 (h)')
        axes[1,0].set_ylabel('浓度 (mM)')
        axes[1,0].set_title('关键代谢物')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 生物量
        axes[1,1].plot(t_eng, sol_eng[:, 6], 'r-', linewidth=2, label='工程菌')
        axes[1,1].plot(t_wt, sol_wt[:, 6], 'b--', linewidth=2, label='野生型')
        axes[1,1].set_xlabel('时间 (h)')
        axes[1,1].set_ylabel('生物量 (gDW/L)')
        axes[1,1].set_title('生物量动力学')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # T7活性图
        t7_values = [heat_shock_protocol(t) for t in t_eng]
        axes[1,2].plot(t_eng, t7_values, 'k-', linewidth=2, label='T7活性')
        axes[1,2].axvspan(8, 12, alpha=0.2, color='red', label='热激期')
        axes[1,2].set_xlabel('时间 (h)')
        axes[1,2].set_ylabel('T7活性 (AU)')
        axes[1,2].set_title('T7聚合酶活性协议')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'therapeutic_glu_model_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ 分析图已保存: results/therapeutic_glu_model_analysis.png")
        
        # 显示图片
        plt.show()
        
    except Exception as e:
        print(f"\n图形生成失败: {e}")
    
    print("\n" + "=" * 60)
    print("          模型验证完成")
    print("=" * 60)
