#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合治疗模型模块 (Integrated Therapy Model Module)
—— 10态谷氨酸代谢 + 肿瘤/工程细胞动力学 的端到端模型
"""
import os
import numpy as np
from scipy.integrate import odeint

# 导入上游模块
from .and_gate import SimpleANDGate
from .glu_metabolism_en import GluMetabolismModel  # 10态版本
from .diffusion_pk import run_neurotox_from_glu_timeseries


class IntegratedTherapyModel:
    """
    状态向量 y (len=13):
      0  N_tumor   存活肿瘤细胞数
      1  D_tumor   死亡肿瘤细胞数
      2  N_eng     存活工程细胞数
      3  Glc_ext   细胞外葡萄糖
      4  NH4_ext   细胞外铵
      5  ICIT
      6  AKG
      7  Glu_in    细胞内谷氨酸 (mM)
      8  NADPH
      9  X         代谢模型的生物量代理
     10  Glu_ext   细胞外谷氨酸 (mM)
     11  fold_ICD
     12  fold_GDH
    """
    IDX = dict(
        N_tumor=0, D_tumor=1, N_eng=2,
        Glc_ext=3, NH4_ext=4, ICIT=5, AKG=6, Glu_in=7,
        NADPH=8, X=9, Glu_ext=10, fold_ICD=11, fold_GDH=12
    )

    def __init__(self, **params):
        # AND 门
        self.and_gate = SimpleANDGate()

        # 10态谷氨酸代谢模型（glu_metabolism_en.py）
        # 这里传入的额外键会被忽略无害；必要参数该文件内部有默认值
        self.glu_metabolism = GluMetabolismModel(**params)

        # 肿瘤生长 & 铁死亡参数（沿用你原来的）
        self.r = params.get('r', 0.01)
        self.K_tumor = params.get('K_tumor', 1e9)
        self.k_ferroptosis_max = params.get('k_ferroptosis_max', 15.0)
        self.K_glu = params.get('K_glu', 30.0)
        self.n_glu = params.get('n_glu', 5.0)

        # 工程细胞增长
        self.r_eng = params.get('r_eng', 0.2)
        self.K_eng = params.get('K_eng', 5e8)

        # 体积（用于神经毒性PK）
        self.V_cell = params.get('V_cell', 2e-12)
        self.V_tumor_ext = params.get('V_tumor_ext', 0.01)

        # 代谢模块默认初值（与 glu_metabolism_en.py 的 simulate 默认一致）
        self.defaults_glu10 = np.array([
            50.0,  # Glc_ext
            10.0,  # NH4_ext
            0.1,   # ICIT
            0.5,   # AKG
            20.0,  # Glu_in
            0.10,  # NADPH
            0.10,  # X
            0.0,   # Glu_ext
            1.0,   # fold_ICD
            1.0    # fold_GDH
        ], dtype=float)

    def get_t7_activity(self, env_conditions):
        """从AND门模块获取T7活性。"""
        return self.and_gate.get_t7_activity(
            env_conditions['O2_percent'],
            env_conditions['Temp_C']
        )

    def dydt(self, y, t, env_conditions):
        ix = self.IDX

        # --- 解包（细胞数量3个）---
        N_tumor = y[ix['N_tumor']]
        D_tumor = y[ix['D_tumor']]
        N_eng   = y[ix['N_eng']]

        # --- 代谢10态（按顺序组装，避免错位）---
        y_glu10 = [
            y[ix['Glc_ext']], y[ix['NH4_ext']], y[ix['ICIT']],    y[ix['AKG']],
            y[ix['Glu_in']],  y[ix['NADPH']],  y[ix['X']],        y[ix['Glu_ext']],
            y[ix['fold_ICD']], y[ix['fold_GDH']]
        ]

        # --- 环境 → T7 ---
        t7_activity = self.get_t7_activity(env_conditions)

        # --- 代谢导数（10个）---
        (dGlc_ext_dt, dNH4_ext_dt, dICIT_dt, dAKG_dt, dGlu_in_dt,
         dNADPH_dt, dX_dt, dGlu_ext_dt, dfold_ICD_dt, dfold_GDH_dt) = \
            self.glu_metabolism.dydt(y_glu10, t, t7_activity)

        # --- 肿瘤/工程细胞动力学 ---
        # 肿瘤增长（含承载量）
        growth_rate_tumor = self.r * N_tumor * (1 - (N_tumor + N_eng) / self.K_tumor)
        # 铁死亡由细胞外谷氨酸触发
        Glu_ext = y[ix['Glu_ext']]
        ferroptosis_rate = self.k_ferroptosis_max * (Glu_ext**self.n_glu) / (self.K_glu**self.n_glu + Glu_ext**self.n_glu)
        death_term_tumor = ferroptosis_rate * N_tumor

        # 工程细胞增长（T7 开关）
        n_hill = getattr(self.glu_metabolism, 'n_hill', 3.0)
        K_T7   = getattr(self.glu_metabolism, 'K_T7', 800.0)
        t7_hill_factor = (t7_activity**n_hill) / (K_T7**n_hill + t7_activity**n_hill)
        dN_eng_dt = self.r_eng * N_eng * (1 - (N_tumor + N_eng) / self.K_tumor) * t7_hill_factor

        dN_tumor_dt = growth_rate_tumor - death_term_tumor
        dD_tumor_dt = death_term_tumor

        # 数值保护（避免负增长在近零区间抖动）
        if N_tumor < 1.0:
            dN_tumor_dt = 0.0
        if N_eng < 1.0:
            dN_eng_dt = 0.0

        # --- 组装返回 ---
        dydt_vec = np.zeros_like(y)
        dydt_vec[ix['N_tumor']] = dN_tumor_dt
        dydt_vec[ix['D_tumor']] = dD_tumor_dt
        dydt_vec[ix['N_eng']]   = dN_eng_dt

        dydt_vec[ix['Glc_ext']]  = dGlc_ext_dt
        dydt_vec[ix['NH4_ext']]  = dNH4_ext_dt
        dydt_vec[ix['ICIT']]     = dICIT_dt
        dydt_vec[ix['AKG']]      = dAKG_dt
        dydt_vec[ix['Glu_in']]   = dGlu_in_dt
        dydt_vec[ix['NADPH']]    = dNADPH_dt
        dydt_vec[ix['X']]        = dX_dt
        dydt_vec[ix['Glu_ext']]  = dGlu_ext_dt
        dydt_vec[ix['fold_ICD']] = dfold_ICD_dt
        dydt_vec[ix['fold_GDH']] = dfold_GDH_dt

        return dydt_vec

    def simulate(self, env_conditions, t_end=100.0, dt=0.5, with_neurotox=True):
        """
        运行整合治疗模拟。
        返回：t, solution, neurotox（当 with_neurotox=False 时，neurotox 为 None）
        """
        t = np.arange(0, t_end, dt)

        # 初始条件：3(肿瘤/工程) + 10(代谢)
        y0 = np.zeros(13, dtype=float)
        y0[self.IDX['N_tumor']] = 1e6
        y0[self.IDX['D_tumor']] = 0.0
        y0[self.IDX['N_eng']]   = 5e5

        # 代谢默认初值（与 glu_metabolism_en.py 保持一致）
        y0[self.IDX['Glc_ext'] : self.IDX['Glu_ext'] + 1] = self.defaults_glu10[:8]
        y0[self.IDX['fold_ICD']] = self.defaults_glu10[8]
        y0[self.IDX['fold_GDH']] = self.defaults_glu10[9]

        solution = odeint(self.dydt, y0, t, args=(env_conditions,))

        # === 神经毒性评估 ===
        neurotox = None
        if with_neurotox:
            Glu_extra_mM = solution[:, self.IDX['Glu_ext']]
            neurotox = run_neurotox_from_glu_timeseries(
                t_h=t,
                glu_extra_mM=Glu_extra_mM,
                V_tumor_ext_L=self.V_tumor_ext,
                pk_params=None,
                tox_thr=None,
                baseline_uM=50.0
            )
        return t, solution, neurotox


# --- 示例（可直接运行本文件做烟雾测试）---
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("="*20 + " 整合治疗模型（10态）测试 " + "="*20)
    model = IntegratedTherapyModel()

    therapy_conditions = {'O2_percent': 1.0, 'Temp_C': 42.0}
    control_conditions = {'O2_percent': 21.0, 'Temp_C': 37.0}

    print("\n正在运行 '治疗' 条件下的模拟...")
    t_th, sol_th, nt_th = model.simulate(env_conditions=therapy_conditions, t_end=200, with_neurotox=True)
    print("正在运行 '对照' 条件下的模拟...")
    t_ct, sol_ct, nt_ct = model.simulate(env_conditions=control_conditions,  t_end=200, with_neurotox=True)
    print("模拟完成。")

    IX = model.IDX
    # 提取关键量（使用索引表避免硬编码）
    N_t_th, D_t_th, N_e_th = sol_th[:, IX['N_tumor']], sol_th[:, IX['D_tumor']], sol_th[:, IX['N_eng']]
    Glu_in_th, Glu_ex_th  = sol_th[:, IX['Glu_in']],  sol_th[:, IX['Glu_ext']]
    ICD_th, GDH_th        = sol_th[:, IX['fold_ICD']], sol_th[:, IX['fold_GDH']]

    N_t_ct, D_t_ct, N_e_ct = sol_ct[:, IX['N_tumor']], sol_ct[:, IX['D_tumor']], sol_ct[:, IX['N_eng']]
    Glu_in_ct, Glu_ex_ct  = sol_ct[:, IX['Glu_in']],  sol_ct[:, IX['Glu_ext']]
    ICD_ct, GDH_ct        = sol_ct[:, IX['fold_ICD']], sol_ct[:, IX['fold_GDH']]

    # 计算T7活性
    t7_th = model.get_t7_activity(therapy_conditions)
    t7_ct = model.get_t7_activity(control_conditions)

    # 计算治疗效果指标
    tumor_reduction_ratio = N_t_th[-1] / max(N_t_ct[-1], 1e-9)
    glu_ratio = Glu_ex_th[-1] / (Glu_ex_ct[-1] + 1e-9)

    print(f"\n=== 治疗效果分析（10态）===")
    print(f"T7活性 - 治疗组: {t7_th:.1f} AU, 对照组: {t7_ct:.1f} AU")
    print(f"谷氨酸浓度(外) - 治疗组: {Glu_ex_th[-1]:.3f} mM, 对照组: {Glu_ex_ct[-1]:.3f} mM")
    print(f"谷氨酸比值 (治疗/对照): {glu_ratio:.3f}")
    print(f"肿瘤细胞数量 - 治疗组: {N_t_th[-1]:.1e}, 对照组: {N_t_ct[-1]:.1e}")
    print(f"死亡细胞数量 - 治疗组: {D_t_th[-1]:.1e}, 对照组: {D_t_ct[-1]:.1e}")
    print(f"肿瘤细胞存活率 (治疗/对照): {tumor_reduction_ratio:.3f}")

    # 简单画图（治疗 vs 对照的外源Glu）
    try:
        os.makedirs("results", exist_ok=True)
        plt.figure(figsize=(9,5))
        plt.plot(t_th, Glu_ex_th, label='Therapy Glu_ext (mM)')
        plt.plot(t_ct, Glu_ex_ct, '--', label='Control Glu_ext (mM)')
        plt.xlabel('Time (h)'); plt.ylabel('Concentration (mM)')
        plt.legend(); plt.tight_layout()
        out_png = "results/glu_ext_10state.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"已保存: {out_png}")
    except Exception as e:
        print(f"绘图失败: {e}")
