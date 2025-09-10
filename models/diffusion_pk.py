#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩散与药代动力学模块 (Diffusion and Pharmacokinetics Module)

功能:
- 提供两种模型来描述治疗药物（如谷氨酸）在体内的分布和扩散：
  1. MultiCompartmentPK: 一个多室药代动力学(PK)模型，用于模拟药物在全身各主要器官（血液、肝脏、肿瘤等）的分布。基于ODE。
  2. TumorDiffusion: 一个反应-扩散模型，用于模拟药物在肿瘤微环境(TME)中的空间分布。基于偏微分方程(PDE)。

如何使用:
- 实例化 `MultiCompartmentPK` 或 `TumorDiffusion`。
- 调用相应模型的 `simulate` 方法来运行模拟。
"""

import numpy as np
from scipy.integrate import odeint
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from . import pk_toxicity as pkt

class MultiCompartmentPK:
    """
    多室药代动力学(PK)模型。
    
    该模型将人体简化为几个相互连接的室(compartment)，如血液、肝脏、
    健康组织和肿瘤。它通过ODE描述药物在这些室之间的流动和清除。
    """
    def __init__(self, **params):
        """
        初始化模型参数。
        
        Parameters:
        -----------
        q_ij : float
            从室i到室j的流速。
        k_elim : float
            药物在肝脏的清除率。
        V_i : float
            室i的体积。
        """
        # 体积 (L)
        self.V_blood = params.get('V_blood', 5.0)
        self.V_liver = params.get('V_liver', 1.5)
        self.V_tumor = params.get('V_tumor', 0.5)
        self.V_other = params.get('V_other', 60.0)
        
        # 流速 (L/hr)
        self.q_blood_liver = params.get('q_blood_liver', 90.0)
        self.q_blood_tumor = params.get('q_blood_tumor', 10.0)
        self.q_blood_other = params.get('q_blood_other', 200.0)
        
        # 清除率 (1/hr)
        self.k_elim_liver = params.get('k_elim_liver', 0.5)
        self.k_uptake_tumor = params.get('k_uptake_tumor', 0.2)

    def dydt(self, y, t, infusion_rate):
        """
        定义多室PK模型的ODE系统。
        
        y: array [C_blood, C_liver, C_tumor, C_other]
        """
        C_blood, C_liver, C_tumor, C_other = y
        
        # 血液室
        dC_blood_dt = (infusion_rate / self.V_blood +
                       (self.q_blood_liver * C_liver - self.q_blood_liver * C_blood) / self.V_blood +
                       (self.q_blood_tumor * C_tumor - self.q_blood_tumor * C_blood) / self.V_blood +
                       (self.q_blood_other * C_other - self.q_blood_other * C_blood) / self.V_blood)
        
        # 肝脏室
        dC_liver_dt = ((self.q_blood_liver * C_blood - self.q_blood_liver * C_liver) / self.V_liver -
                       self.k_elim_liver * C_liver)
                       
        # 肿瘤室
        dC_tumor_dt = ((self.q_blood_tumor * C_blood - self.q_blood_tumor * C_tumor) / self.V_tumor -
                       self.k_uptake_tumor * C_tumor)

        # 其他组织室
        dC_other_dt = ((self.q_blood_other * C_blood - self.q_blood_other * C_other) / self.V_other)
        
        return [dC_blood_dt, dC_liver_dt, dC_tumor_dt, dC_other_dt]

    def simulate(self, infusion_rate_func, t_end=24.0, dt=0.1):
        """
        运行PK模拟。
        
        infusion_rate_func: function
            一个函数 `f(t)`，返回在时间t的药物输注速率 (e.g., mg/hr)。
        """
        t = np.arange(0, t_end, dt)
        y0 = [0, 0, 0, 0]
        
        # 动态输注率
        infusion_rates = [infusion_rate_func(ti) for ti in t]
        
        solution = np.zeros((len(t), len(y0)))
        solution[0] = y0
        
        for i in range(1, len(t)):
            y_prev = solution[i-1]
            inf_rate = infusion_rates[i-1]
            dydt_val = self.dydt(y_prev, t[i-1], inf_rate)
            solution[i] = y_prev + np.array(dydt_val) * dt

        return t, solution

class TumorDiffusion:
    """
    肿瘤微环境中的反应-扩散模型 (1D)。
    
    使用有限差分法求解PDE: ∂C/∂t = D * ∂²C/∂x² - k_uptake * C
    """
    def __init__(self, **params):
        self.D = params.get('D', 0.01)  # 扩散系数 (cm²/hr)
        self.k_uptake = params.get('k_uptake', 0.2) # 肿瘤细胞摄取率 (1/hr)
        self.L = params.get('L', 1.0) # 肿瘤直径 (cm)
        self.Nx = params.get('Nx', 50) # 空间网格点数
        self.dx = self.L / (self.Nx - 1)

    def simulate(self, C_boundary, t_end=10.0, dt=0.01):
        """
        运行1D扩散模拟。
        
        C_boundary: float
            肿瘤边界的药物浓度 (mM)，由PK模型提供。
        """
        t = np.arange(0, t_end, dt)
        Nt = len(t)
        
        C = np.zeros((self.Nx, Nt))
        
        # 设置边界条件 (Dirichlet)
        C[0, :] = C_boundary
        C[-1, :] = C_boundary
        
        # 构建有限差分矩阵
        # 对于 spdiags，所有对角线数组必须具有相同长度
        A_diag = np.ones(self.Nx) * (1 + 2 * self.D * dt / self.dx**2 + self.k_uptake * dt)
        A_upper = np.zeros(self.Nx)
        A_upper[:-1] = -self.D * dt / self.dx**2
        A_lower = np.zeros(self.Nx)
        A_lower[1:] = -self.D * dt / self.dx**2
        
        A = spdiags([A_lower, A_diag, A_upper], [-1, 0, 1], self.Nx, self.Nx, format='csc')
        
        # 时间步进
        for n in range(Nt - 1):
            b = C[:, n]
            # 边界条件处理
            b[0] += self.D * dt / self.dx**2 * C_boundary
            b[-1] += self.D * dt / self.dx**2 * C_boundary
            
            C[:, n+1] = spsolve(A, b)
            
        return self.dx * np.arange(self.Nx), t, C
 # === Adapter: map Glu_extra(t) -> tumor secretion flux S_t(t) and run neurotox PK ===

def _finite_diff_flux_from_conc(
    t_h: np.ndarray,
    glu_extra_mM: np.ndarray,
    V_tumor_ext_L: float,
    clip_nonneg: bool = True,
) -> np.ndarray:
    """
    把细胞外浓度时序 C_extra(t) 转成肿瘤室分泌通量 S_t(t)（μmol/h）：
        S_t(t) ≈ d/dt [ C_extra(t) [mM] * V_ext [L] ] * 1000
    """
    t = np.asarray(t_h, dtype=float)
    c = np.asarray(glu_extra_mM, dtype=float)

    # 排序+去重，保证时间严格递增
    order = np.argsort(t)
    t = t[order]; c = c[order]
    t_unique = np.unique(t)
    if t_unique.size < 2:
        # 不足以做微分，返回全零通量
        return np.zeros_like(t_unique, dtype=float)
    if t_unique.size != t.size:
        c = np.interp(t_unique, t, c)
        t = t_unique

    # 数值微分（单位：mM/h）
    dc_dt_mM_per_h = np.gradient(c, t, edge_order=2)

    # mM * L = mmol  -> ×1000 变成 μmol
    umol_per_h = 1000.0 * V_tumor_ext_L * dc_dt_mM_per_h

    return np.clip(umol_per_h, 0.0, None) if clip_nonneg else umol_per_h



def run_neurotox_from_glu_timeseries(
    t_h: np.ndarray,
    glu_extra_mM: np.ndarray,
    V_tumor_ext_L: float,
    pk_params=None,
    tox_thr=None,
    baseline_uM: float = 50.0,
):
    """
    用 Glu_extra(t) 推导分泌通量 S_t(t)，驱动三室 PK，并评估神经毒性。
    返回 dict: {'t_h','S_t_umol_per_h','Cb_uM','Ct_uM','Cn_uM','tox_report'}
    """
    

    # ——— 1) 标准化 & 去重时间序列 —— 
    t_h = np.asarray(t_h, dtype=float)
    glu_extra_mM = np.asarray(glu_extra_mM, dtype=float)

    order = np.argsort(t_h)
    t_h = t_h[order]
    glu_extra_mM = glu_extra_mM[order]

    # 去重（严格递增）
    uniq_t = np.unique(t_h)
    if len(uniq_t) != len(t_h):
        # 重插值到去重后的时间轴
        glu_extra_mM = np.interp(uniq_t, t_h, glu_extra_mM)
        t_h = uniq_t

    # ——— 2) 浓度 → 分泌通量 S(t)（μmol/h），并清洗 —— 
    S_t = _finite_diff_flux_from_conc(t_h, glu_extra_mM, V_tumor_ext_L, clip_nonneg=True)
    S_t = np.nan_to_num(S_t, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(S_t > 0):
        # 去掉极端尖峰
        S_t = np.clip(S_t, 0.0, np.percentile(S_t, 99.5))
        # 轻微平滑，减少刚性（窗口≈2%长度，>=3且为奇数）
        if len(S_t) >= 5:
            win = max(3, int(0.02 * len(S_t)))
            if win % 2 == 0: win += 1
            win = min(win, len(S_t) if len(S_t) % 2 == 1 else len(S_t) - 1)
            if win >= 3:
                pad = win // 2
                S_pad = np.pad(S_t, (pad, pad), mode="edge")
                ker = np.ones(win, dtype=float) / win
                S_t = np.convolve(S_pad, ker, mode="valid")[:len(S_t)]
    else:
        S_t = np.zeros_like(S_t)

    # ——— 3) PK 参数/阈值对象 —— 
    pk_params = pk_params or pkt.PKParams()
    tox_thr   = tox_thr   or pkt.ToxicityThresholds()

    # ——— 4) 三室 PK（三个室的初始浓度给到生理基线 μM）—— 
    Cb_uM, Ct_uM, Cn_uM = pkt.simulate_three_comp_pk(
        t_grid_h=t_h,
        S_t_umol_per_h=S_t,
        params=pk_params,
        Cb0_uM=baseline_uM, Ct0_uM=baseline_uM, Cn0_uM=baseline_uM
    )

    # ——— 5) 神经毒性评估 —— 
    report = pkt.assess_neurotoxicity(Cb_uM, t_h, thr=tox_thr)

    return {
        "t_h": t_h,
        "S_t_umol_per_h": S_t,
        "Cb_uM": Cb_uM,
        "Ct_uM": Ct_uM,
        "Cn_uM": Cn_uM,
        "tox_report": report,
    }

   

# --- 示例 ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- 1. 测试多室PK模型 ---
    print("--- 1. 测试多室PK模型 ---")
    pk_model = MultiCompartmentPK()
    
    # 定义一个简单的输注方案：前2小时恒定输注
    def infusion(t):
        return 100.0 if t <= 2.0 else 0.0
        
    t_pk, sol_pk = pk_model.simulate(infusion, t_end=24)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_pk, sol_pk[:, 0], label='Blood')
    plt.plot(t_pk, sol_pk[:, 1], label='Liver')
    plt.plot(t_pk, sol_pk[:, 2], label='Tumor')
    plt.plot(t_pk, sol_pk[:, 3], label='Other Tissues')
    plt.title('Multi-Compartment PK Model')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 2. 测试肿瘤扩散模型 ---
    print("\n--- 2. 测试肿瘤扩散模型 ---")
    pde_model = TumorDiffusion()
    
    # 假设肿瘤边界浓度恒定为10mM
    C_bound = 10.0
    x, t_pde, C_pde = pde_model.simulate(C_bound, t_end=5)
    
    plt.figure(figsize=(10, 6))
    for i in range(0, len(t_pde), len(t_pde)//5):
        plt.plot(x, C_pde[:, i], label=f't = {t_pde[i]:.1f} hr')
    
    plt.title('1D Tumor Diffusion Model')
    plt.xlabel('Position in Tumor (cm)')
    plt.ylabel('Concentration (mM)')
    plt.legend()
    plt.grid(True)
    plt.show()
