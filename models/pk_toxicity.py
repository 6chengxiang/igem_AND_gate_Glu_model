# models/pk_toxicity.py
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

@dataclass
class PKParams:
    # 小鼠理想参数（单位见注释）
    Vb: float = 0.002    # L, 血浆体积 ~2 mL
    Vt: float = 0.0005   # L, 肿瘤体积 0.5 mL
    Vn: float = 0.02     # L, 正常组织等效 20 mL
    k_bt: float = 1.0    # h^-1, 血<->肿瘤交换
    k_bn: float = 0.5    # h^-1, 血<->正常组织交换
    k_b_clr: float = 0.5 # h^-1, 血浆清除
    k_t_clr: float = 0.2 # h^-1, 肿瘤清除
    k_n_clr: float = 0.2 # h^-1, 正常组织清除

@dataclass
class ToxicityThresholds:
    caution_um: float = 100.0   # μM, 提示阈值
    danger_um: float = 1000.0   # μM, 危险阈值（1 mM）

def _interp_series(t_query, t_src, y_src):
    return np.interp(t_query, t_src, y_src)

def simulate_three_comp_pk(
    t_grid_h: np.ndarray,
    S_t_umol_per_h: np.ndarray,
    params: PKParams = PKParams(),
    Cb0_uM: float = 0.0, Ct0_uM: float = 0.0, Cn0_uM: float = 0.0,
):
    """
    三隔室 PK:
    状态变量 = [Cb, Ct, Cn] (μM)
    S_t(t)   = 肿瘤分泌通量 (μmol/h)
    """
    t0, t1 = float(t_grid_h[0]), float(t_grid_h[-1])

    def rhs(t, y):
        Cb, Ct, Cn = y
        S_t = _interp_series(t, t_grid_h, S_t_umol_per_h)
        Vb, Vt, Vn = params.Vb, params.Vt, params.Vn
        kbt, kbn = params.k_bt, params.k_bn
        kbclr, ktclr, knclr = params.k_b_clr, params.k_t_clr, params.k_n_clr

        J_bt = kbt * (Vt*Ct - Vb*Cb)
        J_bn = kbn * (Vn*Cn - Vb*Cb)

        dCb = (J_bt + J_bn)/Vb - kbclr*Cb
        dCt = (S_t + kbt*(Vb*Cb - Vt*Ct))/Vt - ktclr*Ct
        dCn = (kbn*(Vb*Cb - Vn*Cn))/Vn - knclr*Cn
        return [dCb, dCt, dCn]

    sol = solve_ivp(rhs, (t0, t1), [Cb0_uM, Ct0_uM, Cn0_uM],
                    t_eval=t_grid_h, method="LSODA", rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"PK ODE failed: {sol.message}")
    Cb, Ct, Cn = sol.y
    return Cb, Ct, Cn

def assess_neurotoxicity(Cb_uM: np.ndarray, t_h: np.ndarray,
                         thr: ToxicityThresholds = ToxicityThresholds()):
    Cb_max = float(np.max(Cb_uM))
    above_caution = Cb_uM >= thr.caution_um
    above_danger  = Cb_uM >= thr.danger_um
    t_above_caution_h = float(np.trapz(above_caution.astype(float), t_h))
    t_above_danger_h  = float(np.trapz(above_danger.astype(float),  t_h))
    return {
        "Cb_max_uM": Cb_max,
        "time_above_caution_h": t_above_caution_h,
        "time_above_danger_h":  t_above_danger_h,
        "flag_caution": bool(np.any(above_caution)),
        "flag_danger":  bool(np.any(above_danger)),
        "caution_threshold_uM": thr.caution_um,
        "danger_threshold_uM":  thr.danger_um,
    }
