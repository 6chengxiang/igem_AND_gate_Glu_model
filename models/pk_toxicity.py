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

def _prep_time_and_series(t_grid_h, S_t_umol_per_h):
    """保证时间严格递增，并清洗/平滑 S(t) 以避免求解器发散。"""
    t_src = np.asarray(t_grid_h, dtype=float)
    S_src = np.asarray(S_t_umol_per_h, dtype=float)

    # 排序并去重
    order = np.argsort(t_src)
    t_sorted = t_src[order]
    S_sorted = S_src[order]
    t = np.unique(t_sorted)
    if t.size < 2:
        raise ValueError("t_h needs at least 2 unique points")

    # 重采样 S 到去重后的时间轴
    S = np.interp(t, t_sorted, S_sorted)

    # 清洗：去 NaN/Inf，截断极端尖峰，非负
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(S > 0):
        upper = np.percentile(S, 99.5)
        if upper <= 0:
            upper = np.max(S)
        S = np.clip(S, 0.0, upper)
    else:
        S = np.zeros_like(S)

    # 轻微平滑（移动平均，避免尖峰）；保持长度不变
    if S.size >= 5:
        win = max(3, int(0.02 * len(S)))
        if win % 2 == 0:
            win += 1
        win = min(win, len(S) if len(S) % 2 == 1 else len(S)-1)
        if win >= 3:
            pad = win // 2
            S_pad = np.pad(S, (pad, pad), mode="edge")
            ker = np.ones(win, dtype=float) / win
            S = np.convolve(S_pad, ker, mode="valid")
            S = S[:len(t)]
    return t, S

def simulate_three_comp_pk(
    t_grid_h: np.ndarray,
    S_t_umol_per_h: np.ndarray,
    params: PKParams = PKParams(),
    Cb0_uM: float = 0.0, Ct0_uM: float = 0.0, Cn0_uM: float = 0.0,
):
    """
    三隔室 PK（鲁棒版，BDF/Radau）
    状态变量 = [Cb, Ct, Cn] (μM)
    S_t(t)   = 肿瘤分泌通量 (μmol/h)
    """
    # 预处理时间与通量
    t, S = _prep_time_and_series(t_grid_h, S_t_umol_per_h)
    t0, t1 = float(t[0]), float(t[-1])

    # 体积/参数安全夹紧
    tiny = 1e-12
    Vb = max(params.Vb, tiny)
    Vt = max(params.Vt, tiny)
    Vn = max(params.Vn, tiny)
    kbt, kbn = float(params.k_bt), float(params.k_bn)
    kbclr, ktclr, knclr = float(params.k_b_clr), float(params.k_t_clr), float(params.k_n_clr)

    # 插值器
    def S_interp(tt):
        if tt <= t[0]:  return float(S[0])
        if tt >= t[-1]: return float(S[-1])
        i = np.searchsorted(t, tt) - 1
        i = np.clip(i, 0, len(t)-2)
        w = (tt - t[i]) / (t[i+1] - t[i])
        return float(S[i] * (1.0 - w) + S[i+1] * w)

    def rhs(tt, y):
        Cb, Ct, Cn = y
        # 防止负值传播
        Cb = max(Cb, 0.0); Ct = max(Ct, 0.0); Cn = max(Cn, 0.0)

        Sin = S_interp(tt)        # μmol/h
        # 通量（μmol/h）
        J_bt = kbt * (Vt * Ct - Vb * Cb)
        J_bn = kbn * (Vn * Cn - Vb * Cb)

        dCb = (J_bt + J_bn) / Vb - kbclr * Cb
        dCt = (Sin + kbt * (Vb * Cb - Vt * Ct)) / Vt - ktclr * Ct
        dCn = (kbn * (Vb * Cb - Vn * Cn)) / Vn - knclr * Cn

        cap = 1e6
        return [
            float(np.clip(dCb, -cap, cap)),
            float(np.clip(dCt, -cap, cap)),
            float(np.clip(dCn, -cap, cap)),
        ]

    y0 = [max(Cb0_uM, 0.0), max(Ct0_uM, 0.0), max(Cn0_uM, 0.0)]
    min_dt = float(np.min(np.diff(t)))
    max_step = max(min_dt, 1e-3)

    last_msg = "unknown"
    for method in ("BDF", "Radau"):
        sol = solve_ivp(
            rhs, (t0, t1), y0, t_eval=t,
            method=method, rtol=1e-6, atol=1e-9, max_step=max_step
        )
        if sol.success:
            Cb, Ct, Cn = sol.y
            return Cb, Ct, Cn
        last_msg = sol.message

    raise RuntimeError(f"PK ODE failed: {last_msg}")

def assess_neurotoxicity(Cb_uM: np.ndarray, t_h: np.ndarray,
                         thr: ToxicityThresholds = ToxicityThresholds()):
    Cb_uM = np.asarray(Cb_uM, float)
    t_h   = np.asarray(t_h, float)

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
