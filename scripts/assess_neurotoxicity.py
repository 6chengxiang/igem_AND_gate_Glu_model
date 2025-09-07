# --- path bootstrap ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------------

# scripts/assess_neurotoxicity.py
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    from models.pk_toxicity import (
        PKParams, ToxicityThresholds,
        simulate_three_comp_pk, assess_neurotoxicity
    )
except Exception as e:
    print("[ERROR] failed to import models.pk_toxicity:", e, flush=True)
    raise


def trapezoid_flux(t, t_on, t_plateau, t_off, ramp_hours, peak_umol_per_h):
    """
    2 段线性渐变 + 平台的梯形通量（单位 μmol/h）：
    - 从 t_on 开始，用 ramp_hours 线性上升到 peak；
    - 到 t_plateau 进入平台，保持到 t_off；
    - 之后用 ramp_hours 线性下降到 0。
    """
    # 上升段
    if t_on <= t < t_on + ramp_hours:
        return peak_umol_per_h * (t - t_on) / ramp_hours
    # 平台段
    if t_on + ramp_hours <= t < t_off:
        # 介于 (t_plateau) 与 t_off 区间保证平台
        if t < t_plateau:
            return peak_umol_per_h
        if t_plateau <= t < t_off:
            return peak_umol_per_h
    # 下降段
    if t_off <= t < t_off + ramp_hours:
        return peak_umol_per_h * (1 - (t - t_off) / ramp_hours)
    # 其他时间为 0
    return 0.0


def main():
    print("[INFO] preparing timeline & physiologic-ish flux...", flush=True)

    # ==== 配置区：可改 ====
    sim_hours = 48.0
    dt_h = 0.1
    baseline_uM = 50.0  # 血浆/组织/脑脊液三个室的基线
    intensity = "mild"  # 可改为 'high' 看更强分泌
    # mild / high 两档峰值（μmol/h）
    peak_umol_per_h = 0.02 if intensity == "mild" else 0.2
    # 通量时序（单位：小时）
    t_on = 8.0
    ramp_hours = 2.0
    t_plateau = t_on + ramp_hours   # 10 h 进入平台
    t_off = 34.0                    # 34 h 开始下降
    # =====================

    # 时间网格
    t_h = np.linspace(0.0, sim_hours, int(sim_hours / dt_h) + 1)

    # 构造理想化梯形通量（作用在肿瘤室，单位 μmol/h）
    S_t = np.array([
        trapezoid_flux(t, t_on=t_on, t_plateau=t_plateau, t_off=t_off,
                       ramp_hours=ramp_hours, peak_umol_per_h=peak_umol_per_h)
        for t in t_h
    ], dtype=float)

    print(
        "[CHK] S_t stats -> min={:.3g} μmol/h, max={:.3g} μmol/h, nonzero={} / {}".format(
            float(S_t.min()), float(S_t.max()), int((S_t != 0).sum()), S_t.size
        ),
        flush=True
    )

    print("[INFO] running PK integration...", flush=True)
    params = PKParams()
    # 给三室都设定 ~50 μM 的初值，表示基线
    Cb, Ct, Cn = simulate_three_comp_pk(
        t_grid_h=t_h,
        S_t_umol_per_h=S_t,
        params=params,
        Cb0_uM=baseline_uM, Ct0_uM=baseline_uM, Cn0_uM=baseline_uM
    )

    print("[INFO] assessing risk...", flush=True)
    thr = ToxicityThresholds()
    report = assess_neurotoxicity(Cb, t_h, thr)

    print("\n=== Neurotoxicity Risk Report (physio-ish) ===", flush=True)
    for k, v in report.items():
        print(f"{k}: {v}", flush=True)

    # 输出目录（带时间戳与强度，避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path("results") / f"neurotoxicity_{timestamp}_{intensity}"
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "plasma_glu_neurotoxicity.png"

    # 画图
    print(f"[INFO] saving figure -> {out_png}", flush=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(t_h, Cb, label="Plasma Glu (μM)")
    plt.axhline(baseline_uM, linestyle=":", label=f"Baseline {baseline_uM:.0f} μM")
    plt.axhline(thr.caution_um, linestyle="--", label=f"Caution {thr.caution_um:.0f} μM")
    plt.axhline(thr.danger_um, linestyle="--", label=f"Danger {thr.danger_um/1000:.1f} mM")

    # 辅助标注：通量窗口
    plt.axvspan(t_on, t_off, alpha=0.1, label="Tumor secretion window")

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration (μM)")
    plt.title(
        f"Plasma Glutamate vs Neurotoxicity Thresholds\n"
        f"(Intensity={intensity}, peak={peak_umol_per_h} μmol/h)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[INFO] done.", flush=True)


if __name__ == "__main__":
    print("Running assess_neurotoxicity.py ...", flush=True)
    try:
        main()
    except Exception as e:
        print("[FATAL] script crashed:", e, flush=True)
        raise
    print("Done.", flush=True)
