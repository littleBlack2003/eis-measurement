"""
活性层电容: 由 C-f 谱的高频平台读几何电容, 换算活性层厚度.

    C_p(ω) = Im(1/Z) / ω              (并联模型电容)
    d      = ε0 εr A / C_geo          (已知 εr 求厚度)
    εr     = C_geo d / (ε0 A)         (已知厚度求等效介电常数)

低频处陷阱/离子能跟上交流信号, C 偏大; 频率升高后它们冻结, C 落到只由
活性层几何决定的平台。再往上, 串联电阻 Rs 让并联模型电容滚降 (截止约
1/(2π Rs C)), 所以平台取"最高频的那一段平坦区", 而不是直接取最高频点。
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS0 = 8.8541878128e-12   # F/m

# 平坦判据: |d lnC / d lnf| 低于该值视为平台 (频率每变 e 倍, C 变化 < 3%)
PLATEAU_SLOPE_TOL = 0.03
PLATEAU_MIN_POINTS = 3


def read_spectrum_csv(path: str) -> Dict:
    """读 save_eis_channel 写出的 CSV, 去掉无效点 (仪器状态 -1/1/2 存为 NaN)."""
    df = pd.read_csv(path)
    freq = df['freq [Hz]'].to_numpy(dtype=float)
    z = df['Z_real [ohm]'].to_numpy(dtype=float) + 1j * df['Z_imag [ohm]'].to_numpy(dtype=float)
    keep = np.isfinite(freq) & np.isfinite(z)
    order = np.argsort(freq[keep])
    return {'freq': freq[keep][order], 'z': z[keep][order], 'path': path}


def capacitance_spectrum(freq: np.ndarray, z: np.ndarray) -> Dict:
    """并联模型 C_p 与 G_p/ω."""
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        y = 1.0 / np.asarray(z, dtype=complex)
    return {'C': y.imag / omega, 'G_over_omega': y.real / omega}


def smooth_log_spectrum(values: np.ndarray, window: int = 5) -> np.ndarray:
    """在 (近似均匀的) 对数频率轴上做 Savitzky-Golay 平滑; 点太少时原样返回."""
    from scipy.signal import savgol_filter
    n = len(values)
    if n < window + 2:
        return np.asarray(values, dtype=float)
    return savgol_filter(values, window, 2)


def find_geometric_capacitance(freq: np.ndarray, cap: np.ndarray,
                               slope_tol: float = PLATEAU_SLOPE_TOL,
                               min_points: int = PLATEAU_MIN_POINTS) -> Dict:
    """在 C-f 谱上找最高频的平坦段, 取中位数为几何电容.

    找不到平坦段时退而取上半频段里最平的那一点, 并标记 found=False ——
    此时数值只能参考 (多半是陷阱响应一直延伸到测量上限, 或 Rs 滚降太早)。
    """
    freq = np.asarray(freq, dtype=float)
    cap = np.asarray(cap, dtype=float)
    ok = np.isfinite(freq) & np.isfinite(cap) & (cap > 0)
    freq, cap = freq[ok], cap[ok]
    if len(freq) < min_points + 2:
        raise ValueError(f'有效点太少 ({len(freq)}), 无法判断电容平台')

    lnf = np.log(freq)
    lnc = smooth_log_spectrum(np.log(cap))
    slope = np.gradient(lnc, lnf)
    flat = np.abs(slope) < slope_tol

    # 连续平坦段 (起止下标, 含两端)
    runs, start = [], None
    for i, f in enumerate(flat):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flat) - 1))
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_points]

    if runs:
        lo, hi = max(runs, key=lambda r: r[1])
        found = True
    else:
        upper = np.arange(len(freq) // 2, len(freq))
        best = int(upper[np.argmin(np.abs(slope[upper]))])
        lo = hi = best
        found = False
        logger.warning('C-f 谱上没有找到平坦段, 几何电容取最平的一点, 仅供参考')

    sel = slice(lo, hi + 1)
    return {
        'C_geo': float(np.median(cap[sel])),
        'C_spread': float(np.ptp(cap[sel]) / np.median(cap[sel])) if hi > lo else np.nan,
        'f_lo': float(freq[lo]),
        'f_hi': float(freq[hi]),
        'n_points': hi - lo + 1,
        'found': found,
        'slope': slope,
    }


def thickness_from_capacitance(c_geo: float, area_m2: float, eps_r: float) -> float:
    return EPS0 * eps_r * area_m2 / c_geo


def eps_from_thickness(c_geo: float, area_m2: float, thickness_m: float) -> float:
    return c_geo * thickness_m / (EPS0 * area_m2)


def analyze_capacitance(path: str, area_cm2: Optional[float] = None,
                        eps_r: Optional[float] = None,
                        thickness_nm: Optional[float] = None) -> Dict:
    """单个 CSV 的几何电容分析; 给了面积再配 εr 或厚度, 就换算另一项."""
    spec = read_spectrum_csv(path)
    cs = capacitance_spectrum(spec['freq'], spec['z'])
    geo = find_geometric_capacitance(spec['freq'], cs['C'])
    res = {
        'path': path,
        'freq': spec['freq'],
        'C': cs['C'],
        **geo,
        'area_cm2': area_cm2,
        'eps_r': eps_r,
        'thickness_nm': thickness_nm,
        'thickness_from_eps_nm': None,
        'eps_from_thickness': None,
    }
    if area_cm2:
        area_m2 = area_cm2 * 1e-4
        res['C_geo_per_area'] = geo['C_geo'] / area_cm2          # F/cm²
        if eps_r:
            res['thickness_from_eps_nm'] = thickness_from_capacitance(
                geo['C_geo'], area_m2, eps_r) * 1e9
        if thickness_nm:
            res['eps_from_thickness'] = eps_from_thickness(
                geo['C_geo'], area_m2, thickness_nm * 1e-9)
    return res


def analyze_cf_run(run_dir: str, area_cm2: Optional[float] = None,
                   eps_r: Optional[float] = None,
                   thickness_nm: Optional[float] = None) -> Dict:
    """分析一次 C-f 测量目录下全部通道, 写 capacitance_result.csv 和 cf.png."""
    import csv
    import os
    import re

    results = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'eis_ch(\d+)\.csv$', name)
        if m:
            results.append((f'ch{m.group(1)}', analyze_capacitance(
                os.path.join(run_dir, name), area_cm2, eps_r, thickness_nm)))
    if not results:
        raise FileNotFoundError(f'目录里没有 eis_ch*.csv: {run_dir}')
    with open(os.path.join(run_dir, 'capacitance_result.csv'), 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(('channel', 'C_geo_F', 'C_geo_per_area_F_cm2', 'plateau_f_lo_Hz',
                    'plateau_f_hi_Hz', 'plateau_points', 'plateau_found', 'area_cm2',
                    'eps_r', 'thickness_from_eps_nm', 'thickness_nm', 'eps_from_thickness'))
        for label, r in results:
            w.writerow((label, r['C_geo'], r.get('C_geo_per_area', ''), r['f_lo'], r['f_hi'],
                        r['n_points'], r['found'], area_cm2 or '', eps_r or '',
                        r['thickness_from_eps_nm'] or '', thickness_nm or '',
                        r['eps_from_thickness'] or ''))
    plot_capacitance(results, os.path.join(run_dir, 'cf.png'))
    return {'run_dir': run_dir, 'results': results}


def plot_capacitance(results, out_png: str):
    """C-f 曲线 + 选中的平台区间. 标题只用 ASCII (DejaVu Sans 没有中文字形)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, r in results:
        line, = ax.semilogx(r['freq'], r['C'] * 1e9, 'o-', ms=3, lw=1, label=label)
        ax.axvspan(r['f_lo'], r['f_hi'], color=line.get_color(), alpha=0.12)
        ax.axhline(r['C_geo'] * 1e9, color=line.get_color(), ls='--', lw=0.8)
    ax.set_xlabel('f [Hz]')
    ax.set_ylabel('C_p [nF]')
    ax.set_title('C-f, shaded = plateau used for C_geo')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
