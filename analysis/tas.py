"""
热导纳谱 (TAS): 各温度 C-f 谱 → 发射频率 → Arrhenius 求缺陷激活能.

陷阱能跟上交流信号时贡献电容, 跟不上时冻结, 所以 C(f) 在
ω0 = e_n 附近出现台阶; 台阶拐点就是 -dC/d(ln f) 的峰 (Walter et al.,
J. Appl. Phys. 80, 4411, 1996):

    ω0(T) = 2 ξ0 T² exp(-Ea / kT)
    ln(ω0 / T²) = ln(2 ξ0) - Ea / (k T)

为什么不用 G/ω 峰: 漏电流贡献的 G_dc/ω 在低频发散, 峰常被它吞掉, 取最大值
时会落到最低频点。-dC/d(ln f) 不受 G_dc 影响。

某温度下峰落在测量频段边缘或根本没有峰时, 该温度记为"无峰"而不参与拟合 ——
硬取最大值会把边界频率当成发射频率, 得出一条看似合理但毫无意义的直线。
"""
import csv
import logging
import os
import re
from typing import Dict, List

import numpy as np

from analysis.capacitance import capacitance_spectrum, read_spectrum_csv, smooth_log_spectrum

logger = logging.getLogger(__name__)

K_B = 1.380649e-23        # J/K
Q = 1.602176634e-19       # C
K_B_EV = K_B / Q          # eV/K

MANIFEST = 'tas_sequence.csv'
# 峰离频段两端至少这么多个点才算"内部峰"
EDGE_POINTS = 2
# 峰高至少达到 -dC/dlnf 最大值的这一比例, 用来滤掉噪声起伏
MIN_PROMINENCE = 0.15


def emission_peak(freq: np.ndarray, cap: np.ndarray) -> Dict:
    """返回 -dC/d(ln f) 曲线及其内部主峰频率 (无峰时 f_peak = nan)."""
    from scipy.signal import find_peaks

    freq = np.asarray(freq, dtype=float)
    cap = np.asarray(cap, dtype=float)
    lnf = np.log(freq)
    c_s = smooth_log_spectrum(cap, window=7)
    deriv = -np.gradient(c_s, lnf)

    out = {'deriv': deriv, 'f_peak': np.nan, 'peak_height': np.nan}
    top = float(np.nanmax(deriv)) if np.isfinite(deriv).any() else 0.0
    if top <= 0:
        return out
    peaks, props = find_peaks(deriv, prominence=MIN_PROMINENCE * top)
    peaks = [p for p in peaks if EDGE_POINTS <= p < len(deriv) - EDGE_POINTS]
    if not peaks:
        return out
    p = max(peaks, key=lambda i: deriv[i])
    # 在 ln f 上做三点抛物线插值, 峰位不受频点间距限制
    y0, y1, y2 = deriv[p - 1], deriv[p], deriv[p + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    shift = float(np.clip(shift, -1, 1))
    step = 0.5 * (lnf[p + 1] - lnf[p - 1])
    out['f_peak'] = float(np.exp(lnf[p] + shift * step))
    out['peak_height'] = float(y1)
    return out


def arrhenius_fit(temps_k: np.ndarray, f_peaks: np.ndarray) -> Dict:
    """ln(ω0/T²) 对 1/T 线性拟合; 需要至少 3 个有效温度点."""
    from scipy.stats import linregress

    temps_k = np.asarray(temps_k, dtype=float)
    f_peaks = np.asarray(f_peaks, dtype=float)
    ok = np.isfinite(temps_k) & np.isfinite(f_peaks) & (temps_k > 0) & (f_peaks > 0)
    t, f = temps_k[ok], f_peaks[ok]
    res = {'T': t, 'f_peak': f, 'n': int(ok.sum()),
           'inv_T': 1.0 / t if len(t) else t,
           'y': np.log(2 * np.pi * f / t ** 2) if len(t) else t,
           'Ea_eV': np.nan, 'Ea_err_eV': np.nan, 'xi0': np.nan, 'r2': np.nan}
    if len(t) < 3:
        return res
    fit = linregress(res['inv_T'], res['y'])
    res['Ea_eV'] = -fit.slope * K_B_EV
    res['Ea_err_eV'] = fit.stderr * K_B_EV
    # 截距 = ln(2 ξ0)
    res['xi0'] = float(np.exp(fit.intercept) / 2.0)
    res['r2'] = float(fit.rvalue ** 2)
    res['slope'] = float(fit.slope)
    res['intercept'] = float(fit.intercept)
    return res


def read_manifest(run_dir: str) -> List[Dict]:
    rows = []
    with open(os.path.join(run_dir, MANIFEST), newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def analyze_tas(run_dir: str) -> Dict:
    """分析一次 TAS 测量目录: 每个通道一组 (各温度谱 + 峰 + Arrhenius)."""
    rows = read_manifest(run_dir)
    if not rows:
        raise ValueError(f'{MANIFEST} 是空的: {run_dir}')
    channels: Dict[int, Dict] = {}
    for row in rows:
        folder = os.path.join(run_dir, row['folder'])
        t_actual = float(row['actual_K'])
        for name in sorted(os.listdir(folder)):
            m = re.match(r'eis_ch(\d+)\.csv$', name)
            if not m:
                continue
            ch = int(m.group(1))
            spec = read_spectrum_csv(os.path.join(folder, name))
            cs = capacitance_spectrum(spec['freq'], spec['z'])
            peak = emission_peak(spec['freq'], cs['C'])
            channels.setdefault(ch, {'spectra': []})['spectra'].append({
                'T': t_actual, 'target_T': float(row['target_K']),
                'freq': spec['freq'], 'C': cs['C'], **peak,
            })
    for ch, data in channels.items():
        data['spectra'].sort(key=lambda s: s['T'])
        temps = np.array([s['T'] for s in data['spectra']])
        peaks = np.array([s['f_peak'] for s in data['spectra']])
        data['arrhenius'] = arrhenius_fit(temps, peaks)
        missing = [s['T'] for s in data['spectra'] if not np.isfinite(s['f_peak'])]
        data['no_peak_T'] = missing
        if missing:
            logger.warning('通道 %d: %s K 下没有找到内部发射峰, 不参与 Arrhenius 拟合',
                           ch, ', '.join(f'{t:g}' for t in missing))
    return {'run_dir': run_dir, 'channels': channels}


def save_tas_results(result: Dict):
    """写 tas_peaks.csv 并画 C-f、-dC/dlnf、Arrhenius 三张图."""
    run_dir = result['run_dir']
    with open(os.path.join(run_dir, 'tas_peaks.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(('channel', 'T_K', 'target_T_K', 'f_peak_Hz', 'peak_height_F'))
        for ch, data in sorted(result['channels'].items()):
            for s in data['spectra']:
                w.writerow((ch, s['T'], s['target_T'], s['f_peak'], s['peak_height']))
    with open(os.path.join(run_dir, 'tas_arrhenius.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(('channel', 'Ea_eV', 'Ea_err_eV', 'xi0', 'R2', 'n_temperatures'))
        for ch, data in sorted(result['channels'].items()):
            a = data['arrhenius']
            w.writerow((ch, a['Ea_eV'], a['Ea_err_eV'], a['xi0'], a['r2'], a['n']))
    _plot_tas(result)


def _plot_tas(result: Dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    for ch, data in sorted(result['channels'].items()):
        spectra = data['spectra']
        colors = cm.coolwarm(np.linspace(0, 1, max(len(spectra), 2)))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for s, color in zip(spectra, colors):
            label = f"{s['T']:g} K"
            axes[0].semilogx(s['freq'], s['C'] * 1e9, '-', color=color, label=label)
            axes[1].semilogx(s['freq'], s['deriv'] * 1e9, '-', color=color, label=label)
            if np.isfinite(s['f_peak']):
                h = np.interp(np.log(s['f_peak']), np.log(s['freq']), s['deriv'])
                axes[1].plot(s['f_peak'], h * 1e9, 'v', color=color, ms=7)
        axes[0].set(xlabel='f [Hz]', ylabel='C_p [nF]', title=f'ch{ch}: C-f')
        axes[1].set(xlabel='f [Hz]', ylabel='-dC/dln f [nF]',
                    title='-dC/dln f (markers = emission peak)')
        a = data['arrhenius']
        if a['n']:
            axes[2].plot(a['inv_T'] * 1000, a['y'], 'o', label='peaks')
        if np.isfinite(a['Ea_eV']):
            x = np.linspace(a['inv_T'].min(), a['inv_T'].max(), 50)
            axes[2].plot(x * 1000, a['intercept'] + a['slope'] * x, '-',
                         label=f"Ea = {a['Ea_eV']:.3f} +/- {a['Ea_err_eV']:.3f} eV\n"
                               f"R^2 = {a['r2']:.4f}")
            axes[2].set_title('Arrhenius')
        else:
            axes[2].set_title(f"Arrhenius: need >= 3 peaks (have {a['n']})")
        axes[2].set(xlabel='1000/T [1/K]', ylabel='ln(omega_0 / T^2)')
        for ax in axes:
            ax.grid(True, which='both', alpha=0.3)
        axes[0].legend(fontsize=8)
        if a['n']:
            axes[2].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(result['run_dir'], f'tas_ch{ch}.png'), dpi=150)
        plt.close(fig)
