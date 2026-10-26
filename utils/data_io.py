"""
EIS 数据保存: 每通道频谱 CSV + Nyquist/Bode 绘图.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


def ensure_output_dir(config: Dict) -> str:
    out_dir = config['output']['directory']
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_eis_channel(result: Dict, out_dir: str):
    """保存单通道 EIS 频谱到 CSV."""
    ch = result['channel']
    df = pd.DataFrame({
        'freq [Hz]':   result['freq'],
        'Z_real [ohm]': result['z'].real,
        'Z_imag [ohm]': result['z'].imag,
        '|Z| [ohm]':   result['z_mag'],
        'theta [deg]': result['theta_deg'],
        'C_p [F]':     result['C_p'],
        'G_p [S]':     result['G_p'],
        'Z_raw_real [ohm]': result['z_raw'].real,
        'Z_raw_imag [ohm]': result['z_raw'].imag,
        # 4284A FETCH? 状态码: 0 正常; -1/1/2 无效 (Z 为 NaN); 3/4 数值保留但不保证准确
        'status': result.get('status', np.zeros(len(result['freq']), dtype=int)),
    })
    fname = os.path.join(out_dir, f"eis_ch{ch}.csv")
    df.to_csv(fname, index=False)
    logger.info(f"Saved EIS channel {ch} to {fname}")


def _arc_view_limit(values, q_lo: float = 5.0, q_hi: float = 95.0,
                    pad: float = 0.35):
    """线性轴上限: 容纳整条弧, 但把极端离群点留在视野外.

    以前靠 ``f >= 1 kHz`` 硬裁掉低频点来防止 MΩ 级离群点把弧压扁, 代价是
    **真实数据被静默丢掉** —— 弧顶在 1 kHz 以下的器件会少画一大截。
    现在改为保留全部有效点, 只收紧坐标轴: 上限取高分位数再加若干倍
    分位跨度, 于是少数 MΩ 点落在框外而弧保持满量程。

    ``pad`` 由实测标定 (0.7 V 数据, 87 点):
      - 干净数据: q95 + 0.15*span 就已超过 max -> 上限恒等于 max, 永不误裁,
        所以 pad 取小只影响"有离群点时弧占多大屏"
      - 弧 + 1 个 MΩ 离群点: max/q95 ≈ 1, 上限完全由 pad 决定,
        pad=0.35 -> 弧仍占约 3/4 屏宽 (pad=2.0 时只剩 1/3)
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None
    lo_q = float(np.percentile(v, q_lo))
    hi_q = float(np.percentile(v, q_hi))
    span = max(hi_q - lo_q, 0.0)
    return float(min(v.max(), hi_q + pad * span))


def _valid_nyquist(r: Dict):
    """Nyquist 有效点: 第一象限 (Re Z > 0 且 -Im Z > 0)。"""
    z = np.asarray(r['z'])
    zr, zi = z.real, -z.imag
    mask = (zr > 0) & (zi > 0)
    return zr, zi, mask


def _fmt_freq(f: float) -> str:
    """频率轴端点的紧凑写法 (4 位有效数字, 别把 22.4676 Hz 整个塞进标题)."""
    if f >= 1e6:
        return f"{f / 1e6:.4g} MHz"
    if f >= 1e3:
        return f"{f / 1e3:.4g} kHz"
    return f"{f:.4g} Hz"


def plot_eis_all(results: List[Dict], out_dir: str):
    """绘制各通道 Nyquist 图与 Bode 图."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skip plotting")
        return

    # Nyquist (full range, log-log 以容纳 MΩ 级散点)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    n_plot = 0
    for r in results:
        zr, zi = r['z'].real, -r['z'].imag
        mask = (zr > 0) & (zi > 0)
        if mask.any():
            n_plot += int(mask.sum())
            ax.loglog(zr[mask], zi[mask], 'o', ms=4,
                      label=f"ch{r['channel']}")
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    title = "EIS Nyquist (log-log, full range)"
    if n_plot == 0:
        # 一个可绘点都没有 —— 多半是探针没接触好 (4284A 对开路也会返回
        # status=0 的"有效"读数, 于是整条谱的 Re Z 变成负数)。空坐标系配上
        # 空图例是最容易被误读的形态, 直接写明白。
        # 只能用 ASCII: DejaVu Sans 没有中文字形。
        title += "\nNo plottable point: Re Z <= 0 everywhere!"
        ax.text(0.5, 0.5,
                "no point in the first quadrant\n(Re Z <= 0 everywhere)\n"
                "check probe contact",
                transform=ax.transAxes, ha='center', va='center', fontsize=11,
                color='#c0392b', bbox=dict(boxstyle='round', fc='#fdecea',
                                           ec='#c0392b', alpha=0.9))
    ax.set_title(title)
    if n_plot:
        ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eis_nyquist.png'), dpi=150)
    plt.close(fig)

    # Nyquist zoom (线性轴看 RC 弧; 画全部有效点, 只自动收紧坐标范围)
    # 分两步: 先收集全部有效点并把视野上限定下来, 再滤掉视野外的点后才绘线。
    # 若先画后设 xlim/ylim, 'o-' 会拖一条线穿过边框伸到框外 —— 纯视觉噪声。
    payload = []
    for r in results:
        zr, zi, mask = _valid_nyquist(r)
        if mask.any():
            payload.append((r, zr[mask], zi[mask],
                            np.asarray(r['freq'], dtype=float)[mask]))

    xmax = _arc_view_limit(np.concatenate([p[1] for p in payload])) if payload else None
    ymax = _arc_view_limit(np.concatenate([p[2] for p in payload])) if payload else None

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    n_off, all_f = 0, []
    for r, zr, zi, ff in payload:
        keep = np.ones(zr.shape, dtype=bool)
        if xmax:
            keep &= zr <= xmax * 1.05
        if ymax:
            keep &= zi <= ymax * 1.05
        n_off += int((~keep).sum())
        all_f.append(ff[keep])
        ax.plot(zr[keep], zi[keep], 'o-', ms=4, lw=1,
                label=f"ch{r['channel']}")

    title = "EIS Nyquist (zoom)"
    if all_f:
        all_f = np.concatenate(all_f)
        if all_f.size:
            title = (f"EIS Nyquist (zoom, {_fmt_freq(all_f.min())}"
                     f" – {_fmt_freq(all_f.max())})")
    if xmax:
        ax.set_xlim(0, xmax * 1.05)
    if ymax:
        ax.set_ylim(0, ymax * 1.05)
    if n_off:
        # 视野外的点只是没画出来, 数据本身没被删 (CSV 仍完整) —— 标出来避免误读。
        # 注意: 标题只能用 ASCII —— matplotlib 默认字体 (DejaVu Sans) 没有中文
        # 字形, 中文会渲染成豆腐块 (实测会刷一屏 missing-glyph 警告)。
        title += f"\n{n_off} outlier point(s) outside view"
    if not payload:
        title += "\nNo valid point (Re Z <= 0 everywhere)!"
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title(title)
    if payload:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eis_nyquist_zoom.png'), dpi=150)
    plt.close(fig)

    # Bode: |Z| 和 -θ (横轴取对数, 纵轴线性)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    n_ph = 0
    for r in results:
        axes[0].semilogx(r['freq'], r['z_mag'], '-o', ms=3, label=f"ch{r['channel']}")
        # 相位仅显示物理合理区段 (-90..90), 避免 ±180 跳变
        theta = np.asarray(r['theta_deg'])
        mask = (theta > -90) & (theta < 90)
        n_ph += int(mask.sum())
        axes[1].semilogx(np.asarray(r['freq'])[mask], -theta[mask], '-o', ms=3,
                         label=f"ch{r['channel']}")
    if n_ph == 0 and results:
        # 全部频点的相位都在 -90° 以外 (Re Z < 0) —— 相位面板会整块空白。
        axes[1].text(0.5, 0.5,
                     "no physical phase point\n"
                     "(theta < -90 deg everywhere)",
                     transform=axes[1].transAxes, ha='center', va='center',
                     fontsize=10, color='#c0392b',
                     bbox=dict(boxstyle='round', fc='#fdecea', ec='#c0392b',
                               alpha=0.9))
        axes[1].set_title("Phase: no point within -90..90 deg", fontsize=10)
    axes[0].set_ylabel("|Z| [Ω]")
    axes[0].set_title("EIS Bode")
    if results:
        axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].set_ylabel("-θ [deg]")
    axes[1].set_xlabel("f [Hz]")
    axes[1].set_ylim(-5, 95)
    axes[1].grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eis_bode.png'), dpi=150)
    plt.close(fig)
    logger.info(f"Saved EIS plots to {out_dir}")
