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
    })
    fname = os.path.join(out_dir, f"eis_ch{ch}.csv")
    df.to_csv(fname, index=False)
    logger.info(f"Saved EIS channel {ch} to {fname}")


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
    for r in results:
        zr, zi = r['z'].real, -r['z'].imag
        mask = (zr > 0) & (zi > 0)
        if mask.any():
            ax.loglog(zr[mask], zi[mask], 'o', ms=4,
                      label=f"ch{r['channel']}")
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title("EIS Nyquist (log-log, full range)")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eis_nyquist.png'), dpi=150)
    plt.close(fig)

    # Nyquist zoom (高频可靠区段, 线性轴以显示 RC 弧)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for r in results:
        f = r['freq']
        zr, zi = r['z'].real, -r['z'].imag
        # 仅显示 f >= 1 kHz 的可靠区段 (排除 MΩ 级低频离群点)
        mask = (f >= 1e3) & (zr > 0) & (zi > 0)
        if mask.any():
            ax.plot(zr[mask], zi[mask], 'o-', ms=4, lw=1,
                    label=f"ch{r['channel']}")
    ax.set_xlabel("Z' [Ω]")
    ax.set_ylabel("-Z'' [Ω]")
    ax.set_title("EIS Nyquist (f >= 1 kHz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eis_nyquist_zoom.png'), dpi=150)
    plt.close(fig)

    # Bode: |Z| 和 -θ (横轴取对数, 纵轴线性)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for r in results:
        axes[0].semilogx(r['freq'], r['z_mag'], '-o', ms=3, label=f"ch{r['channel']}")
        # 相位仅显示物理合理区段 (-90..90), 避免 ±180 跳变
        theta = np.asarray(r['theta_deg'])
        mask = (theta > -90) & (theta < 90)
        axes[1].semilogx(np.asarray(r['freq'])[mask], -theta[mask], '-o', ms=3,
                         label=f"ch{r['channel']}")
    axes[0].set_ylabel("|Z| [Ω]")
    axes[0].set_title("EIS Bode")
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
