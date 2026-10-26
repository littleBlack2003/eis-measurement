"""从已有 CSV 重新生成 EIS 图 (无需重新测量).

用法:
    python replot.py                                  # 默认 data/
    python replot.py data/eis_20261014_161543_bias+0p70V
    python replot.py <目录> --only eis_ch0.csv
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_io import plot_eis_all


def load_results(out_dir: str, only=None):
    results = []
    for fname in sorted(os.listdir(out_dir)):
        # 严格匹配 eis_ch<数字>.csv: 跳过 *_raw.csv (备份) 与 *_live.csv
        # (扫描逐点落盘的中间产物)
        m = re.fullmatch(r'eis_ch(\d+)\.csv', fname)
        if not m:
            continue
        if only and fname not in only:
            continue
        df = pd.read_csv(os.path.join(out_dir, fname))
        ch = int(m.group(1))
        z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
        z_raw = z
        if {'Z_raw_real [ohm]', 'Z_raw_imag [ohm]'} <= set(df.columns):
            z_raw = (df['Z_raw_real [ohm]'].values
                     + 1j * df['Z_raw_imag [ohm]'].values)
        results.append({
            'channel': ch,
            'freq': df['freq [Hz]'].values,
            'z': z,
            'z_raw': z_raw,
            'z_mag': np.abs(z),
            'theta_deg': df['theta [deg]'].values,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir', nargs='?', default='data')
    ap.add_argument('--only', nargs='*', default=None,
                    help='只画指定的 CSV 文件名')
    args = ap.parse_args()

    out_dir = args.out_dir
    results = load_results(out_dir, args.only)
    if not results:
        print(f"no eis_ch*.csv found in {out_dir}")
        return 1
    for r in results:
        print(f"ch{r['channel']}: {len(r['freq'])} pts, "
              f"{r['freq'].min():.4g} – {r['freq'].max():.4g} Hz")
    plot_eis_all(results, out_dir)
    print("Plots regenerated:",
          sorted(f for f in os.listdir(out_dir) if f.endswith('.png')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
