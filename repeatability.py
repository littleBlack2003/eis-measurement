"""Quantify scan-to-scan repeatability between two runs on the *same* device.

Usage:
    python repeatability.py A.csv B.csv [--labels tagA tagB] [--out file.png]

Matching is done on the exact frequency column (the axis is deterministic:
20 pts/decade over 20 Hz..1 MHz), so a plain merge on frequency is enough.
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SENTINEL_MAG = 1e30


def load(path):
    df = pd.read_csv(path)
    df = df[np.abs(df['|Z| [ohm]']) < SENTINEL_MAG]
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    return df['freq [Hz]'].values.astype(float), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_a')
    ap.add_argument('csv_b')
    ap.add_argument('--labels', nargs=2, default=['A', 'B'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    fa, za = load(args.csv_a)
    fb, zb = load(args.csv_b)

    # match on frequency (rounded to mHz to defeat float formatting noise)
    da = pd.DataFrame({'freq': np.round(fa, 3), 'a': za})
    db = pd.DataFrame({'freq': np.round(fb, 3), 'b': zb})
    m = da.merge(db, on='freq', how='inner').sort_values('freq')
    if m.empty:
        log.error('no overlapping frequencies between the two files')
        return 1

    f = m['freq'].values
    a = m['a'].values
    b = m['b'].values
    mag_a, mag_b = np.abs(a), np.abs(b)
    th_a, th_b = np.angle(a, deg=True), np.angle(b, deg=True)

    rel_mag = np.abs(mag_b - mag_a) / mag_a * 100.0           # % deviation in |Z|
    d_th = np.abs(th_b - th_a)                                # deg deviation in phase

    # complex-plane distance, normalised by |Z|
    rel_z = np.abs(b - a) / mag_a * 100.0

    print()
    print('=== repeatability: %s vs %s ===' % (args.labels[0], args.labels[1]))
    print('matched frequency points: %d  (%.1f Hz .. %.4g Hz)' % (len(f), f.min(), f.max()))
    print()
    print('%-14s %8s %8s %8s %8s' % ('quantity', 'median', 'mean', 'p95', 'max'))
    for name, v in (('|Z| dev [%]', rel_mag), ('theta dev [deg]', d_th),
                    ('complex dev [%]', rel_z)):
        print('%-14s %8.3f %8.3f %8.3f %8.3f' % (
            name, np.median(v), v.mean(), np.percentile(v, 95), v.max()))

    # split by decade to see where the scatter lives
    print()
    print('%-16s %6s %10s %10s %10s' % ('band', 'n', '|Z| med', 'dev med', 'dev max'))
    edges = [(0, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 2e6)]
    for lo, hi in edges:
        s = (f >= lo) & (f < hi)
        if not s.any():
            continue
        print('%-16s %6d %10.4g %9.3f%% %9.3f%%' % (
            '%.0f-%.0f Hz' % (lo, hi), s.sum(), np.median(mag_a[s]),
            np.median(rel_mag[s]), rel_mag[s].max()))

    # phase/real-part sanity: did either run go non-physical?
    print()
    for tag, z in ((args.labels[0], a), (args.labels[1], b)):
        nneg = int((z.real <= 0).sum())
        print('%-12s Re Z <= 0: %2d/%d pts   theta range %7.2f .. %7.2f deg' % (
            tag, nneg, len(z), np.angle(z, deg=True).min(), np.angle(z, deg=True).max()))

    # fitted-parameter comparison if both summaries exist
    print()
    for tag, path in ((args.labels[0], args.csv_a), (args.labels[1], args.csv_b)):
        js = os.path.join(os.path.dirname(path), 'fit_summary.json')
        if os.path.exists(js):
            import json
            d = json.loads(open(js, encoding='utf-8').read())
            print('%-12s model=%-11s err=%.3f%%  Rp=%.1f ohm  Rs=%.2f ohm' % (
                tag, d.get('model', '?'), 100 * d.get('norm_error', float('nan')),
                d.get('Rp', float('nan')), d.get('Rs', float('nan'))))

    # ---- plot ----
    out = args.out or os.path.join(os.path.dirname(args.csv_b), 'repeatability.png')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ax = axes[0]
    ax.plot(a.real, -a.imag, 'o-', ms=4, lw=1, label=args.labels[0])
    ax.plot(b.real, -b.imag, 's--', ms=4, lw=1, label=args.labels[1])
    ax.set_xlabel("Z' [ohm]"); ax.set_ylabel("-Z'' [ohm]")
    ax.set_title('Nyquist overlay (same device)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'datalim')

    ax = axes[1]
    ax.semilogx(f, rel_mag, 'o-', ms=4, lw=1, color='#c23b28')
    ax.axhline(np.median(rel_mag), ls='--', lw=1, color='#666',
               label='median %.2f%%' % np.median(rel_mag))
    ax.set_xlabel('f [Hz]'); ax.set_ylabel('|Z| deviation [%]')
    ax.set_title('Magnitude repeatability'); ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[2]
    ax.semilogx(f, d_th, 'o-', ms=4, lw=1, color='#1c5ca8')
    ax.axhline(np.median(d_th), ls='--', lw=1, color='#666',
               label='median %.3f deg' % np.median(d_th))
    ax.set_xlabel('f [Hz]'); ax.set_ylabel('phase deviation [deg]')
    ax.set_title('Phase repeatability'); ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('%s vs %s  -  scan-to-scan repeatability (0.7 V DC bias)' %
                 (args.labels[0], args.labels[1]), y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info('plot -> %s', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
