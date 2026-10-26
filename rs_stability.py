"""Why is Rs so unstable?  Drop the highest-frequency points one at a time and
watch how each fitted parameter moves.  If Rs swings wildly while Rp/Q/n stay
put, then Rs is carried by a handful of points and cannot be reported as a
device parameter.

Usage:
    python rs_stability.py <csv> [--max-drop 12]
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from analysis.fitting import fit_equivalent_circuit   # noqa: E402
from fit_with_L import fit_rsl, model_rsl_rpc         # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(message)s')
SENTINEL_MAG = 1e30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--max-drop', type=int, default=12)
    ap.add_argument('--with-L', action='store_true',
                    help='use Rs + jwL + (Rp||C) instead of Rs(RpCPE)')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[np.abs(df['|Z| [ohm]']) < SENTINEL_MAG]
    df = df.sort_values('freq [Hz]').reset_index(drop=True)
    f = df['freq [Hz]'].values.astype(float)
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

    print('file: %s' % args.csv)
    print('points: %d   Re Z <= 0: %d' % (len(f), int((z.real <= 0).sum())))
    print('model : %s' % ('Rs + jwL + (Rp||C)' if args.with_L else 'Rs(RpCPE)'))
    print()
    if args.with_L:
        print('%-6s %-10s %9s %10s %10s %11s %8s' % (
            'n_drop', 'f_max', 'Rs[ohm]', 'L[H]', 'Rp[ohm]', 'C[F]', 'err[%]'))
        print('-' * 68)
    else:
        print('%-6s %-10s %9s %10s %11s %8s %8s' % (
            'n_drop', 'f_max', 'Rs[ohm]', 'Rp[ohm]', 'Q[F]', 'n', 'err[%]'))
        print('-' * 68)

    rs, rp, qq, nn, err, ll = [], [], [], [], [], []
    for k in range(0, args.max_drop + 1):
        if len(f) - k < 12:
            break
        fk, zk = f[:len(f) - k], z[:len(z) - k]
        if args.with_L:
            r = fit_rsl(fk, zk)
            rs.append(r['Rs [ohm]']); ll.append(r['L [H]'])
            rp.append(r['Rp [ohm]']); qq.append(r['C [F]'])
            nn.append(float('nan')); err.append(100 * r['norm_error'])
            print('%-6d %-10.4g %9.4g %10.4g %10.5g %11.5g %8.3f' % (
                k, fk.max(), r['Rs [ohm]'], r['L [H]'], r['Rp [ohm]'],
                r['C [F]'], 100 * r['norm_error']))
        else:
            r = fit_equivalent_circuit(fk, zk, use_cpe=True, freq_min=0)
            rs.append(r['Rs']); rp.append(r['Rp']); qq.append(r['Q'])
            nn.append(r['n']); err.append(100 * r['norm_error'])
            print('%-6d %-10.4g %9.4g %10.5g %11.5g %8.5f %8.3f' % (
                k, fk.max(), r['Rs'], r['Rp'], r['Q'], r['n'],
                100 * r['norm_error']))

    print()
    print('Parameter drift over the sweep (max / min):')
    series = [('Rs', rs), ('L', ll), ('Rp', rp)] if args.with_L else \
             [('Rs', rs), ('Rp', rp), ('Q', qq), ('n', nn)]
    series.append(('err', err))
    for name, v in series:
        v = np.asarray(v, dtype=float)
        lo, hi = v.min(), v.max()
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        ratio = (hi / lo) if lo > 0 else float('inf')
        print('  %-4s  min=%-12.5g max=%-12.5g  spread=%.1fx  (%.2f%% of median)' % (
            name, lo, hi, ratio, 100 * (hi - lo) / np.median(v)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
