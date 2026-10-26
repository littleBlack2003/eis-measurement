"""Fit the re-run scan with a series-inductance term.

Why: the 17:24 scan shows a series resonance. Phase runs -38 deg -> -91 deg ->
-50 deg between 440 kHz and 1 MHz and Re Z actually goes slightly negative
(-0.93 ohm at 628 kHz, -0.81 ohm at 705 kHz). That is a parasitic series
inductance (leads / fixture), no resonance in Rs(RpC). Forcing Rs(RpC) over the
whole band pushes Rs onto its lower bound (0.001 ohm), which is meaningless.

This compares three treatments of the same data:
  1. Rs(RpC) over the whole band          -> what the app reported
  2. Rs(RpC) restricted below resonance   -> keeps Rs+Rp+C but throws the
                                             inductive tail away
  3. Rs + jwL + (Rp||C) over the whole band -> physically complete
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analysis.fitting import fit_equivalent_circuit

SENTINEL = 1.0e30


def model_rsl_rpc(freq, rs, L, rp, c):
    w = 2.0 * np.pi * np.asarray(freq)
    return rs + 1j * w * L + 1.0 / (1.0 / rp + 1j * w * c)


def fit_rsl(freq, z, max_nfev=20000):
    f = np.asarray(freq, float)
    zf = np.asarray(z, complex)
    rs0 = max(float(np.median(zf.real[-5:])), 1.0)
    rp0 = max(float(np.median(zf.real[:5])) - rs0, rs0 * 10)
    f_apex = np.sqrt(f[0] * f[-1])
    c0 = 1.0 / (2 * np.pi * rp0 * f_apex)
    # L from the observed resonance: the most negative phase sets f0,
    # f0 = 1/(2*pi*sqrt(L*C))
    theta = np.angle(zf, deg=True)
    f0 = float(f[np.argmin(theta)])
    L0 = 1.0 / ((2 * np.pi * f0) ** 2 * c0)
    p0 = [rs0, L0, rp0, c0]
    lower = [0.0, 0.0, 0.0, 1e-15]
    upper = [1e4, 1e-1, 1e12, 1e-3]
    w = np.sqrt(np.abs(zf))

    def residual(p):
        diff = (model_rsl_rpc(f, *p) - zf) / w
        return np.concatenate([diff.real, diff.imag])

    res = least_squares(residual, p0, bounds=(lower, upper), max_nfev=max_nfev)
    z_model = model_rsl_rpc(f, *res.x)
    norm_err = (np.sqrt(np.mean(np.abs(z_model - zf) ** 2))
                / np.sqrt(np.mean(np.abs(zf) ** 2)))
    try:
        jac = res.jac
        dof = max(len(zf) * 2 - len(res.x), 1)
        cov = (np.sum(res.fun ** 2) / dof) * np.linalg.inv(jac.T @ jac)
        perr = np.sqrt(np.abs(np.diag(cov)))
    except Exception:
        perr = np.full(len(res.x), np.nan)
    names = ['Rs [ohm]', 'L [H]', 'Rp [ohm]', 'C [F]']
    out = dict(zip(names, res.x))
    out['_err'] = dict(zip(names, perr))
    out['norm_error'] = norm_err
    out['n_points'] = len(f)
    out['f_resonance_Hz'] = 1.0 / (2 * np.pi * np.sqrt(res.x[1] * res.x[3]))
    return out


def main():
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path('data/eis_20261014_172443_bias+0p70V/eis_ch0.csv')
    df = pd.read_csv(csv)
    df = df[np.abs(df['|Z| [ohm]']) < SENTINEL].reset_index(drop=True)
    f = df['freq [Hz]'].values.astype(float)
    z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values
    print(f"{csv.parent.name}: {len(f)} points\n")

    print("[1] Rs(RpC), full band")
    r1 = fit_equivalent_circuit(f, z, use_cpe=False, freq_min=0)
    print(f"    Rs={r1['Rs']:.4g}  Rp={r1['Rp']:.4g}  C={r1['C']:.4g}  "
          f"err={r1['norm_error']*100:.2f}%")

    # resonance = where phase is most negative
    theta = np.angle(z, deg=True)
    f_res = f[np.argmin(theta)]
    cut = f_res * 0.6
    m = f <= cut
    print(f"\n[2] Rs(RpC), f <= {cut:.4g} Hz (below resonance, n={m.sum()})")
    r2 = fit_equivalent_circuit(f[m], z[m], use_cpe=False, freq_min=0)
    print(f"    Rs={r2['Rs']:.4g}  Rp={r2['Rp']:.4g}  C={r2['C']:.4g}  "
          f"err={r2['norm_error']*100:.2f}%")
    print(f"    tau={r2['tau']:.4g} s, f_apex={r2['f_apex']:.4g} Hz")

    print(f"\n[3] Rs+jwL+(Rp||C), full band")
    r3 = fit_rsl(f, z)
    for k in ('Rs [ohm]', 'L [H]', 'Rp [ohm]', 'C [F]'):
        print(f"    {k:<10s} = {r3[k]:.6g} +/- {r3['_err'][k]:.3g}")
    print(f"    err={r3['norm_error']*100:.2f}%")
    print(f"    series resonance at {r3['f_resonance_Hz']:.4g} Hz "
          f"(observed phase minimum at {f_res:.4g} Hz)")

    print("\n[4] CPE variant of [2] (is the arc still ideal?)")
    r4 = fit_equivalent_circuit(f[m], z[m], use_cpe=True, freq_min=0)
    print(f"    Rs={r4['Rs']:.4g}  Rp={r4['Rp']:.4g}  Q={r4['Q']:.4g}  "
          f"n={r4['n']:.4g}  err={r4['norm_error']*100:.2f}%")
    return 0


if __name__ == '__main__':
    sys.exit(main())
