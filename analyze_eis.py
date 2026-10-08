"""重测数据分析: 聚焦高频可用区段, 尝试等效电路拟合."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv('data/eis_ch0.csv')
freq = df['freq [Hz]'].values
z = df['Z_real [ohm]'].values + 1j * df['Z_imag [ohm]'].values

print("=" * 60)
print("a-Si 电池重测数据分析 (高频可用区段)")
print("=" * 60)

# 高频段 (>=10 kHz): 检查是否为纯电容特征
hi = freq >= 1e4
f_hi, z_hi = freq[hi], z[hi]

# 估算串联电阻 Rs: 高频极限时 Re(Z) -> Rs
print(f"\n【高频段 (10 kHz - 1 MHz)】")
print(f"Re(Z) 范围: {z_hi.real.min():.1f} - {z_hi.real.max():.1f} ohm")
print(f"Re(Z) 中位数: {np.median(z_hi.real):.1f} ohm  (≈ 串联电阻 Rs)")

# 估算并联电容: C = -1/(omega * Z'')
omega = 2 * np.pi * f_hi
c_est = -1.0 / (omega * z_hi.imag)
print(f"C 估算: 中位数 {np.median(c_est):.3e} F, 均值 {np.mean(c_est):.3e} F")
print(f"C 标准差/均值: {np.std(c_est)/np.mean(c_est)*100:.1f}%  (越小越接近纯电容)")
print(f"相位范围: {np.degrees(np.angle(z_hi)).min():.1f} - {np.degrees(np.angle(z_hi)).max():.1f} deg")

# 相位接近 -90 度 -> 电容主导
phase_hi = np.degrees(np.angle(z_hi))
print(f"|相位+90| 平均值: {np.mean(np.abs(phase_hi + 90)):.1f} deg")

# 检查 C 随频率是否稳定 (理想电容应恒定)
print("\nC 按频段:")
for lo, hi2 in [(1e4, 3e4), (3e4, 1e5), (1e5, 3e5), (3e5, 1e6)]:
    m = (f_hi >= lo) & (f_hi < hi2)
    if m.sum():
        print(f"  {lo:>8.0f}-{hi2:>8.0f} Hz: C = {np.median(c_est[m]):.3e} F, "
              f"Rs = {np.median(z_hi.real[m]):.1f} ohm")

# 中频段 (1-10 kHz): 过渡区
mid = (freq >= 1e3) & (freq < 1e4)
print(f"\n【中频段 (1-10 kHz)】: Re(Z) 范围 {z[mid].real.min():.0f} - {z[mid].real.max():.0f} ohm")
print(f"  |Z| 范围 {np.abs(z[mid]).min():.0f} - {np.abs(z[mid]).max():.0f} ohm")

# 低频段问题
lo = freq < 1e3
print(f"\n【低频段 (<1 kHz)】: Re(Z)>0 比例 {100*np.mean(z[lo].real > 0):.0f}%")
print(f"  |Z| 范围 {np.abs(z[lo]).min():.0f} - {np.abs(z[lo]).max():.0f} ohm")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
