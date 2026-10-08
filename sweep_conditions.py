"""
系统化扫描 EIS 测量条件, 寻找最适合的参数组合.

扫描维度:
  - 交流幅值: 10 / 20 / 30 / 50 / 100 mVrms (平均次数=3)
  - 平均次数: 1 / 3 / 10 (幅值固定 30 mVrms)

每个条件保存独立 CSV 到 data/sweep_<label>.csv, 并输出质量指标.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drivers.hp4284a import HP4284A
from measurements.eis import build_freq_axis

config = yaml.safe_load(open('eis_config.yaml', encoding='utf-8'))

# 测量条件矩阵: (label, v_ac [Vrms], averaging)
conditions = [
    ('10mV_avg3',  0.010, 3),
    ('20mV_avg3',  0.020, 3),
    ('30mV_avg3',  0.030, 3),
    ('50mV_avg3',  0.050, 3),
    ('100mV_avg3', 0.100, 3),
    ('30mV_avg1',  0.030, 1),
    ('30mV_avg10', 0.030, 10),
]

freqs = build_freq_axis(
    config['measurement']['eis']['freq_start'],
    config['measurement']['eis']['freq_stop'],
    config['measurement']['eis']['points_per_decade'],
)

os.makedirs('data', exist_ok=True)
inst = HP4284A(config['instrument']['address'], timeout=30, function='ztd')

try:
    for label, vac, avg in conditions:
        print(f"\n=== 测量: {label} (Vac={vac*1000:.0f} mV, avg={avg}) ===")
        z = np.array([inst.measure_complex_z(f, vac, 0.0, averaging=avg)
                      for f in freqs])
        df = pd.DataFrame({
            'freq [Hz]': freqs,
            'Z_real [ohm]': z.real,
            'Z_imag [ohm]': z.imag,
            '|Z| [ohm]': np.abs(z),
            'theta [deg]': np.degrees(np.angle(z)),
        })
        df.to_csv(f'data/sweep_{label}.csv', index=False)
        print(f"  完成, 保存到 data/sweep_{label}.csv")
        time.sleep(1.0)  # 条件间稳定
finally:
    inst.close()

print("\n全部条件测量完成")
