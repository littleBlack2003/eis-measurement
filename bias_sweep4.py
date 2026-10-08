"""
偏压依赖 EIS 扫描 (第四次): 固定 10 mV 幅值, DC 偏压 +1.5 到 -1.5 V (0.2 V 步长).

每个偏压点执行一次全频段 (20 Hz - 1 MHz) 复数阻抗扫描, 完成后立即生成报告.
输出: 偏压扫描4/data/bias_<Vdc>V.csv + 偏压扫描4/偏压报告/Bias_<Vdc>V_报告.pdf
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drivers.hp4284a import HP4284A
from measurements.eis import build_freq_axis
from analysis.bias_report import generate_bias_report

config = yaml.safe_load(open('eis_config.yaml', encoding='utf-8'))

V_AC = 0.010          # 固定 10 mVrms
V_DC_START = 1.5      # 从 +1.5 V 开始
V_DC_STOP = -1.5      # 到 -1.5 V
V_DC_STEP = 0.2
AVG = 3

# 输出到子文件夹4
DATA_DIR = os.path.join('偏压扫描4', 'data')
REPORT_DIR = os.path.join('偏压扫描4', '偏压报告')

freqs = build_freq_axis(
    config['measurement']['eis']['freq_start'],
    config['measurement']['eis']['freq_stop'],
    config['measurement']['eis']['points_per_decade'],
)

# +1.5 -> -1.5
vdc_list = np.arange(V_DC_START, V_DC_STOP - 1e-9, -V_DC_STEP)
print(f"偏压点: {len(vdc_list)} 个 (+1.5 到 -1.5 V, 步长 0.2 V)")
print(f"交流幅值: {V_AC*1000:.0f} mVrms")
print(f"每偏压频点: {len(freqs)} 个, 预计总耗时 ~{len(vdc_list)*len(freqs)*1.4/60:.0f} 分钟")
print(f"输出: {DATA_DIR}/ + {REPORT_DIR}/")

os.makedirs(DATA_DIR, exist_ok=True)
inst = HP4284A(config['instrument']['address'], timeout=30, function='ztd')

try:
    for vdc in vdc_list:
        label = f"{vdc:+.1f}V".replace('+', '').replace('-', 'n').replace('.', 'p')
        fname = os.path.join(DATA_DIR, f'bias_{label}.csv')
        # 已存在则跳过 (支持断点续测), 但已存在也补报告
        if os.path.exists(fname):
            print(f"  Vdc={vdc:+.1f} V: 数据已存在, 补生成报告", flush=True)
            generate_bias_report(fname, report_dir=REPORT_DIR)
            continue
        print(f"  Vdc={vdc:+.1f} V ...", flush=True)
        z = np.array([inst.measure_complex_z(f, V_AC, vdc, averaging=AVG)
                      for f in freqs])
        df = pd.DataFrame({
            'freq [Hz]': freqs,
            'Z_real [ohm]': z.real,
            'Z_imag [ohm]': z.imag,
            '|Z| [ohm]': np.abs(z),
            'theta [deg]': np.degrees(np.angle(z)),
        })
        df.to_csv(fname, index=False)
        print(f"    -> {fname}", flush=True)
        # 每个偏压完成后立即出报告
        generate_bias_report(fname, report_dir=REPORT_DIR)
finally:
    inst.close()

print("\n偏压扫描完成")
