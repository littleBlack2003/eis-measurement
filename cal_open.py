"""
开路校准脚本 (仅执行开路校准).

前提: 探针悬空, 未接触器件 (开路状态).
完成后保存校准数据到 data/calibration_open.json, 并提示用户安装器件.
"""
import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drivers.hp4284a import HP4284A
from measurements.eis import build_freq_axis
from cal.calibration import CalibrationStore, ChannelCalibration

config = yaml.safe_load(open('eis_config.yaml', encoding='utf-8'))

# 开路校准参数 (与 EIS 扫描同一频率轴)
V_AC = float(config['measurement']['eis']['v_ac'])
V_DC = float(config['measurement']['eis']['v_dc'])
AVG = int(config['measurement']['calibration']['averaging'])
channels = config['measurement']['channels']

freqs = build_freq_axis(
    config['measurement']['eis']['freq_start'],
    config['measurement']['eis']['freq_stop'],
    config['measurement']['eis']['points_per_decade'],
)

print("=" * 60)
print("开路校准")
print("=" * 60)
print(f"频率: {freqs[0]:.0f} - {freqs[-1]:.0f} Hz, {len(freqs)} 点")
print(f"交流幅值: {V_AC*1000:.0f} mVrms, 平均: {AVG} 次")
print(f"通道: {channels}")
print()

input("请确认探针处于【开路】状态 (悬空, 不接触任何器件), 按回车开始...")

inst = HP4284A(config['instrument']['address'], timeout=30, function='ztd')
store = CalibrationStore()

try:
    for ch in channels:
        print(f"\n通道 {ch}: 测量开路阻抗...", flush=True)
        z_open = np.array([inst.measure_complex_z(f, V_AC, V_DC, averaging=AVG)
                           for f in freqs])
        cal = ChannelCalibration(ch, freqs)
        cal.set_open(z_open)
        store.add(cal)
        print(f"  完成, |Z_open| 中位数 = {np.median(np.abs(z_open)):.3e} ohm")
finally:
    inst.close()

os.makedirs('data', exist_ok=True)
store.save('data/calibration_open.json')
print("\n开路校准完成, 已保存到 data/calibration_open.json")

# 同时保存开路校准的频谱供检查
for ch in channels:
    cal = store.get(ch)
    df = pd.DataFrame({
        'freq [Hz]': freqs,
        'Z_open_real [ohm]': cal.z_open.real,
        'Z_open_imag [ohm]': cal.z_open.imag,
        '|Z_open| [ohm]': np.abs(cal.z_open),
    })
    df.to_csv(f'data/cal_open_ch{ch}.csv', index=False)
    print(f"开路频谱已保存: data/cal_open_ch{ch}.csv")

print()
print("*" * 60)
print("开路校准完成!")
print("请现在安装/放置被测器件到探针下。")
print("安装完成后告知我, 我将继续短路校准并开始偏压测试。")
print("*" * 60)
