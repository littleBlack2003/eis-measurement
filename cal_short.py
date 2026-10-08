"""
短路校准脚本 (与已有开路校准合并为完整校准表).

前提: 已完成开路校准 (data/calibration_open.json 存在).
执行短路校准后, 将开路+短路合并保存到 data/calibration.json,
供 EIS 测量做每通道开路/短路修正.
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

V_AC = float(config['measurement']['eis']['v_ac'])
V_DC = float(config['measurement']['eis']['v_dc'])
AVG = int(config['measurement']['calibration']['averaging'])
channels = config['measurement']['channels']

freqs = build_freq_axis(
    config['measurement']['eis']['freq_start'],
    config['measurement']['eis']['freq_stop'],
    config['measurement']['eis']['points_per_decade'],
)

OPEN_FILE = 'data/calibration_open.json'
CAL_FILE = 'data/calibration.json'

if not os.path.exists(OPEN_FILE):
    raise FileNotFoundError(f"未找到开路校准 {OPEN_FILE}, 请先运行 cal_open.py")

print("=" * 60)
print("短路校准")
print("=" * 60)
print(f"频率: {freqs[0]:.0f} - {freqs[-1]:.0f} Hz, {len(freqs)} 点")
print(f"交流幅值: {V_AC*1000:.0f} mVrms, 平均: {AVG} 次")
print(f"通道: {channels}")
print()

input("请将探针置于【短路】状态 (两探针直接接触/短接), 按回车开始...")

inst = HP4284A(config['instrument']['address'], timeout=30, function='ztd')

# 加载已有开路校准
open_store = CalibrationStore.load(OPEN_FILE)

# 创建完整校准表 (开路 + 短路)
full_store = CalibrationStore()

try:
    for ch in channels:
        print(f"\n通道 {ch}: 测量短路阻抗...", flush=True)
        z_short = np.array([inst.measure_complex_z(f, V_AC, V_DC, averaging=AVG)
                            for f in freqs])

        cal = ChannelCalibration(ch, freqs)
        # 从开路校准中取该通道的开路数据
        cal_open = open_store.get(ch)
        cal.set_open(cal_open.z_open)
        cal.set_short(z_short)
        full_store.add(cal)

        print(f"  完成, |Z_short| 中位数 = {np.median(np.abs(z_short)):.3e} ohm")
finally:
    inst.close()

os.makedirs('data', exist_ok=True)
full_store.save(CAL_FILE)
print(f"\n完整校准 (开路+短路) 已保存到 {CAL_FILE}")

# 保存短路频谱供检查
for ch in channels:
    cal = full_store.get(ch)
    df = pd.DataFrame({
        'freq [Hz]': freqs,
        'Z_short_real [ohm]': cal.z_short.real,
        'Z_short_imag [ohm]': cal.z_short.imag,
        '|Z_short| [ohm]': np.abs(cal.z_short),
    })
    df.to_csv(f'data/cal_short_ch{ch}.csv', index=False)
    print(f"短路频谱已保存: data/cal_short_ch{ch}.csv")

print()
print("*" * 60)
print("短路校准完成! 完整校准表已就绪。")
print("可以进行偏压 EIS 测试 (带每通道开路/短路修正)。")
print("*" * 60)
