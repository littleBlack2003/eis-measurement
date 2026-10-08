"""快速验证 4284A 连接与单点测量的测试脚本."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drivers.hp4284a import HP4284A

inst = HP4284A('GPIB0::17::INSTR', timeout=30, function='ztd')
print('IDN OK')

for f in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]:
    z = inst.measure_complex_z(f, 0.03, 0.0, averaging=3)
    theta = np.angle(z, deg=True)
    print(f'Z({f:>9.0f} Hz) = {z.real:12.4f} + j{z.imag:12.4f} ohm'
          f'  |Z|={abs(z):12.4f}  theta={theta:8.3f} deg')

inst.close()
print('Done')
